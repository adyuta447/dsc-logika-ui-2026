"""Alternative model family for KS ranking.

Uses manually selected numeric features and two sklearn tree families:
ExtraTrees and HistGradientBoosting. No external data, no AutoML, no edits
outside agas/. Feature selection is fitted inside each training fold.
"""
import sys
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline


def rank01(x):
    return rankdata(x) / len(x)


def metrics(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--top-features", type=int, default=120)
    ap.add_argument("--out-prefix", default="v59alt")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(__import__("shared.bootstrap", fromlist=["x"]).add_shared(__file__)))
    import paths  # noqa: E402
    root = paths.ROOT
    os.chdir(root)
    out = here.parent / "output"
    out.mkdir(exist_ok=True)
    log_path = out / f"train_log_{args.out_prefix}.txt"
    if log_path.exists():
        log_path.unlink()

    def log(s):
        print(s, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    start = time.time()
    cache = paths.FEATURES_CACHE
    f = pd.read_pickle(cache) if cache.exists() else None
    if f is None:
        import sys
        sys.path.insert(0, str(root))
        import features
        f = features.build()
        f.to_pickle(cache)

    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    y = tr.target.astype(int).to_numpy()
    ids = ["application_id", "customer_id", "target"]
    X = tr.drop(columns=ids)
    Xt = te.drop(columns=ids)
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric].replace([np.inf, -np.inf], np.nan)
    Xt = Xt[numeric].replace([np.inf, -np.inf], np.nan)
    log(f"rows train={len(X)} test={len(Xt)} numeric_features={len(numeric)}")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    oof_extra = np.zeros(len(X))
    oof_hist = np.zeros(len(X))
    pred_extra = np.zeros(len(Xt))
    pred_hist = np.zeros(len(Xt))
    selected = []

    for k, (a, b) in enumerate(skf.split(X, y)):
        xa, xb = X.iloc[a], X.iloc[b]
        med = xa.median()
        xa_fill = xa.fillna(med)
        xb_fill = xb.fillna(med)
        xt_fill = Xt.fillna(med)

        # Manual, fold-safe feature selection: absolute point-biserial-like
        # separation, then retain diverse high-signal columns.
        scores = {}
        for c in numeric:
            s = xa_fill[c]
            pos = s[y[a] == 1].mean()
            neg = s[y[a] == 0].mean()
            sd = s.std()
            scores[c] = abs(pos - neg) / (sd + 1e-9)
        cols = list(pd.Series(scores).sort_values(ascending=False).head(args.top_features).index)
        selected.append(cols)
        log(f"fold={k} selected={len(cols)}")

        extra = make_pipeline(
            ExtraTreesClassifier(
                n_estimators=700,
                max_features=0.55,
                min_samples_leaf=18,
                max_depth=None,
                class_weight=None,
                random_state=args.seed + k,
                n_jobs=-1,
            )
        )
        hist = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=700,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=120,
                l2_regularization=8.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=80,
                random_state=args.seed + 100 + k,
            ),
        )
        extra.fit(xa_fill[cols], y[a])
        hist.fit(xa[cols], y[a])
        oof_extra[b] = extra.predict_proba(xb_fill[cols])[:, 1]
        pred_extra += extra.predict_proba(xt_fill[cols])[:, 1] / args.folds
        oof_hist[b] = hist.predict_proba(xb[cols])[:, 1]
        pred_hist += hist.predict_proba(Xt[cols])[:, 1] / args.folds
        log(f"fold={k} extra={metrics(y[b], oof_extra[b]):s}" if False else
            f"fold={k} extra_auc={roc_auc_score(y[b], oof_extra[b]):.5f} "
            f"hist_auc={roc_auc_score(y[b], oof_hist[b]):.5f}")

    report = {
        "extra_trees": metrics(y, oof_extra),
        "hist_gradient_boosting": metrics(y, oof_hist),
    }
    blend_oof = 0.65 * rank01(oof_extra) + 0.35 * rank01(oof_hist)
    blend_pred = 0.65 * rank01(pred_extra) + 0.35 * rank01(pred_hist)
    report["rank_blend_65_extra_35_hist"] = metrics(y, blend_oof)
    log(json.dumps(report, indent=2))

    test_ids = te.application_id.to_numpy()
    files = {}
    for name, pred in [
        ("extra_trees", pred_extra),
        ("hist_gradient_boosting", pred_hist),
        ("rank_blend_65_extra_35_hist", blend_pred),
    ]:
        path = out / f"submission_{args.out_prefix}_{name}.csv"
        pd.DataFrame({"application_id": test_ids, "default_probability": rank01(pred)}).to_csv(path, index=False)
        files[name] = str(path)

    nb_dir = here.parent / "notebook"
    nb_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = nb_dir / f"notebook_{args.out_prefix}.ipynb"
    summary = {
        "params": vars(args),
        "numeric_features": len(numeric),
        "selected_features_per_fold": [len(x) for x in selected],
        "metrics": report,
        "files": files,
        "elapsed_minutes": (time.time() - start) / 60,
        "notebook": str(notebook_path),
    }
    with open(out / f"summary_{args.out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"summary={out / f'summary_{args.out_prefix}.json'}")


if __name__ == "__main__":
    main()
