"""XGBoost diversity leg for v71.

Same lean columns and fold-safe factor columns as the LightGBM/CatBoost legs.
This is a third tree geometry; it is useful only if it decorrelates the blend.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"
NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(OUT / "features_v71.pkl"))
    ap.add_argument("--drops", default="pn_,rf_")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", default="42")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--use-factor", action="store_true")
    ap.add_argument("--out-prefix", default="xgb_v71")
    args = ap.parse_args()
    t0 = time.time()
    drops = tuple(s.strip() for s in args.drops.split(",") if s.strip())
    f = pd.read_pickle(args.features)
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X = tr.drop(columns=ids)
    Xte = te.drop(columns=ids)
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    lean = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)
            and not c.startswith(drops)]
    y = tr.target.astype(int).values
    log(f"features={Path(args.features).name} rows={len(X)} lean={len(lean)}")
    import features as ft
    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([tr.application_id.values, te.application_id.values])].values
    y_app = app.target.values
    n_tr = len(tr)
    fM, fE = ft.factor_inputs(app)

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    oof_by_fs, test_by_fs, results = {}, {}, {}
    for fs in fold_seeds:
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=fs).split(X, y))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        factor_cols = list(fac_k[0].columns)
        cols = lean + (factor_cols if args.use_factor else [])
        log(f"  train columns={len(cols)} factor={len(factor_cols) if args.use_factor else 0}")
        oof_sum = np.zeros(n_tr)
        test_sum = np.zeros(len(Xte))
        seed_aucs = []
        for sd in seeds:
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk = pd.concat([X.reset_index(drop=True),
                                fac_k[k].iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1)
                xt = pd.concat([Xte.reset_index(drop=True),
                                fac_k[k].iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1)
                m = XGBClassifier(
                    n_estimators=4000, learning_rate=0.03, max_depth=6,
                    min_child_weight=20, subsample=0.80, colsample_bytree=0.30,
                    reg_lambda=20.0, reg_alpha=0.0, tree_method="hist",
                    max_bin=128, eval_metric="auc", early_stopping_rounds=200,
                    n_jobs=10, random_state=sd, verbosity=0,
                )
                m.fit(xk.iloc[a][cols], y[a],
                      eval_set=[(xk.iloc[b][cols], y[b])], verbose=False)
                oof[b] = m.predict_proba(xk.iloc[b][cols])[:, 1]
                pred += m.predict_proba(xt[cols])[:, 1] / len(folds)
                log(f"  fs={fs} sd={sd} fold={k} best={m.best_iteration} "
                    f"auc={roc_auc_score(y[b], oof[b]):.6f}")
            auc = roc_auc_score(y, oof)
            seed_aucs.append(float(auc))
            log(f"  fs={fs} sd={sd} auc={auc:.6f} ks={ks(y, oof):.6f}")
            oof_sum += oof / len(seeds)
            test_sum += pred / len(seeds)
        auc = roc_auc_score(y, oof_sum)
        results[str(fs)] = {"auc": float(auc), "ks": ks(y, oof_sum),
                            "auc_seed_mean": float(np.mean(seed_aucs)),
                            "auc_seed_sd": float(np.std(seed_aucs))}
        oof_by_fs[f"oof_{fs}"] = oof_sum
        test_by_fs[f"test_{fs}"] = test_sum
        log(f"fs={fs} auc={auc:.6f} ks={ks(y, oof_sum):.6f}")
        np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                            train_ids=tr.application_id.values,
                            test_ids=te.application_id.values, y=y,
                            **oof_by_fs, **test_by_fs)
    vals = [results[str(fs)]["auc"] for fs in fold_seeds]
    ksv = [results[str(fs)]["ks"] for fs in fold_seeds]
    summary = {
        "features": str(args.features), "drops": drops, "folds": args.folds,
        "fold_seeds": fold_seeds, "seeds": seeds, "use_factor": bool(args.use_factor),
        "n_lean": len(lean), "mean_auc": float(np.mean(vals)),
        "sd_auc": float(np.std(vals)), "mean_ks": float(np.mean(ksv)),
        "per_fold_seed": results, "minutes": (time.time() - t0) / 60.0,
    }
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"SUMMARY auc={summary['mean_auc']:.6f} sd={summary['sd_auc']:.6f} "
        f"ks={summary['mean_ks']:.6f} in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
