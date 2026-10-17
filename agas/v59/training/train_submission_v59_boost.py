"""Alternative boosting geometry: XGBoost lossguide + CatBoost Lossguide.

This is separate from the existing depth-wise v57/v58 recipes. All feature
selection is manual and fold-safe. The script only creates files under agas/.
"""
import sys
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rank01(x):
    return rankdata(x) / len(x)


def score(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--top-features", type=int, default=180)
    ap.add_argument("--out-prefix", default="v59boost")
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
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    start = time.time()
    f = pd.read_pickle(paths.FEATURES_CACHE)
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    y = tr.target.astype(int).to_numpy()
    drop = ["application_id", "customer_id", "target"]
    X = tr.drop(columns=drop)
    Xt = te.drop(columns=drop)
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric].replace([np.inf, -np.inf], np.nan)
    Xt = Xt[numeric].replace([np.inf, -np.inf], np.nan)
    log(f"train={len(X)} test={len(Xt)} numeric={len(numeric)}")

    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(X, y))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    pred_xgb = np.zeros(len(Xt))
    pred_cat = np.zeros(len(Xt))

    for k, (a, b) in enumerate(folds):
        xa, xb = X.iloc[a], X.iloc[b]
        med = xa.median()
        xa_fill, xb_fill, xt_fill = xa.fillna(med), xb.fillna(med), Xt.fillna(med)
        # Manual selection: standardized train-fold class separation.
        sep = {}
        for c in numeric:
            sd = xa_fill[c].std()
            sep[c] = abs(xa_fill.loc[y[a] == 1, c].mean() - xa_fill.loc[y[a] == 0, c].mean()) / (sd + 1e-9)
        cols = list(pd.Series(sep).sort_values(ascending=False).head(args.top_features).index)

        mx = xgb.XGBClassifier(
            n_estimators=10000, learning_rate=0.02, grow_policy="lossguide",
            max_leaves=31, max_depth=0, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.35, reg_lambda=20,
            tree_method="hist", max_bin=128, eval_metric="auc",
            early_stopping_rounds=250, random_state=args.seed + k,
            n_jobs=-1,
        )
        mx.fit(xa_fill[cols], y[a], eval_set=[(xb_fill[cols], y[b])], verbose=False)
        oof_xgb[b] = mx.predict_proba(xb_fill[cols])[:, 1]
        pred_xgb += mx.predict_proba(xt_fill[cols])[:, 1] / args.folds

        mc = CatBoostClassifier(
            iterations=5000, learning_rate=0.04, grow_policy="Lossguide",
            max_leaves=31, min_data_in_leaf=100, l2_leaf_reg=10,
            eval_metric="AUC", od_type="Iter", od_wait=250,
            random_seed=args.seed + 100 + k, verbose=0, thread_count=-1,
        )
        mc.fit(xa_fill[cols], y[a], eval_set=(xb_fill[cols], y[b]))
        oof_cat[b] = mc.predict_proba(xb_fill[cols])[:, 1]
        pred_cat += mc.predict_proba(xt_fill[cols])[:, 1] / args.folds
        log(f"fold={k} xgb_auc={roc_auc_score(y[b], oof_xgb[b]):.5f} cat_auc={roc_auc_score(y[b], oof_cat[b]):.5f}")

    rank_xgb, rank_cat = rank01(oof_xgb), rank01(oof_cat)
    rank_xgb_t, rank_cat_t = rank01(pred_xgb), rank01(pred_cat)
    blend_oof = 0.55 * rank_xgb + 0.45 * rank_cat
    blend_pred = 0.55 * rank_xgb_t + 0.45 * rank_cat_t
    report = {
        "xgb_lossguide": score(y, oof_xgb),
        "catboost_lossguide": score(y, oof_cat),
        "rank_blend_55_xgb_45_cat": score(y, blend_oof),
    }
    log(json.dumps(report, indent=2))
    test_ids = te.application_id.to_numpy()
    files = {}
    for name, p in [("xgb_lossguide", pred_xgb), ("catboost_lossguide", pred_cat), ("rank_blend_55_xgb_45_cat", blend_pred)]:
        path = out / f"submission_{args.out_prefix}_{name}.csv"
        pd.DataFrame({"application_id": test_ids, "default_probability": rank01(p)}).to_csv(path, index=False)
        files[name] = str(path)
    summary = {
        "params": vars(args),
        "metrics": report,
        "files": files,
        "elapsed_minutes": (time.time() - start) / 60,
        "method": "manual fold-safe separation selection + XGBoost/CatBoost Lossguide",
    }
    with open(out / f"summary_{args.out_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
