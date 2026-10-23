"""v70 exploratory: how much signal lives in each raw data source?

The v61-v69 models train one GBM on every feature at once. This script trains
the same LightGBM on disjoint column groups (bureau, repayment, account,
ecommerce, previous applications, application/profile) and on their rank
blend, to see whether a source-specific stack has more headroom than the
single full-feature model.

It is screening only: 5 folds, fold seed 42, LightGBM seed 0.
"""

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

GROUPS = {
    "bureau": ("bu_", "bua_", "buh_", "bx_", "b2_"),
    "repayment": ("rh_", "rx_", "r2_", "rq_", "sq_"),
    "account": ("ab_", "ax_", "a2_", "cf_"),
    "ecommerce": ("ec_", "ex_", "e2_", "cs_"),
    "previous": ("pa_", "px_", "pv_"),
}
PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=15,
              min_child_samples=200, feature_fraction=0.30,
              bagging_fraction=0.80, bagging_freq=1, lambda_l2=20.0,
              max_bin=127, seed=0, deterministic=True, force_row_wise=True,
              n_jobs=10, verbose=-1)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def rank01(v):
    return rankdata(v) / len(v)


def main():
    t0 = time.time()
    f = pd.read_pickle(ROOT / "agas/v69/output/features_v69.pkl")
    tr = f[f.target.notna()].reset_index(drop=True)
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    y = tr.target.astype(int).values
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")

    full = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    assign = {}
    for name, prefixes in GROUPS.items():
        assign[name] = [c for c in full if c.startswith(prefixes)]
    used = {c for v in assign.values() for c in v}
    assign["app_profile"] = [c for c in full if c not in used]
    assign["all"] = full

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    oof = {}
    for name, cols in assign.items():
        o = np.zeros(len(X))
        for a, b in folds:
            m = lgb.train(PARAMS, lgb.Dataset(X.iloc[a][cols], y[a]),
                          num_boost_round=12000,
                          valid_sets=[lgb.Dataset(X.iloc[b][cols], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            o[b] = m.predict(X.iloc[b][cols])
        oof[name] = o
        print(f"{name:12s} n={len(cols):3d} auc={roc_auc_score(y, o):.6f} "
              f"ks={ks(y, o):.6f}  {time.time()-t0:.0f}s", flush=True)

    blend = np.mean([rank01(oof[n]) for n in GROUPS], axis=0)
    print(f"{'blend5':12s} auc={roc_auc_score(y, blend):.6f} ks={ks(y, blend):.6f}")
    allb = np.mean([rank01(oof[n]) for n in list(GROUPS) + ["app_profile"]], axis=0)
    print(f"{'blend6':12s} auc={roc_auc_score(y, allb):.6f} ks={ks(y, allb):.6f}")
    corr = pd.DataFrame({k: rank01(v) for k, v in oof.items()}).corr(method="spearman")
    print(corr.round(3).to_string())
    np.savez_compressed(OUT / "source_auc_v70.npz",
                        **{f"oof_{k}": v for k, v in oof.items()}, y=y)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
