"""v70 exploratory: does the 5-level interest-rate grade hide within-grade signal?

`annual_interest_rate` takes exactly five values (0.16, 0.20, 0.24, 0.28,
0.32) whose default rates are 5.5%, 10.6%, 17.8%, 30.4%, 50.3%. That looks like
an underwriting grade assigned from a latent risk score. A single tree ensemble
can split on the grade, but it must share capacity across all five regimes.

This script compares three ways of using the grade:
  global   - one model on every row (the current recipe)
  bygrade  - one model per grade, trained only on rows of that grade
  hybrid   - rank(global) + rank(bygrade) averaged

5 folds, fold seed 42, LightGBM seed 0; the same params as the v69 lean run.
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
PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=15,
              min_child_samples=200, feature_fraction=0.30,
              bagging_fraction=0.80, bagging_freq=1, lambda_l2=20.0,
              max_bin=127, seed=0, deterministic=True, force_row_wise=True,
              n_jobs=10, verbose=-1)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def rank01(v):
    return rankdata(v) / len(v)


def fit(cols, X, y, folds):
    o = np.zeros(len(X))
    for a, b in folds:
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[a][cols], y[a]),
                      num_boost_round=12000,
                      valid_sets=[lgb.Dataset(X.iloc[b][cols], y[b])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        o[b] = m.predict(X.iloc[b][cols])
    return o


def main():
    t0 = time.time()
    f = pd.read_pickle(ROOT / "agas/v69/output/features_v69.pkl")
    tr = f[f.target.notna()].reset_index(drop=True)
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    y = tr.target.astype(int).values
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
    cols = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    grade = tr.interest_rate.round(2).values

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    g = fit(cols, X, y, folds)
    print(f"global   auc={roc_auc_score(y, g):.6f} ks={ks(y, g):.6f} {time.time()-t0:.0f}s", flush=True)

    per = np.zeros(len(X))
    for gv in np.unique(grade):
        idx = np.where(grade == gv)[0]
        sub = X.iloc[idx].reset_index(drop=True)
        yy = y[idx]
        subf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(sub, yy))
        o = fit(cols, sub, yy, subf)
        per[idx] = o
        print(f"  grade {gv:.2f} n={len(idx):5d} auc={roc_auc_score(yy, o):.4f}", flush=True)
    print(f"bygrade  auc={roc_auc_score(y, per):.6f} ks={ks(y, per):.6f} {time.time()-t0:.0f}s", flush=True)

    for w in (0.25, 0.5, 0.75):
        h = (1 - w) * rank01(g) + w * rank01(per)
        print(f"hybrid w={w:.2f} auc={roc_auc_score(y, h):.6f} ks={ks(y, h):.6f}", flush=True)

    np.savez_compressed(OUT / "grade_v70.npz", y=y, grade=grade, global_=g, bygrade=per)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
