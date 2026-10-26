"""Fase 2: test the mechanism directly by training on the dropped columns.

v77's reading of the gradient is that `base - w*lgb_lean` is a
full-columns-minus-lean-columns contrast, so raising `w` raises how much of the
108 columns `lean` drops reaches the ranking. If that is right, the contrast is
a noisy way to do something that can be done directly: train one leg on those
108 columns alone and give it a positive weight.

The two are not the same object. The contrast carries the dropped columns'
signal *and* whatever disagreement the model families have on the shared
columns; a leg trained only on the dropped columns carries the first and none
of the second. Comparing them is therefore a test of the explanation, not just
another candidate.

Leakage control is unchanged: the columns come from the same cutoff-respecting
cache, and folds are stratified on the label with no fold seeing its own rows.
"""

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
os.chdir(ROOT)
import paths  # noqa: E402

OUT = HERE.parent / "output"
FOLDS, SEEDS = 10, (42, 7)

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=15,
              min_child_samples=200, feature_fraction=0.5, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, max_bin=127, extra_trees=True,
              deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = pd.read_pickle(paths.FEATURES_CACHE)
    ex = pd.read_pickle(paths.EXTRA_CACHE)
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X, Xte = tr.drop(columns=ids), te.drop(columns=ids)
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    y = tr.target.values.astype(int)
    del f, ex

    dropped = [c for c in X.columns
               if c in NOISE_EXACT or c.startswith(NOISE_PREFIX)]
    print(f"dropped columns: {len(dropped)} of {X.shape[1]}")
    print(f"  named: {sorted(c for c in dropped if c in NOISE_EXACT)}")
    pref = {p: sum(c.startswith(p) for c in dropped) for p in NOISE_PREFIX}
    print(f"  by prefix: {pref}")

    oof = np.zeros(len(y))
    test = np.zeros(len(Xte))
    for seed in SEEDS:
        o = np.zeros(len(y))
        p = np.zeros(len(Xte))
        for a, b in StratifiedKFold(FOLDS, shuffle=True,
                                    random_state=seed).split(X, y):
            m = lgb.train(dict(PARAMS, seed=seed),
                          lgb.Dataset(X.iloc[a][dropped], y[a]),
                          num_boost_round=16000,
                          valid_sets=[lgb.Dataset(X.iloc[b][dropped], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            o[b] = m.predict(X.iloc[b][dropped])
            p += m.predict(Xte[dropped]) / FOLDS
        print(f"  seed {seed}: auc={roc_auc_score(y, o):.6f} ks={ks(o, y):.6f}")
        oof += o / len(SEEDS)
        test += p / len(SEEDS)

    auc, k = float(roc_auc_score(y, oof)), ks(oof, y)
    print(f"\ndropped-columns leg: AUC {auc:.6f} KS {k:.6f}  "
          f"[{(time.time()-t0)/60:.1f} min]")
    np.savez_compressed(OUT / "preds_dropped_v77.npz",
                        train_ids=tr.application_id.values,
                        test_ids=te.application_id.values, y=y,
                        oof_dropped=oof, test_dropped=test)
    (OUT / "dropped_v77.json").write_text(json.dumps(
        {"n_dropped": len(dropped), "n_all": int(X.shape[1]),
         "dropped_named": sorted(c for c in dropped if c in NOISE_EXACT),
         "by_prefix": pref, "folds": FOLDS, "seeds": list(SEEDS),
         "oof_auc": auc, "oof_ks": k,
         "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"saved {OUT / 'preds_dropped_v77.npz'}")


if __name__ == "__main__":
    main()
