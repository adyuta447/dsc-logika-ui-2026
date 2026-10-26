"""Fase 1: is the OOF estimator itself the problem?

Every version from v57 on measured a candidate with `ks_2samp` over the pooled
OOF vector. That vector is stitched together from one model per fold, and each
fold model has its own score scale. KS is a global ranking statistic, so any
between-fold scale drift blurs the pooled ranking and pushes KS down. Test
predictions have no such problem: all 30000 rows are scored by one ensemble.

If that is what the +0.032 OOF->public offset is, then a model whose fold-to-
fold variation is larger looks *better* under the broken estimator without
being better on test - exactly the pattern in the public ledger, where the
highest OOF KS candidate has the smallest offset.

This script measures it directly. One LightGBM `lean` leg, 10 folds, two
independent fold seeds, keeping the fold index of every row so the same OOF
vector can be scored three ways:

    ks_pooled    pooled ranking, the v57-v73 estimator
    ks_foldnorm  rank01 inside each fold first, then pool
    ks_perfold   mean of the within-fold KS values

ks_foldnorm is the estimator that removes between-fold scale drift while
keeping all 70000 rows. The gate is +0.005.
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
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

OUT = HERE.parent / "output"
FOLDS = 10
FOLD_SEEDS = (42, 7)
GATE = 0.005

# copied from v66 so the leg is the same object the anchor is built from
NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")
LGB = dict(objective="binary", learning_rate=0.01, num_leaves=15,
           min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, max_bin=127, seed=0,
           deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def load_frames():
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
    return (X, Xte, tr.application_id.values, te.application_id.values,
            tr.target.values.astype(int))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_path = OUT / "log_foldnorm_v74.txt"

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    X, Xte, train_ids, test_ids, y = load_frames()
    lean = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"train={X.shape} test={Xte.shape} lean={len(lean)} pos_rate={y.mean():.6f}")

    store, rows = {}, []
    for fs in FOLD_SEEDS:
        folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=fs).split(X, y))
        oof = np.zeros(len(y))
        fold_of = np.full(len(y), -1, dtype=np.int16)
        pred = np.zeros(len(Xte))
        for k, (a, b) in enumerate(folds):
            m = lgb.train(LGB, lgb.Dataset(X.iloc[a][lean], y[a]),
                          num_boost_round=12000,
                          valid_sets=[lgb.Dataset(X.iloc[b][lean], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[b] = m.predict(X.iloc[b][lean])
            fold_of[b] = k
            pred += m.predict(Xte[lean]) / FOLDS
            log(f"  fs{fs} fold {k} rounds={m.best_iteration} "
                f"ks={ks(oof[b], y[b]):.6f}")
        assert (fold_of >= 0).all()

        # the three estimators on one and the same OOF vector
        pooled = ks(oof, y)
        fn = np.empty(len(y))
        per = []
        for k in range(FOLDS):
            m_ = fold_of == k
            fn[m_] = rank01(oof[m_])
            per.append(ks(oof[m_], y[m_]))
        foldnorm = ks(fn, y)
        r = {"fold_seed": fs,
             "auc_pooled": float(roc_auc_score(y, oof)),
             "auc_foldnorm": float(roc_auc_score(y, fn)),
             "ks_pooled": pooled,
             "ks_foldnorm": foldnorm,
             "ks_perfold_mean": float(np.mean(per)),
             "ks_perfold_sd": float(np.std(per, ddof=1)),
             "gain_foldnorm": foldnorm - pooled}
        rows.append(r)
        store[f"oof_fs{fs}"] = oof
        store[f"fold_fs{fs}"] = fold_of
        store[f"test_fs{fs}"] = pred
        log(f"fs{fs}: pooled {pooled:.6f} | foldnorm {foldnorm:.6f} "
            f"({foldnorm - pooled:+.6f}) | perfold {np.mean(per):.6f} "
            f"+-{np.std(per, ddof=1):.6f} | auc {r['auc_pooled']:.6f}")

    gain = float(np.mean([r["gain_foldnorm"] for r in rows]))
    verdict = "H1 CONFIRMED" if gain >= GATE else "H1 REJECTED"
    log(f"\nmean foldnorm gain {gain:+.6f} vs gate {GATE} -> {verdict}")

    np.savez_compressed(OUT / "preds_foldnorm_v74.npz",
                        train_ids=train_ids, test_ids=test_ids, y=y, **store)
    (OUT / "foldnorm_v74.json").write_text(json.dumps(
        {"gate": GATE, "mean_gain": gain, "verdict": verdict,
         "minutes": (time.time() - t0) / 60, "runs": rows}, indent=2))
    log(f"saved in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
