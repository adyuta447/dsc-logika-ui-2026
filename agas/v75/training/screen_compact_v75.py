"""Fase B': is 490 features too many for 9.500 positives?

What the fase A learning curve established: the 0.850 plateau is a variance
limit, not an information limit - AUC was still climbing at 56.000 training
rows. What the v70 source ablation establishes: the seven sources are heavily
redundant. Each one standalone reaches AUC 0.62-0.83, yet all of them together
reach only 0.846. That is the signature of a single latent risk factor that
every source measures a noisy copy of.

If that reading is right, 490 columns is not 490 pieces of information. It is
one piece of information measured 490 times, and fitting 490 of them against
9.500 positives - 19 positives per column - spends capacity on noise. A compact
model should then match or beat the wide one.

The selection has to be honest or the answer is meaningless, so the ranking is
recomputed **inside every training fold** from that fold's rows only: a short
booster is fitted, its gain importance orders the columns, and the real model is
refitted on the top k of that fold's own ranking. No fold ever sees a ranking
derived from its own labels.

`gam` (max_depth=1, a boosted additive model) and `gam2` (max_depth=2, additive
plus pairwise) are screened in the same harness, because they are the other
shape the latent-factor reading implies.
"""

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
os.chdir(ROOT)
import paths  # noqa: E402

OUT = HERE.parent / "output"
SIZES = (40, 80, 160, 320, 0)        # 0 means every lean column: the baseline
SEEDS = (42, 7, 13)
GATE = 0.0015

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

FAST = dict(objective="binary", learning_rate=0.03, num_leaves=15,
            min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, max_bin=127, deterministic=True,
            force_row_wise=True, n_jobs=-1, verbose=-1)
# the selector only has to order columns, so it is short and cheap
SEL = dict(FAST, learning_rate=0.05, feature_fraction=0.6)
SEL_ROUNDS = 600
GAM = dict(FAST, num_leaves=2, max_depth=1, learning_rate=0.05,
           feature_fraction=1.0, min_child_samples=100)
GAM2 = dict(FAST, num_leaves=4, max_depth=2, learning_rate=0.05,
            feature_fraction=0.7, min_child_samples=100)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def load():
    f = pd.read_pickle(paths.FEATURES_CACHE)
    ex = pd.read_pickle(paths.EXTRA_CACHE)
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    tr = f[f.target.notna()].reset_index(drop=True)
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
    return X, tr.target.values.astype(int)


def fit(params, X, y, a, b, cols, rounds):
    m = lgb.train(params, lgb.Dataset(X.iloc[a][cols], y[a]),
                  num_boost_round=rounds,
                  valid_sets=[lgb.Dataset(X.iloc[b][cols], y[b])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(X.iloc[b][cols]), m.best_iteration


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logp = OUT / "log_compact_v75.txt"

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(logp, "a") as fh:
            fh.write(line + "\n")

    X, y = load()
    lean = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"rows={len(y)} lean={len(lean)} positives={int(y.sum())} "
        f"({y.sum() / len(lean):.1f} positives per column)")

    runs = {}            # (variant) -> {seed: oof}
    kept_counts = []
    for seed in SEEDS:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        oof = {}
        for k in SIZES:
            oof[f"top{k or len(lean)}"] = np.zeros(len(y))
        oof["gam"] = np.zeros(len(y))
        oof["gam2"] = np.zeros(len(y))

        for a, b in folds:
            # fold-safe column ranking from this fold's training rows only
            sel = lgb.train(dict(SEL, seed=seed),
                            lgb.Dataset(X.iloc[a][lean], y[a]),
                            num_boost_round=SEL_ROUNDS)
            gain = sel.feature_importance("gain")
            order = np.argsort(-gain)
            kept_counts.append(int((gain > 0).sum()))

            for k in SIZES:
                cols = lean if k == 0 else [lean[i] for i in order[:k]]
                p, _ = fit(dict(FAST, seed=seed), X, y, a, b, cols, 6000)
                oof[f"top{k or len(lean)}"][b] = p

            p, _ = fit(dict(GAM, seed=seed), X, y, a, b, lean, 12000)
            oof["gam"][b] = p
            p, _ = fit(dict(GAM2, seed=seed), X, y, a, b, lean, 10000)
            oof["gam2"][b] = p

        for name, o in oof.items():
            runs.setdefault(name, {})[seed] = o
            log(f"  seed={seed} {name:8s} auc={roc_auc_score(y, o):.6f} "
                f"ks={ks(o, y):.6f}")
        log(f"  elapsed {(time.time() - t0) / 60:.1f} min")

    base = f"top{len(lean)}"
    rows = []
    for name, per in runs.items():
        aucs = [roc_auc_score(y, per[s]) for s in SEEDS]
        kss = [ks(per[s], y) for s in SEEDS]
        d = [a - roc_auc_score(y, runs[base][s]) for s, a in zip(SEEDS, aucs)]
        rows.append({"variant": name,
                     "auc_mean": round(float(np.mean(aucs)), 6),
                     "auc_sd": round(float(np.std(aucs, ddof=1)), 6),
                     "ks_mean": round(float(np.mean(kss)), 6),
                     "d_auc_mean": round(float(np.mean(d)), 6),
                     "d_auc_all_positive": bool(all(x > 0 for x in d)),
                     "per_seed_d_auc": [round(x, 6) for x in d]})
    rows.sort(key=lambda r: -r["d_auc_mean"])

    log("")
    log(f"{'variant':10s} {'AUC mean':>10} {'sd':>9} {'KS mean':>9} "
        f"{'dAUC':>10} {'all+':>5}")
    for r in rows:
        log(f"{r['variant']:10s} {r['auc_mean']:10.6f} {r['auc_sd']:9.6f} "
            f"{r['ks_mean']:9.6f} {r['d_auc_mean']:+10.6f} "
            f"{str(r['d_auc_all_positive']):>5}")

    winners = [r for r in rows
               if r["variant"] != base and r["d_auc_mean"] >= GATE
               and r["d_auc_all_positive"]]
    log(f"\ngate: dAUC >= {GATE} and positive in all {len(SEEDS)} seeds")
    log(f"passed: {[r['variant'] for r in winners] or 'none'}")
    log(f"median columns with non-zero gain in the selector: "
        f"{int(np.median(kept_counts))}")

    np.savez_compressed(OUT / "preds_compact_v75.npz", y=y,
                        **{f"{n}_s{s}": v for n, per in runs.items()
                           for s, v in per.items()})
    (OUT / "compact_v75.json").write_text(json.dumps(
        {"lean_cols": len(lean), "positives": int(y.sum()), "sizes": SIZES,
         "seeds": SEEDS, "gate": GATE, "baseline": base, "results": rows,
         "passed": [r["variant"] for r in winners],
         "selector_nonzero_median": int(np.median(kept_counts)),
         "minutes": (time.time() - t0) / 60}, indent=2))
    log(f"saved in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
