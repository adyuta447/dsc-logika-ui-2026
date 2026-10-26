"""Fase 1: tune the leg hyper-parameters, which no version ever did.

v66's docstring is explicit that its parameters are "copied from v61" so the
column sets and settings would be the ones that scored 0.56312. That was the
right call for reproducing a known result, but it means the boosting
parameters behind every leg in the anchor have been inherited unchanged since
v61 and never searched. v70 is the only indirect evidence that they are not
optimal: its `extra` config reached a single-leg AUC of 0.8478 and `stiff7`
0.8472, against about 0.8467 for v66's own LightGBM legs.

So this screens an explicitly written grid on the `lean` columns - nine
configurations, each one a stated hypothesis about what the inherited settings
get wrong, not a search over a parameter space. 5 folds x 2 fold seeds each,
and a configuration only counts as better if it wins on both seeds.

This is hand-specified model tuning, not AutoML: there is no automated search,
no pipeline selection, and every configuration below was written out with its
reason.
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
SEEDS = (42, 7)
FOLDS = 5

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

COMMON = dict(objective="binary", bagging_freq=1, deterministic=True,
              force_row_wise=True, n_jobs=-1, verbose=-1)

# name -> (params, rounds, why this configuration exists)
GRID = {
    "v61_inherited": (dict(COMMON, learning_rate=0.01, num_leaves=15,
                           min_child_samples=200, feature_fraction=0.3,
                           bagging_fraction=0.8, lambda_l2=5.0, max_bin=127),
                      12000, "the settings every leg has used since v61"),
    "wide_extra": (dict(COMMON, learning_rate=0.012, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35,
                        bagging_fraction=0.85, lambda_l2=12.0, max_bin=127,
                        extra_trees=True),
                   10000, "v70's `extra` shape, the best single leg measured so far"),
    "stiff": (dict(COMMON, learning_rate=0.01, num_leaves=7,
                   min_child_samples=400, feature_fraction=0.2,
                   bagging_fraction=0.7, lambda_l2=40.0, max_bin=127),
              16000, "v70's `stiff7` shape: far less capacity, far more shrinkage"),
    "slow": (dict(COMMON, learning_rate=0.005, num_leaves=15,
                  min_child_samples=200, feature_fraction=0.3,
                  bagging_fraction=0.8, lambda_l2=5.0, max_bin=127),
             30000, "half the step size; more rounds is the cheapest variance cut"),
    "deep": (dict(COMMON, learning_rate=0.015, num_leaves=63,
                  min_child_samples=60, feature_fraction=0.4,
                  bagging_fraction=0.8, lambda_l2=30.0, max_bin=127),
             8000, "much more capacity, held back by shrinkage instead of depth"),
    "thin_ff": (dict(COMMON, learning_rate=0.01, num_leaves=15,
                     min_child_samples=200, feature_fraction=0.15,
                     bagging_fraction=0.8, lambda_l2=5.0, max_bin=127),
                14000, "harder column subsampling: 490 redundant columns should "
                       "tolerate it and it decorrelates the trees"),
    "loose_l2": (dict(COMMON, learning_rate=0.01, num_leaves=15,
                      min_child_samples=100, feature_fraction=0.3,
                      bagging_fraction=0.8, lambda_l2=1.0, max_bin=127),
                 12000, "checks whether the inherited shrinkage is too strong"),
    "extra_only": (dict(COMMON, learning_rate=0.01, num_leaves=15,
                        min_child_samples=200, feature_fraction=0.3,
                        bagging_fraction=0.8, lambda_l2=5.0, max_bin=127,
                        extra_trees=True),
                   14000, "the inherited config with random split thresholds, "
                          "isolating the one change `wide_extra` bundles"),
    "fine_bin": (dict(COMMON, learning_rate=0.01, num_leaves=15,
                      min_child_samples=200, feature_fraction=0.3,
                      bagging_fraction=0.8, lambda_l2=5.0, max_bin=255),
                 12000, "twice the histogram resolution"),
}


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logp = OUT / "log_params_v76.txt"

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(logp, "a") as fh:
            fh.write(line + "\n")

    X, y = load()
    lean = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"rows={len(y)} lean={len(lean)} configs={len(GRID)} "
        f"folds={FOLDS} seeds={list(SEEDS)}")

    res = {}
    for name, (params, rounds, why) in GRID.items():
        per = {}
        for seed in SEEDS:
            oof = np.zeros(len(y))
            iters = []
            for a, b in StratifiedKFold(FOLDS, shuffle=True,
                                        random_state=seed).split(X, y):
                m = lgb.train(dict(params, seed=seed),
                              lgb.Dataset(X.iloc[a][lean], y[a]),
                              num_boost_round=rounds,
                              valid_sets=[lgb.Dataset(X.iloc[b][lean], y[b])],
                              callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(X.iloc[b][lean])
                iters.append(m.best_iteration)
            per[seed] = {"auc": float(roc_auc_score(y, oof)), "ks": ks(oof, y),
                         "median_rounds": int(np.median(iters))}
            log(f"  {name:14s} seed={seed} auc={per[seed]['auc']:.6f} "
                f"ks={per[seed]['ks']:.6f} rounds~{per[seed]['median_rounds']}")
        res[name] = {"why": why, "per_seed": per,
                     "auc_mean": float(np.mean([v["auc"] for v in per.values()])),
                     "ks_mean": float(np.mean([v["ks"] for v in per.values()])),
                     "params": {k: v for k, v in params.items()
                                if k not in COMMON or k == "learning_rate"},
                     "rounds": rounds}
        log(f"  {name:14s} MEAN auc={res[name]['auc_mean']:.6f} "
            f"ks={res[name]['ks_mean']:.6f}   [{(time.time()-t0)/60:.1f} min]")

    base = res["v61_inherited"]["auc_mean"]
    rows = []
    for name, r in res.items():
        d = [r["per_seed"][s]["auc"] - res["v61_inherited"]["per_seed"][s]["auc"]
             for s in SEEDS]
        rows.append({"config": name, "auc_mean": round(r["auc_mean"], 6),
                     "ks_mean": round(r["ks_mean"], 6),
                     "d_auc_mean": round(r["auc_mean"] - base, 6),
                     "d_auc_per_seed": [round(x, 6) for x in d],
                     "wins_both_seeds": bool(all(x > 0 for x in d)),
                     "why": r["why"]})
    rows.sort(key=lambda r: -r["auc_mean"])

    log("")
    log(f"{'config':15s} {'AUC mean':>10} {'KS mean':>9} {'dAUC':>10} {'both':>5}")
    for r in rows:
        log(f"{r['config']:15s} {r['auc_mean']:10.6f} {r['ks_mean']:9.6f} "
            f"{r['d_auc_mean']:+10.6f} {str(r['wins_both_seeds']):>5}")

    winners = [r for r in rows
               if r["config"] != "v61_inherited" and r["wins_both_seeds"]]
    log(f"\nbeat the inherited config on both seeds: "
        f"{[r['config'] for r in winners] or 'none'}")
    if winners:
        log(f"best: {winners[0]['config']} at {winners[0]['d_auc_mean']:+.6f} AUC")

    (OUT / "params_v76.json").write_text(json.dumps(
        {"folds": FOLDS, "seeds": list(SEEDS), "n_lean": len(lean),
         "baseline": "v61_inherited", "results": rows,
         "winners": [r["config"] for r in winners],
         "detail": res, "minutes": (time.time() - t0) / 60}, indent=2))
    log(f"saved in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
