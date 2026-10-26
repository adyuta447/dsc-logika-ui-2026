"""Fase A: where is the headroom, if any?

Two questions, both cheap, both answered before any model is built.

1. **Is 0.850 an information ceiling or a model ceiling?** Train the same
   configuration on 20/40/60/80/100 percent of the training rows and read the
   learning curve. If AUC is still climbing at 100 percent, the plateau is a
   variance problem and better-constrained models help. If it is flat, the data
   is exhausted and no feature work will move it.

2. **Is the generator monotone?** The spline logistic leg reaches AUC 0.8428
   against 0.8484 for the best boosted leg. On real credit data that gap is far
   wider, which suggests the response is close to additive and monotone in each
   risk feature. If so, imposing monotonicity on LightGBM removes variance at
   almost no bias cost - the standard credit scorecard argument - and none of
   v57-v74 ever tried it.

Constraint directions are derived **inside each training fold** from the sign of
the Spearman correlation between the feature and the label, so no fold sees its
own labels. Features whose correlation is weaker than `RHO_MIN` stay
unconstrained, as do categoricals.
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
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = (42, 7, 13)
RHO_MIN = 0.015

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

# a faster config than the v66 legs: this phase compares shapes, not absolutes
FAST = dict(objective="binary", learning_rate=0.03, num_leaves=15,
            min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, max_bin=127, deterministic=True,
            force_row_wise=True, n_jobs=-1, verbose=-1)


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


def spearman_signs(X, y, cols, rows):
    """Monotone direction per column, from training-fold labels only."""
    yr = rankdata(y[rows])
    yc = yr - yr.mean()
    denom = np.sqrt((yc ** 2).sum())
    out = []
    for c in cols:
        s = X[c].iloc[rows]
        if str(s.dtype) == "category":
            out.append(0)
            continue
        v = s.to_numpy(dtype=float)
        if not np.isfinite(v).any():
            out.append(0)
            continue
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        vr = rankdata(v)
        vc = vr - vr.mean()
        dv = np.sqrt((vc ** 2).sum())
        rho = 0.0 if dv == 0 or denom == 0 else float((vc @ yc) / (dv * denom))
        out.append(int(np.sign(rho)) if abs(rho) >= RHO_MIN else 0)
    return out


def run_cv(X, y, cols, seed, frac=1.0, monotone=False, log=print):
    oof = np.zeros(len(y))
    seen = np.zeros(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if frac < 1.0:
            a = rng.choice(a, int(round(len(a) * frac)), replace=False)
        params = dict(FAST, seed=seed)
        if monotone:
            params["monotone_constraints"] = spearman_signs(X, y, cols, a)
            params["monotone_constraints_method"] = "advanced"
        m = lgb.train(params, lgb.Dataset(X.iloc[a][cols], y[a]),
                      num_boost_round=6000,
                      valid_sets=[lgb.Dataset(X.iloc[b][cols], y[b])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(X.iloc[b][cols])
        seen[b] = True
    assert seen.all()
    return oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logp = OUT / "log_probe_v75.txt"

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(logp, "a") as fh:
            fh.write(line + "\n")

    X, y = load()
    cols = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"rows={len(y)} lean_cols={len(cols)} pos_rate={y.mean():.6f}")

    # ---- question 1: learning curve
    curve = []
    for frac in FRACTIONS:
        o = run_cv(X, y, cols, 42, frac=frac)
        row = {"frac": frac, "train_rows": int(round(len(y) * 0.8 * frac)),
               "auc": float(roc_auc_score(y, o)), "ks": ks(o, y)}
        curve.append(row)
        log(f"  curve frac={frac:.1f} train_rows={row['train_rows']:6d} "
            f"auc={row['auc']:.6f} ks={row['ks']:.6f}")
    slope = (curve[-1]["auc"] - curve[-2]["auc"]) / 0.2
    log(f"learning curve slope over the last 20% of data: {slope:+.6f} AUC per unit frac")

    # ---- question 2: monotone constraints
    mono = []
    for seed in SEEDS:
        base = run_cv(X, y, cols, seed, monotone=False)
        con = run_cv(X, y, cols, seed, monotone=True)
        row = {"seed": seed,
               "auc_free": float(roc_auc_score(y, base)),
               "auc_mono": float(roc_auc_score(y, con)),
               "ks_free": ks(base, y), "ks_mono": ks(con, y)}
        row["d_auc"] = row["auc_mono"] - row["auc_free"]
        row["d_ks"] = row["ks_mono"] - row["ks_free"]
        mono.append(row)
        log(f"  mono seed={seed} auc {row['auc_free']:.6f} -> {row['auc_mono']:.6f} "
            f"({row['d_auc']:+.6f}) | ks {row['ks_free']:.6f} -> {row['ks_mono']:.6f} "
            f"({row['d_ks']:+.6f})")

    d_auc = float(np.mean([r["d_auc"] for r in mono]))
    d_ks = float(np.mean([r["d_ks"] for r in mono]))
    allpos = all(r["d_auc"] > 0 for r in mono)
    log(f"\nmonotone mean dAUC {d_auc:+.6f} dKS {d_ks:+.6f} "
        f"| positive in all {len(SEEDS)} seeds: {allpos}")
    verdict = "PROMISING" if d_auc >= 0.0010 and allpos else "WEAK"
    log(f"verdict: {verdict}  (threshold +0.0010 AUC and all seeds positive)")

    (OUT / "probe_v75.json").write_text(json.dumps(
        {"rho_min": RHO_MIN, "lean_cols": len(cols), "learning_curve": curve,
         "curve_slope_last_step": slope, "monotone": mono,
         "mono_mean_d_auc": d_auc, "mono_mean_d_ks": d_ks,
         "mono_all_seeds_positive": allpos, "verdict": verdict,
         "minutes": (time.time() - t0) / 60}, indent=2))
    log(f"saved in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
