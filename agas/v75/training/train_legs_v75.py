"""Fase B: train the legs the additive hypothesis implies, at production settings.

Three new legs, each a one-line departure from the v66 LightGBM leg, each
motivated by the same reading of the data - that the response is close to
additive and monotone in the risk features:

  `mono`  the v66 lean leg with monotone constraints, directions taken from the
          sign of the training-fold Spearman correlation only.
  `gam`   `max_depth=1`: every tree splits one feature once, so the model is a
          boosted generalised additive model with no interactions at all. If
          the generator is additive this leg loses nothing and carries far less
          variance than a 15-leaf tree.
  `gam2`  `max_depth=2`: additive plus pairwise interactions, the GA2M shape.

`free` reproduces the v66 lean leg under the same folds and seeds so every
comparison is like for like.

Nothing here changes `features.py`, the fold scheme, or the blend recipe. The
only thing under test is leg quality.
"""

import argparse
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
RHO_MIN = 0.015

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")

BASE = dict(objective="binary", learning_rate=0.01, num_leaves=15,
            min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, max_bin=127, deterministic=True,
            force_row_wise=True, n_jobs=-1, verbose=-1)

# a depth-limited tree has far less capacity per round, so it needs a larger
# step and more rounds to reach the same fit
GAM = dict(BASE, num_leaves=2, max_depth=1, learning_rate=0.03,
           feature_fraction=1.0, min_child_samples=100)
GAM2 = dict(BASE, num_leaves=4, max_depth=2, learning_rate=0.03,
            feature_fraction=0.7, min_child_samples=100)

ROUNDS = {"free": 12000, "mono": 12000, "gam": 40000, "gam2": 30000, "top": 12000}

# the selector only has to order columns, so it is short and cheap; it is
# refitted inside every training fold so no fold sees its own labels
SEL = dict(BASE, learning_rate=0.05, feature_fraction=0.6)
SEL_ROUNDS = 600


def rank01(v):
    return rankdata(v) / len(v)


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
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X, Xte = tr.drop(columns=ids), te.drop(columns=ids)
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    return (X, Xte, tr.application_id.values, te.application_id.values,
            tr.target.values.astype(int))


def signs(X, y, cols, rows):
    yr = rankdata(y[rows])
    yc = yr - yr.mean()
    dy = np.sqrt((yc ** 2).sum())
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
        rho = 0.0 if dv == 0 or dy == 0 else float((vc @ yc) / (dv * dy))
        out.append(int(np.sign(rho)) if abs(rho) >= RHO_MIN else 0)
    return out


def base_for(kind):
    if kind.startswith("top"):
        return BASE
    return {"free": BASE, "mono": BASE, "gam": GAM, "gam2": GAM2}[kind]


def params_for(kind, seed, X, y, cols, rows):
    p = dict(base_for(kind), seed=seed)
    if kind == "mono":
        p["monotone_constraints"] = signs(X, y, cols, rows)
        p["monotone_constraints_method"] = "advanced"
    return p


def fold_cols(kind, X, y, cols, rows, seed):
    """Columns this leg uses in this fold. `topK` reranks inside the fold."""
    if not kind.startswith("top"):
        return cols
    k = int(kind[3:])
    sel = lgb.train(dict(SEL, seed=seed), lgb.Dataset(X.iloc[rows][cols], y[rows]),
                    num_boost_round=SEL_ROUNDS)
    order = np.argsort(-sel.feature_importance("gain"))
    return [cols[i] for i in order[:k]]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13")
    ap.add_argument("--lgb-seeds", type=str, default="0,1")
    ap.add_argument("--legs", type=str, default="free,mono,gam,gam2")
    ap.add_argument("--cols", type=str, default="lean")
    return ap.parse_args()


def main():
    args = parse()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logp = OUT / "log_legs_v75.txt"

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(logp, "a") as fh:
            fh.write(line + "\n")

    X, Xte, train_ids, test_ids, y = load()
    full = list(X.columns)
    lean = [c for c in full
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    cols = lean if args.cols == "lean" else full
    legs = [s for s in args.legs.split(",") if s]
    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s]
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s]
    log(f"cols={args.cols}({len(cols)}) legs={legs} folds={args.folds} "
        f"fold_seeds={fold_seeds} lgb_seeds={lgb_seeds}")

    oof = {k: np.zeros(len(y)) for k in legs}
    test = {k: np.zeros(len(Xte)) for k in legs}
    per_seed = {}

    for fs in fold_seeds:
        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                     random_state=fs).split(X, y))
        for kind in legs:
            o = np.zeros(len(y))
            p = np.zeros(len(Xte))
            rounds = ROUNDS["top" if kind.startswith("top") else kind]
            for a, b in folds:
                fc = fold_cols(kind, X, y, cols, a, fs)
                for sd in lgb_seeds:
                    m = lgb.train(params_for(kind, sd, X, y, fc, a),
                                  lgb.Dataset(X.iloc[a][fc], y[a]),
                                  num_boost_round=rounds,
                                  valid_sets=[lgb.Dataset(X.iloc[b][fc], y[b])],
                                  callbacks=[lgb.early_stopping(300, verbose=False)])
                    o[b] += m.predict(X.iloc[b][fc]) / len(lgb_seeds)
                    p += m.predict(Xte[fc]) / (len(folds) * len(lgb_seeds))
            oof[kind] += o / len(fold_seeds)
            test[kind] += p / len(fold_seeds)
            per_seed.setdefault(kind, {})[fs] = {
                "auc": float(roc_auc_score(y, o)), "ks": ks(o, y)}
            log(f"  fs{fs} {kind:5s} auc={per_seed[kind][fs]['auc']:.6f} "
                f"ks={per_seed[kind][fs]['ks']:.6f}")
        np.savez_compressed(OUT / "preds_v75_partial.npz",
                            train_ids=train_ids, test_ids=test_ids, y=y,
                            **{f"oof_{k}": v for k, v in oof.items()},
                            **{f"test_{k}": v for k, v in test.items()})
        log(f"  saved after fold seed {fs}, elapsed {(time.time()-t0)/60:.1f} min")

    log("")
    for kind in legs:
        log(f"BAG {kind:5s} auc={roc_auc_score(y, oof[kind]):.6f} "
            f"ks={ks(oof[kind], y):.6f}")

    np.savez_compressed(OUT / "preds_v75.npz",
                        train_ids=train_ids, test_ids=test_ids, y=y,
                        **{f"oof_{k}": v for k, v in oof.items()},
                        **{f"test_{k}": v for k, v in test.items()})
    (OUT / "summary_v75_legs.json").write_text(json.dumps(
        {"cols": args.cols, "n_cols": len(cols), "folds": args.folds,
         "fold_seeds": fold_seeds, "lgb_seeds": lgb_seeds, "rho_min": RHO_MIN,
         "rounds": ROUNDS, "per_seed": per_seed,
         "bagged": {k: {"auc": float(roc_auc_score(y, oof[k])), "ks": ks(oof[k], y)}
                    for k in legs},
         "minutes": (time.time() - t0) / 60}, indent=2))
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
