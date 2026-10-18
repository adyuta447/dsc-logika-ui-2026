"""Main v65 run.

Three decisions came out of the measurement runs in this folder and are the
reason this is not simply v64 with a different number.

1. Feature selection goes to 250 columns, not 200. At `lr 0.01 / lambda_l2 5`
   the KS spread across fold seeds is 0.00034 at 250 columns against 0.00145
   with no selection at all, and 250 also has the best mean KS. Selection is
   doing real work here; the v63 finding that 200 was the peak was measured at
   `lr 0.02 / lambda_l2 20`, a different operating point.

2. Only four model legs, not eight. The two axes that matter are the model
   quality axis and the feature-richness axis. Extra legs bought diversity for
   blends, and v64 already showed the candidate pool was saturated: 88 vectors
   deduplicated down to 7. Four legs over six fold seeds gives a stronger
   average per seed and the same wall clock as v64's eight legs over four.

3. The extrapolation axis is the one with public evidence behind it. Two of the
   three public scores this project has ever earned above 0.559 came from
   pushing a feature-rich model away from a feature-poor one: `qc_00` at 0.5617
   and v61 at 0.56312. v63 replaced that axis with a selection-sharpness axis
   and scored 0.557. v65 keeps the proven axis as the primary candidate and
   adds the sharpness axis as a secondary, so the two can be compared on equal
   footing instead of one being assumed.

Leakage control, unchanged from v61 onward: every historical aggregate is cut at
`decision_datetime`, feature selection is a gain probe fitted on the training
fold only, and factor loadings are computed from training-fold labels only.
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
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "agas/shared"))
sys.path.insert(0, str(HERE.parents[2]))
os.chdir(HERE.parents[2])

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE.parent / "output"
LOG = OUT / "train_log_v65.txt"

LGB_PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=15,
                  min_child_samples=200, feature_fraction=0.3,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  max_bin=127, deterministic=True, force_row_wise=True,
                  n_jobs=-1, verbose=-1)


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def for_catboost(df, cols):
    """Split a fold frame into CatBoost's numeric and categorical halves.

    CatBoost will not take a pandas `category` column, and stringifying alone is
    not enough either: it then tries to read the strings as numbers and fails.
    The column has to be declared categorical, so this returns the converted
    frame together with the names to declare. The categoricals here are five
    application-level codes and one region code, all low cardinality, and the
    gain probe has already decided whether any of them earn a place at all.
    """
    out = df.copy()
    cat = []
    for c in cols:
        if str(out[c].dtype) in ("category", "object", "string"):
            out[c] = out[c].astype(str)
            cat.append(c)
    return out, cat


def rank01(v):
    return rankdata(v) / len(v)


def metrics(y, p):
    return (float(roc_auc_score(y, p)),
            float(ks_2samp(p[y == 1], p[y == 0]).statistic))


def gain_columns(Xtr, ytr, n, seed):
    """Per-fold gain probe. Fitted on training-fold rows only."""
    probe = lgb.train({**LGB_PARAMS, "feature_fraction": 1.0, "seed": seed,
                       "learning_rate": 0.1},
                      lgb.Dataset(Xtr, ytr), num_boost_round=300)
    g = pd.Series(probe.feature_importance(importance_type="gain"), index=Xtr.columns)
    return list(g.sort_values(ascending=False).index[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13,21,5,101")
    ap.add_argument("--top-n", type=int, default=250)
    ap.add_argument("--out-prefix", type=str, default="v65")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    import features as ft
    import cv_harness as H

    f, tr, te = H.load_frames(True)
    y = tr.target.astype(int).values
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    Xte = te.drop(columns=["application_id", "customer_id", "target"])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    say(f"features: {X.shape[1]} candidates, train {len(tr)}, test {len(te)}")

    app = ft.load_apps()
    seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]

    store = {}
    oof_all = {}
    test_all = {}
    per_seed = {}
    factors_all = None

    for j, fs in enumerate(seeds):
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=fs).split(X, y))
        fac, p_tr, p_te, fac_cols = H.factor_block(tr, te, folds, app, 1)
        base_tr = X[[c for c in X.columns if c not in fac_cols]]
        base_te = Xte[[c for c in Xte.columns if c not in fac_cols]]
        Xt = pd.concat([base_tr, fac[0].iloc[p_tr].reset_index(drop=True)], axis=1)
        Xv = pd.concat([base_te, fac[0].iloc[p_te].reset_index(drop=True)], axis=1)

        legs = ["sel", "lean", "xgb", "cat"]
        oof = {k: np.zeros(len(y)) for k in legs}
        pte = {k: np.zeros(len(Xv)) for k in legs}

        for a, b in folds:
            xa = Xt.iloc[a]
            xb = Xt.iloc[b]
            cols = gain_columns(xa, y[a], args.top_n, fs)

            m = lgb.train({**LGB_PARAMS, "seed": fs}, lgb.Dataset(xa[cols], y[a]),
                          num_boost_round=8000, valid_sets=[lgb.Dataset(xb[cols], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof["sel"][b] = m.predict(xb[cols])
            pte["sel"] += m.predict(Xv[cols]) / len(folds)

            # reduction leg: the feature-poor side of the proven axis. It uses
            # every column the block cache carries, which is what makes it poor
            # relative to the selected model rather than merely smaller.
            lcols = [c for c in xa.columns if not c.startswith(("dq_", "af_"))]
            m = lgb.train({**LGB_PARAMS, "seed": fs}, lgb.Dataset(xa[lcols], y[a]),
                          num_boost_round=8000, valid_sets=[lgb.Dataset(xb[lcols], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof["lean"][b] = m.predict(xb[lcols])
            pte["lean"] += m.predict(Xv[lcols]) / len(folds)

            dxtr = xgb.DMatrix(xa[cols], label=y[a])
            dxva = xgb.DMatrix(xb[cols], label=y[b])
            dxte = xgb.DMatrix(Xv[cols])
            xm = xgb.train({"objective": "binary:logistic", "eta": 0.01,
                            "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.3,
                            "min_child_weight": 20, "lambda": 5.0, "max_bin": 128,
                            "seed": fs, "tree_method": "hist", "nthread": -1},
                           dxtr, num_boost_round=3000, evals=[(dxva, "va")],
                           early_stopping_rounds=200, verbose_eval=False)
            oof["xgb"][b] = xm.predict(dxva, iteration_range=(0, xm.best_iteration + 1))
            pte["xgb"] += xm.predict(dxte, iteration_range=(0, xm.best_iteration + 1)) / len(folds)

            xa_c, cat_cols = for_catboost(xa[cols], cols)
            xb_c, _ = for_catboost(xb[cols], cols)
            cm = CatBoostClassifier(iterations=4000, learning_rate=0.02, depth=6,
                                    l2_leaf_reg=5.0, rsm=0.4, random_seed=fs,
                                    verbose=False, allow_writing_files=False,
                                    early_stopping_rounds=200, cat_features=cat_cols)
            cm.fit(xa_c, y[a], eval_set=(xb_c, y[b]))
            oof["cat"][b] = cm.predict_proba(xb_c)[:, 1]
            xv_c, _ = for_catboost(Xv[cols], cols)
            pte["cat"] += cm.predict_proba(xv_c)[:, 1] / len(folds)

        for k in legs:
            auc, ks = metrics(y, oof[k])
            say(f"seed {fs} {k:5s} auc={auc:.6f} ks={ks:.6f}")

        oof_all[fs] = oof
        test_all[fs] = pte
        per_seed[fs] = {k: metrics(y, oof[k])[0] for k in legs}
        store[f"oofseed_{j}"] = np.array([oof[k] for k in legs])
        store[f"testseed_{j}"] = np.array([pte[k] for k in legs])

        np.savez_compressed(OUT / f"preds_{args.out_prefix}_partial.npz",
                            seeds=np.array(seeds[:j + 1]), legs=np.array(legs),
                            train_ids=np.array(tr.application_id),
                            test_ids=np.array(te.application_id), **store)
        say(f"saved after {j + 1}/{len(seeds)} fold seeds, "
            f"elapsed {(time.time() - t_start) / 60:.1f} min")

    bag = {k: np.mean([oof_all[s][k] for s in seeds], axis=0) for k in legs}
    bagt = {k: np.mean([test_all[s][k] for s in seeds], axis=0) for k in legs}

    summary = {"legs": legs, "fold_seeds": seeds, "top_n": args.top_n,
               "n_candidates": int(X.shape[1]),
               "minutes": (time.time() - t_start) / 60.0, "per_seed": per_seed}

    # Two axes are searched rather than one, because the two public results this
    # project has ever earned came from opposite ends of the same idea and only
    # one of them still works here.
    #
    #   v61 pushed a sharply selected model away from a broad one and scored
    #   0.56312. Reproduced as `sel - lean`.
    #
    #   qc_00 pushed a feature-rich model away from a feature-poor one and scored
    #   0.56171. Reproduced as `lean - sel`.
    #
    # At the v61 operating point more features was better, which is what made
    # both of those point the same way. At `lr 0.01 / lambda_l2 5` with 598
    # candidates the gradient has flipped: the selected 250-column model beats
    # the 576-column one. So the search runs both directions and reports what
    # each does, rather than assuming the old sign.
    rsel, rlean = rank01(bag["sel"]), rank01(bag["lean"])
    for name, (base, direction) in {
        "sel_minus_lean": (rsel, +1),
        "lean_minus_sel": (rlean, -1),
    }.items():
        for r in [0.25, 0.5, 0.75, 1.0, 1.5]:
            p = base + r * direction * (rsel - rlean)
            auc, ks = metrics(y, p)
            summary[f"axis_{name}_r{r}"] = {"auc": auc, "ks": ks}
            say(f"axis {name:14s} r={r:<4} auc={auc:.6f} ks={ks:.6f}")

    mixes = {
        "mix_trees": 0.5 * rsel + 0.25 * rank01(bag["xgb"]) + 0.25 * rank01(bag["cat"]),
        "mix_all4": 0.4 * rsel + 0.2 * rlean + 0.2 * rank01(bag["xgb"]) + 0.2 * rank01(bag["cat"]),
        "mix_sel_xgb": 0.6 * rsel + 0.4 * rank01(bag["xgb"]),
        "mix_sel_cat": 0.6 * rsel + 0.4 * rank01(bag["cat"]),
        "consensus_ranks": (rsel + rlean + rank01(bag["xgb"]) + rank01(bag["cat"])) / 4,
    }
    for k, p in mixes.items():
        auc, ks = metrics(y, p)
        summary[k] = {"auc": auc, "ks": ks}
        say(f"{k:24s} auc={auc:.6f} ks={ks:.6f}")

    for k in legs:
        auc, ks = metrics(y, bag[k])
        summary[f"bag_{k}"] = {"auc": auc, "ks": ks}
        say(f"BAG {k:22s} auc={auc:.6f} ks={ks:.6f} "
            f"per-seed auc={[round(per_seed[s][k], 6) for s in seeds]}")

    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        seeds=np.array(seeds), legs=np.array(legs),
                        train_ids=np.array(tr.application_id),
                        test_ids=np.array(te.application_id),
                        **{f"oof_{k}": bag[k] for k in legs},
                        **{f"test_{k}": bagt[k] for k in legs},
                        **store)
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    say(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()