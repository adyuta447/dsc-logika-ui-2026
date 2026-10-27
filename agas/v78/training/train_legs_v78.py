"""v78 production legs: the v76 tuning, run long, with a memory profile that fits.

Identical to `agas/v76/training/train_legs_v76.py` - same `extra_trees=True` on
the three plain LightGBM legs, same 16000 round cap - with one change: **10
folds instead of 12**.

That change is the whole point. v76 chose 12 folds knowing the trainer caches
one concatenated feature frame per fold inside a fold seed, about 0.5 GB each,
and judged 5.8 GB safe on 16 GB. It was not: fold seed 42 finished in 143
minutes, fold seed 7 began swapping with swap at 15.2 of 16.4 GB, and its
`lgb_lean` leg alone took 98 minutes against 27 one seed earlier. The run was
stopped at one fold seed of five. Ten folds is v66's own profile, which
completed six fold seeds without swapping, so it is the profile that is known
to fit.

Defaults here are 10 folds, six fold seeds and three booster seeds. Six fold
seeds matches v66 exactly, so the bagging is directly comparable; three booster
seeds rather than five because `extra_trees` already randomises the split
thresholds, and five would push the run past ten hours. A checkpoint is written
after every fold seed, so the run can be stopped at any point and whatever is
complete stays usable - ranks are scale-invariant, so a partial sum divided by
the full seed count ranks identically to the proper average.

Leakage control is unchanged: historical aggregates are cut at
`decision_datetime`, factor features are recomputed inside each fold from
training-fold labels only, and the test set is never used for fitting.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "agas/shared"))
sys.path.insert(0, str(HERE.parents[2]))
os.chdir(HERE.parents[2])

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE.parent / "output"

# copied verbatim from v61 so the column sets are the ones that scored 0.56312
NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")
ROUND2_PREFIX = ("r2_", "b2_", "a2_", "e2_")

LEGS = ["fac_full", "lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear"]


def rank01(v):
    return rankdata(v) / len(v)


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13,21,5,101")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--lgb-seeds", type=str, default="0,1,2")
    ap.add_argument("--out-prefix", type=str, default="v78legs")
    return ap.parse_args()


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
    return X, Xte, tr.application_id.values, te.application_id.values


def main():
    args = parse()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import features as ft

    OUT.mkdir(parents=True, exist_ok=True)
    log_file = OUT / f"train_log_{args.out_prefix}.txt"
    if log_file.exists():
        log_file.unlink()
    t_start = time.time()

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(log_file, "a") as fh:
            fh.write(line + "\n")

    X, Xte, train_ids, test_ids = load_frames()
    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([train_ids, test_ids])].values
    y_app = app.target.values
    n_tr = len(train_ids)
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]

    full = list(X.columns)
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    fac_cols = [c for c in full if not c.startswith(ROUND2_PREFIX)]
    log(f"feature counts: full={len(full)} lean={len(lean)} fac={len(fac_cols)}")
    log(f"folds={args.folds} fold_seeds={fold_seeds} lgb_seeds={lgb_seeds}")

    log("computing factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor inputs ready")

    def lgb_params(seed, extra=False):
        if extra:
            return dict(objective="binary", learning_rate=args.lr * 1.2, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35, bagging_fraction=0.85,
                        bagging_freq=1, lambda_l2=args.l2 * 2.4, max_bin=255, seed=seed,
                        deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        # the screened winner: v61's settings plus randomised split thresholds
        return dict(objective="binary", learning_rate=args.lr, num_leaves=15,
                    min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=args.l2, max_bin=127, seed=seed,
                    extra_trees=True,
                    deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)

    oof_all = {k: np.zeros(n_tr) for k in LEGS}
    test_all = {k: np.zeros(len(Xte)) for k in LEGS}
    seed_oof = {}

    for fs in fold_seeds:
        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                      random_state=fs).split(X, y_app[:n_tr]))
        fac_cols_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        y = y_app[:n_tr].astype(int)

        cache = {}

        def frame(k):
            if k not in cache:
                fk = fac_cols_k[k]
                cache[k] = (pd.concat([X.reset_index(drop=True),
                                       fk.iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1),
                            pd.concat([Xte.reset_index(drop=True),
                                       fk.iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1))
            return cache[k]

        def run_lgb(cols_by_fold, seed, extra=False):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                cols = cols_by_fold[k]
                m = lgb.train(lgb_params(seed, extra), lgb.Dataset(xk.iloc[a][cols], y[a]),
                              num_boost_round=16000 if not extra else 8000,
                              valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                              callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(xk.iloc[b][cols])
                pred += m.predict(xt[cols]) / len(folds)
            return oof, pred

        def run_cat(cols_by_fold, seed=0):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                cols = cols_by_fold[k]
                x = xk[cols].copy()
                xt2 = xt[cols].copy()
                cats = [c for c in cols if str(x[c].dtype) == "category"]
                for c in cats:
                    x[c] = x[c].astype(str)
                    xt2[c] = xt2[c].astype(str)
                m = CatBoostClassifier(iterations=4000, learning_rate=3.0 * args.lr, depth=6,
                                       l2_leaf_reg=5, eval_metric="AUC", od_type="Iter",
                                       od_wait=300, random_seed=seed, cat_features=cats, verbose=0)
                m.fit(x.iloc[a], y[a], eval_set=(x.iloc[b], y[b]))
                oof[b] = m.predict_proba(x.iloc[b])[:, 1]
                pred += m.predict_proba(xt2)[:, 1] / len(folds)
            return oof, pred

        def run_xgb(cols_by_fold, seed=0):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                cols = cols_by_fold[k]
                m = xgb.XGBClassifier(n_estimators=9000, learning_rate=args.lr, max_depth=4,
                                      min_child_weight=20, subsample=0.8, colsample_bytree=0.3,
                                      reg_lambda=args.l2 * 4, tree_method="hist", max_bin=128,
                                      enable_categorical=True, eval_metric="auc",
                                      early_stopping_rounds=300, random_state=seed, n_jobs=-1)
                m.fit(xk.iloc[a][cols], y[a], eval_set=[(xk.iloc[b][cols], y[b])], verbose=False)
                oof[b] = m.predict_proba(xk.iloc[b][cols])[:, 1]
                pred += m.predict_proba(xt[cols])[:, 1] / len(folds)
            return oof, pred

        def run_linear(cols):
            num = [c for c in cols if str(X[c].dtype) != "category"]
            # A column that is entirely missing has a NaN median, and fillna with
            # a NaN median leaves the NaN in place, which SplineTransformer
            # rejects. Two v65 `af_` columns are all-NaN, so this is not
            # hypothetical. Dropping them also keeps the linear leg on the same
            # column set v61 used, which is the point of this run.
            num = [c for c in num if X[c].notna().any()]
            flag = [c for c in num if X[c].isna().mean() > 0.01]

            def mat(x, med):
                return pd.concat([x[num].astype(float).fillna(med),
                                  x[flag].isna().astype(float).add_suffix("_na")], axis=1)

            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                med = xk.iloc[a][num].median()
                m = make_pipeline(
                    QuantileTransformer(n_quantiles=200, subsample=10 ** 6, random_state=fs),
                    SplineTransformer(n_knots=4, degree=2),
                    LogisticRegression(C=0.03, max_iter=3000))
                m.fit(mat(xk.iloc[a], med), y[a])
                oof[b] = m.predict_proba(mat(xk.iloc[b], med))[:, 1]
                pred += m.predict_proba(mat(xt, med))[:, 1] / len(folds)
            return oof, pred

        jobs = [
            ("fac_full", [fac_cols] * len(folds), "lgb"),
            ("lgb_full", [full] * len(folds), "lgb"),
            ("lgb_lean", [lean] * len(folds), "lgb"),
            ("lgbx_full", [full] * len(folds), "lgbx"),
            ("cat_full", [full] * len(folds), "cat"),
            ("xgb_full", [full] * len(folds), "xgb"),
            ("linear", lean, "lin"),
        ]
        per_seed = {}
        for key, cols_arg, kind in jobs:
            if kind == "lgb":
                outs = [run_lgb(cols_arg, sd) for sd in lgb_seeds]
                o = np.mean([a for a, _ in outs], axis=0)
                p = np.mean([b for _, b in outs], axis=0)
            else:
                o, p = (run_lgb(cols_arg, 99, True) if kind == "lgbx"
                        else run_cat(cols_arg) if kind == "cat"
                        else run_xgb(cols_arg) if kind == "xgb" else run_linear(cols_arg))
            oof_all[key] += o / len(fold_seeds)
            test_all[key] += p / len(fold_seeds)
            per_seed[key] = float(roc_auc_score(y, o))
            log(f"seed {fs} {key:10s} auc={per_seed[key]:.6f} ks={ks_score(y, o):.6f}")
        seed_oof[fs] = per_seed

        np.savez_compressed(OUT / f"preds_{args.out_prefix}_partial.npz",
                            train_ids=train_ids, test_ids=test_ids,
                            **{f"oof_{k}": v for k, v in oof_all.items()},
                            **{f"test_{k}": v for k, v in test_all.items()},
                            **{f"seed_oof_{list(seed_oof)[-1]}":
                               np.array([seed_oof[list(seed_oof)[-1]][k] for k in LEGS])})
        log(f"saved after {len(seed_oof)}/{len(fold_seeds)} fold seeds, "
            f"elapsed {(time.time() - t_start) / 60:.1f} min")

    y = y_app[:n_tr].astype(int)
    summary = {"fold_seeds": fold_seeds, "lgb_seeds": lgb_seeds,
               "folds": args.folds, "n_full": len(full), "n_lean": len(lean),
               "n_fac": len(fac_cols), "minutes": (time.time() - t_start) / 60.0,
               "per_seed": seed_oof}
    for k in LEGS:
        summary[k] = {"auc": float(roc_auc_score(y, oof_all[k])),
                      "ks": float(ks_score(y, oof_all[k]))}
        log(f"BAG {k:10s} auc={summary[k]['auc']:.6f} ks={summary[k]['ks']:.6f}")

    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        train_ids=train_ids, test_ids=test_ids,
                        **{f"oof_{k}": v for k, v in oof_all.items()},
                        **{f"test_{k}": v for k, v in test_all.items()})
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {(time.time() - t_start) / 60.1:.1f} min")


if __name__ == "__main__":
    main()