"""v68: the v61/v66 recipe, extended to ten fold seeds.

v66 trained six independent 10-fold partitions and averaged them. Its public
winner, `ext0.5|tm30|lgb_lean`, scored 0.56424, above v61's 0.56312. This run
keeps the architecture byte-for-byte identical and adds four more partitions:
3, 17, 11, and 23.

The six v66 partitions do not need to be retrained. v66 stored the equal-weight
average of its six per-seed predictions, so the ten-seed average is exactly

    (6 * v66_average + sum(new_seed_predictions)) / 10

and the new script accumulates that sum directly. Every leg, column set,
parameter, factor computation, and blend formula is unchanged. The only new
object is a partition, not a model.

A partial file stores each finished new seed separately, so an interrupted run
resumes at the next seed instead of restarting four hours of work.

Leakage control is inherited and unchanged: historical aggregates are cut at
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

OUT = HERE.parent / "output"

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
    ap.add_argument("--fold-seeds", type=str, default="3,17,11,23",
                    help="new partitions only; v66's six seeds are reused")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--lgb-seeds", type=str, default="0,1")
    ap.add_argument("--base-prefix", type=str, default="v66")
    ap.add_argument("--base-seed-count", type=int, default=6)
    ap.add_argument("--out-prefix", type=str, default="v68")
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
    n_tr, n_te = len(train_ids), len(test_ids)

    base_out = paths.version_output_of(args.base_prefix)
    base_npz_path = base_out / f"preds_{args.base_prefix}.npz"
    base_summary_path = base_out / f"summary_{args.base_prefix}.json"
    base = np.load(base_npz_path, allow_pickle=True)
    base_summary = json.loads(base_summary_path.read_text())
    assert list(base["train_ids"]) == list(train_ids), "base train ids differ"
    assert list(base["test_ids"]) == list(test_ids), "base test ids differ"
    assert len(base_summary["fold_seeds"]) == args.base_seed_count, (
        f"base has {len(base_summary['fold_seeds'])} seeds, "
        f"--base-seed-count says {args.base_seed_count}")
    assert base_summary["folds"] == args.folds, (
        f"base folds={base_summary['folds']}, run folds={args.folds}")
    assert base_summary["lgb_seeds"] == [int(s) for s in args.lgb_seeds.split(",")], (
        "base LightGBM seeds differ from this run")

    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([train_ids, test_ids])].values
    y_app = app.target.values
    y = y_app[:n_tr].astype(int)

    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
    new_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    total_seeds = args.base_seed_count + len(new_seeds)
    log(f"base={args.base_prefix} ({args.base_seed_count} seeds) "
        f"new={new_seeds} total={total_seeds}")

    full = list(X.columns)
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    fac_cols = [c for c in full if not c.startswith(ROUND2_PREFIX)]
    log(f"feature counts: full={len(full)} lean={len(lean)} fac={len(fac_cols)}")

    agg_oof = {k: base[f"oof_{k}"].astype(float) * args.base_seed_count for k in LEGS}
    agg_test = {k: base[f"test_{k}"].astype(float) * args.base_seed_count for k in LEGS}

    partial_path = OUT / f"preds_{args.out_prefix}_partial.npz"
    seed_oof_arrays = {}
    seed_test_arrays = {}
    seed_oof = {}
    done = []
    if partial_path.exists():
        p = np.load(partial_path, allow_pickle=True)
        same_run = (str(p["base_prefix"]) == args.base_prefix
                    and int(p["base_seed_count"]) == args.base_seed_count
                    and [int(s) for s in p["new_fold_seeds"]] == new_seeds)
        if not same_run:
            raise SystemExit(f"{partial_path} belongs to a different configuration; "
                             "remove it or use another --out-prefix")
        done = [int(s) for s in p["done_fold_seeds"]]
        for fs in done:
            seed_oof[fs] = {}
            for k in LEGS:
                seed_oof_arrays[(fs, k)] = p[f"seed_{fs}_oof_{k}"]
                seed_test_arrays[(fs, k)] = p[f"seed_{fs}_test_{k}"]
                agg_oof[k] += p[f"seed_{fs}_oof_{k}"]
                agg_test[k] += p[f"seed_{fs}_test_{k}"]
                seed_oof[fs][k] = float(roc_auc_score(y, p[f"seed_{fs}_oof_{k}"]))
        log(f"resumed partial with completed seeds {done}")

    log("computing factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor inputs ready")

    def lgb_params(seed, extra=False):
        if extra:
            return dict(objective="binary", learning_rate=args.lr * 1.2, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35, bagging_fraction=0.85,
                        bagging_freq=1, lambda_l2=args.l2 * 2.4, max_bin=255, seed=seed,
                        deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        return dict(objective="binary", learning_rate=args.lr, num_leaves=15,
                    min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=args.l2, max_bin=127, seed=seed,
                    deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)

    for fs in new_seeds:
        if fs in done:
            log(f"skip fold seed {fs}, already in partial")
            continue

        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                      random_state=fs).split(X, y))
        fac_cols_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
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
            pred = np.zeros(n_te)
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                cols = cols_by_fold[k]
                m = lgb.train(lgb_params(seed, extra), lgb.Dataset(xk.iloc[a][cols], y[a]),
                              num_boost_round=12000 if not extra else 8000,
                              valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                              callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(xk.iloc[b][cols])
                pred += m.predict(xt[cols]) / len(folds)
            return oof, pred

        def run_cat(cols_by_fold, seed=0):
            oof = np.zeros(n_tr)
            pred = np.zeros(n_te)
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
            pred = np.zeros(n_te)
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
            num = [c for c in num if X[c].notna().any()]
            flag = [c for c in num if X[c].isna().mean() > 0.01]

            def mat(x, med):
                return pd.concat([x[num].astype(float).fillna(med),
                                  x[flag].isna().astype(float).add_suffix("_na")], axis=1)

            oof = np.zeros(n_tr)
            pred = np.zeros(n_te)
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
            per_seed[key] = float(roc_auc_score(y, o))
            seed_oof_arrays[(fs, key)] = o
            seed_test_arrays[(fs, key)] = p
            agg_oof[key] += o
            agg_test[key] += p
            log(f"seed {fs} {key:10s} auc={per_seed[key]:.6f} ks={ks_score(y, o):.6f}")

        seed_oof[fs] = per_seed
        done.append(fs)
        np.savez_compressed(
            partial_path,
            base_prefix=np.array(args.base_prefix),
            base_seed_count=np.array(args.base_seed_count),
            new_fold_seeds=np.array(new_seeds),
            done_fold_seeds=np.array(done),
            **{f"seed_{fs2}_oof_{k}": seed_oof_arrays[(fs2, k)]
               for fs2 in done for k in LEGS},
            **{f"seed_{fs2}_test_{k}": seed_test_arrays[(fs2, k)]
               for fs2 in done for k in LEGS})
        log(f"saved after {len(done)}/{len(new_seeds)} new fold seeds, "
            f"elapsed {(time.time() - t_start) / 60:.1f} min")

    oof_final = {k: agg_oof[k] / total_seeds for k in LEGS}
    test_final = {k: agg_test[k] / total_seeds for k in LEGS}

    summary = {
        "base_prefix": args.base_prefix,
        "base_seed_count": args.base_seed_count,
        "new_fold_seeds": new_seeds,
        "total_seed_count": total_seeds,
        "folds": args.folds,
        "lgb_seeds": lgb_seeds,
        "n_full": len(full),
        "n_lean": len(lean),
        "n_fac": len(fac_cols),
        "minutes": (time.time() - t_start) / 60.0,
        "per_seed": seed_oof,
    }
    for k in LEGS:
        summary[k] = {"auc": float(roc_auc_score(y, oof_final[k])),
                      "ks": float(ks_score(y, oof_final[k]))}
        log(f"BAG {k:10s} auc={summary[k]['auc']:.6f} ks={summary[k]['ks']:.6f}")

    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        train_ids=train_ids, test_ids=test_ids,
                        **{f"oof_{k}": v for k, v in oof_final.items()},
                        **{f"test_{k}": v for k, v in test_final.items()})
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
