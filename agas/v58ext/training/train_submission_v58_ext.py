"""
Replicates the team's historically best-scoring recipe (public 0.56171, PROGRESS.md
"ekstrapolasi 1.5 v4 dikurangi fac, full_only") instead of inventing a new blend axis.

Method, exactly as found in experiments/harvest_kf.py:
  ext(name, hi, lo, a) = rank(O[hi] + a * (O[hi] - O[lo]))
with hi = "v4|full_only" (trees on the full column set, WITH the 4 v4 feature groups:
season day-of-year, salary share, opening-balance ratio, application/purchase gaps)
and  lo = "fac|full_only" (same trees, WITHOUT those v4 columns, factor model still included).

a=1.5 is the proven ratio (public 0.56171). This script rebuilds both legs with a
slightly stronger ensemble (3 fold seeds, 3 LightGBM seeds, CatBoost, XGBoost per leg)
and also writes a=1.0 / a=2.0 as alternates, since public LB noise means the proven
ratio is not guaranteed optimal on the next split.

Does not touch features.py or any existing file; reads from the existing agas/output
feature cache if present.
"""
import sys
import argparse
import json
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

V4_ONLY_COLS = ["dec_doy", "ab_inc_share", "ab_open_avg", "pa_gap_mean", "ec_gap_mean", "ec_gap_last3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold-seeds", type=str, default="42,7,13")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lgb-seeds", type=str, default="0,1,2")
    p.add_argument("--ext-ratios", type=str, default="1.0,1.5,2.0")
    p.add_argument("--out-prefix", type=str, default="v58ext")
    return p.parse_args()


def rank01(v):
    return rankdata(v) / len(v)


def ks_score(y_true, pred):
    return ks_2samp(pred[y_true == 1], pred[y_true == 0]).statistic


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(__import__("shared.bootstrap", fromlist=["x"]).add_shared(__file__)))
    import paths  # noqa: E402
    root = paths.ROOT
    import os
    os.chdir(root)
    import sys
    sys.path.insert(0, str(root))
    import features as ft  # noqa: E402

    out_dir = here.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"train_log_{args.out_prefix}.txt"
    if log_file.exists():
        log_file.unlink()

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    started = time.time()
    fold_seeds = [int(x) for x in args.fold_seeds.split(",") if x.strip()]
    lgb_seeds = [int(x) for x in args.lgb_seeds.split(",") if x.strip()]
    ext_ratios = [float(x) for x in args.ext_ratios.split(",") if x.strip()]
    log(f"start: fold_seeds={fold_seeds} folds={args.folds} lr={args.lr} lgb_seeds={lgb_seeds} ext_ratios={ext_ratios}")

    cache_file = paths.FEATURES_CACHE
    if cache_file.exists():
        f = pd.read_pickle(cache_file)
        log(f"loaded cache: {cache_file}")
    else:
        f = ft.build()
        f.to_pickle(cache_file)
        log(f"built features and cached: {cache_file}")

    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    y = tr.target.astype(int).values
    ids = ["application_id", "customer_id", "target"]
    X_all = tr.drop(columns=ids)
    X_test_all = te.drop(columns=ids)
    for c in X_all.columns:
        if str(X_all[c].dtype) == "object":
            X_all[c] = X_all[c].astype("category")
            X_test_all[c] = X_test_all[c].astype("category")

    full_cols = list(X_all.columns)
    fac_cols = [c for c in full_cols if c not in V4_ONLY_COLS]
    log(f"full_cols={len(full_cols)} fac_cols(no v4 extras)={len(fac_cols)} dropped={V4_ONLY_COLS}")

    app = ft.load_apps()
    M, E = ft.factor_inputs(app)
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_tr = pos[tr.application_id].values
    p_te = pos[te.application_id].values

    def report(name, pred):
        auc = roc_auc_score(y, pred)
        ks = ks_score(y, pred)
        log(f"{name:28s} auc={auc:.6f} ks={ks:.6f}")
        return {"auc": float(auc), "ks": float(ks)}

    # per fold-seed, per variant (v4 / fac): rank-averaged trees (lgb x N seeds, cat, xgb)
    variant_oof = {"v4": [], "fac": []}
    variant_pred = {"v4": [], "fac": []}

    for seed in fold_seeds:
        log(f"=== fold_seed {seed} ===")
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        folds = list(skf.split(X_all, y))
        fac_feat = [ft.factor_features(M, E, app.target.values, p_tr[a]) for a, _ in folds]

        def frame(k):
            x = pd.concat([X_all.reset_index(drop=True), fac_feat[k].iloc[p_tr].reset_index(drop=True)], axis=1)
            xt = pd.concat([X_test_all.reset_index(drop=True), fac_feat[k].iloc[p_te].reset_index(drop=True)], axis=1)
            for c in x.columns:
                if str(x[c].dtype) == "object":
                    x[c] = x[c].astype("category")
                    xt[c] = xt[c].astype("category")
            return x, xt

        fac_all_cols = fac_cols + list(fac_feat[0].columns)
        full_all_cols = full_cols + list(fac_feat[0].columns)
        cols_by_variant = {"v4": full_all_cols, "fac": fac_all_cols}

        def fit_lgb(cols, seed_lgb):
            params = dict(objective="binary", learning_rate=args.lr, num_leaves=15, min_child_samples=200,
                          feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=127,
                          seed=seed_lgb, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
            oof = np.zeros(len(X_all))
            pred = np.zeros(len(X_test_all))
            for k, (a, b) in enumerate(folds):
                x, xt = frame(k)
                m = lgb.train(params, lgb.Dataset(x.iloc[a][cols], y[a]), 12000,
                               valid_sets=[lgb.Dataset(x.iloc[b][cols], y[b])],
                               callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(x.iloc[b][cols])
                pred += m.predict(xt[cols]) / len(folds)
            return oof, pred

        def fit_cat(cols, seed_cat):
            oof = np.zeros(len(X_all))
            pred = np.zeros(len(X_test_all))
            for k, (a, b) in enumerate(folds):
                x, xt = frame(k)
                xk, xtk = x[cols].copy(), xt[cols].copy()
                cats = [c for c in xk.columns if str(xk[c].dtype) == "category"]
                for c in cats:
                    xk[c] = xk[c].astype(str)
                    xtk[c] = xtk[c].astype(str)
                m = CatBoostClassifier(iterations=int(70 / args.lr), learning_rate=2.5 * args.lr, depth=6,
                                       l2_leaf_reg=5, eval_metric="AUC", od_type="Iter", od_wait=300,
                                       random_seed=seed_cat, cat_features=cats, verbose=0)
                m.fit(xk.iloc[a], y[a], eval_set=(xk.iloc[b], y[b]))
                oof[b] = m.predict_proba(xk.iloc[b])[:, 1]
                pred += m.predict_proba(xtk)[:, 1] / len(folds)
            return oof, pred

        def fit_xgb(cols, seed_xgb):
            oof = np.zeros(len(X_all))
            pred = np.zeros(len(X_test_all))
            for k, (a, b) in enumerate(folds):
                x, xt = frame(k)
                xk, xtk = x[cols], xt[cols]
                m = xgb.XGBClassifier(n_estimators=9000, learning_rate=args.lr, max_depth=4, min_child_weight=20,
                                      subsample=0.8, colsample_bytree=0.3, reg_lambda=20, tree_method="hist",
                                      max_bin=128, enable_categorical=True, eval_metric="auc",
                                      early_stopping_rounds=300, random_state=seed_xgb, n_jobs=-1)
                m.fit(xk.iloc[a], y[a], eval_set=[(xk.iloc[b], y[b])], verbose=False)
                oof[b] = m.predict_proba(xk.iloc[b])[:, 1]
                pred += m.predict_proba(xtk)[:, 1] / len(folds)
            return oof, pred

        for variant, cols in cols_by_variant.items():
            log(f"-- variant {variant} (seed {seed}), {len(cols)} cols --")
            lgb_oofs, lgb_preds = [], []
            for s in lgb_seeds:
                oof, pred = fit_lgb(cols, s)
                lgb_oofs.append(rank01(oof))
                lgb_preds.append(rank01(pred))
                report(f"{variant}_s{seed}_lgb_s{s}", oof)
            lgb_oof_avg = np.mean(lgb_oofs, axis=0)
            lgb_pred_avg = np.mean(lgb_preds, axis=0)

            cat_oof, cat_pred = fit_cat(cols, 0)
            report(f"{variant}_s{seed}_cat", cat_oof)

            xgb_oof, xgb_pred = fit_xgb(cols, 0)
            report(f"{variant}_s{seed}_xgb", xgb_oof)

            trees_oof = (lgb_oof_avg + rank01(cat_oof) + rank01(xgb_oof)) / 3
            trees_pred = (lgb_pred_avg + rank01(cat_pred) + rank01(xgb_pred)) / 3
            report(f"{variant}_s{seed}_trees_full_only", trees_oof)

            variant_oof[variant].append(trees_oof)
            variant_pred[variant].append(trees_pred)

    v4_oof = rank01(np.mean([rank01(o) for o in variant_oof["v4"]], axis=0))
    v4_pred = rank01(np.mean([rank01(p) for p in variant_pred["v4"]], axis=0))
    fac_oof = rank01(np.mean([rank01(o) for o in variant_oof["fac"]], axis=0))
    fac_pred = rank01(np.mean([rank01(p) for p in variant_pred["fac"]], axis=0))

    metrics = {
        "v4_full_only": report("v4_full_only (bag of fold seeds)", v4_oof),
        "fac_full_only": report("fac_full_only (bag of fold seeds)", fac_oof),
    }

    test_ids = te.application_id.values
    for a in ext_ratios:
        ext_oof = rank01(v4_oof + a * (v4_oof - fac_oof))
        ext_pred = rank01(v4_pred + a * (v4_pred - fac_pred))
        name = f"ext{a:g}_v4_fac_full_only"
        metrics[name] = report(name, ext_oof)
        sub = pd.DataFrame({"application_id": test_ids, "default_probability": ext_pred})
        assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
        sub.to_csv(out_dir / f"submission_{args.out_prefix}_{name}.csv", index=False)

    # also save the two raw legs, useful as fallback if extrapolation hurts public again
    for name, pred in (("v4_full_only", v4_pred), ("fac_full_only", fac_pred)):
        sub = pd.DataFrame({"application_id": test_ids, "default_probability": pred})
        sub.to_csv(out_dir / f"submission_{args.out_prefix}_{name}.csv", index=False)

    best_name = max(
        [k for k in metrics if k.startswith("ext")] or list(metrics),
        key=lambda k: metrics[k]["ks"],
    )
    log(f"best ext variant by OOF KS: {best_name} -> {metrics[best_name]}")

    summary = {
        "params": {
            "fold_seeds": fold_seeds,
            "folds": args.folds,
            "lr": args.lr,
            "lgb_seeds": lgb_seeds,
            "ext_ratios": ext_ratios,
            "v4_only_cols_dropped_for_fac": V4_ONLY_COLS,
        },
        "metrics": metrics,
        "best_ext_by_oof_ks": best_name,
        "proven_reference": {
            "note": "a=1.5 scored public 0.56171 in the original run (team's best so far)",
            "file": "agas/qc_00_ext1.5_v4_fac_full_only.csv",
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    summary_file = out_dir / f"summary_{args.out_prefix}.json"
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"summary: {summary_file}")
    log(f"elapsed_min: {summary['elapsed_minutes']:.2f}")


if __name__ == "__main__":
    main()
