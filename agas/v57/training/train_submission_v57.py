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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lgb-seeds", type=str, default="0,1,2")
    parser.add_argument("--out-prefix", type=str, default="v57")
    return parser.parse_args()


def ks_score(y_true, pred):
    return ks_2samp(pred[y_true == 1], pred[y_true == 0]).statistic


def rank01(values):
    return rankdata(values) / len(values)


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(__import__("shared.bootstrap", fromlist=["x"]).add_shared(__file__)))
    import paths  # noqa: E402
    root = paths.ROOT
    os.chdir(root)
    sys.path.insert(0, str(root))
    import features as ft  # noqa: E402

    out_dir = here.parent / "output"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cache_file = paths.FEATURES_CACHE
    log_file = out_dir / f"train_log_{args.out_prefix}.txt"
    started = time.time()

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if log_file.exists():
        log_file.unlink()

    log(f"start: folds={args.folds} fold_seed={args.fold_seed} lr={args.lr}")

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

    noise = ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents",
             "tenure_to_age", "income_per_dependent", "loan_purpose", "decision_lag_h", "app_hour", "app_dow"]
    noise += [c for c in X_all.columns if c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_"))]

    sets = {
        "full": list(X_all.columns),
        "lean": [c for c in X_all.columns if c not in noise],
    }

    app = ft.load_apps()
    M, E = ft.factor_inputs(app)
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_tr = pos[tr.application_id].values
    p_te = pos[te.application_id].values

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.fold_seed)
    folds = list(skf.split(X_all, y))
    fac = [ft.factor_features(M, E, app.target.values, p_tr[a]) for a, _ in folds]
    fac_cols = list(fac[0].columns)

    def frame(k):
        x = pd.concat([X_all.reset_index(drop=True), fac[k].iloc[p_tr].reset_index(drop=True)], axis=1)
        xt = pd.concat([X_test_all.reset_index(drop=True), fac[k].iloc[p_te].reset_index(drop=True)], axis=1)
        for c in x.columns:
            if str(x[c].dtype) == "object":
                x[c] = x[c].astype("category")
                xt[c] = xt[c].astype("category")
        return x, xt

    sets = {k: v + fac_cols for k, v in sets.items()}
    lgb_seeds = [int(x) for x in args.lgb_seeds.split(",") if x.strip()]

    res = {}
    lgb_gain = {k: np.zeros(len(sets["lean"])) for k in range(len(folds))}

    def fit_lgb(cols_by_fold, tag, seed, extra=False):
        if extra:
            params = dict(
                objective="binary",
                learning_rate=args.lr * 1.2,
                num_leaves=31,
                min_child_samples=120,
                feature_fraction=0.45,
                bagging_fraction=0.85,
                bagging_freq=1,
                lambda_l2=12,
                max_bin=255,
                seed=seed,
                deterministic=True,
                force_row_wise=True,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            params = dict(
                objective="binary",
                learning_rate=args.lr,
                num_leaves=15,
                min_child_samples=200,
                feature_fraction=0.3,
                bagging_fraction=0.8,
                bagging_freq=1,
                lambda_l2=20,
                max_bin=127,
                seed=seed,
                deterministic=True,
                force_row_wise=True,
                n_jobs=-1,
                verbose=-1,
            )
        oof = np.zeros(len(X_all))
        pred = np.zeros(len(X_test_all))
        models = []
        for k, (a, b) in enumerate(folds):
            x, xt = frame(k)
            cols = cols_by_fold[k]
            dtr = lgb.Dataset(x.iloc[a][cols], y[a])
            dva = lgb.Dataset(x.iloc[b][cols], y[b])
            m = lgb.train(
                params,
                dtr,
                num_boost_round=12000 if not extra else 8000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(300, verbose=False)],
            )
            oof[b] = m.predict(x.iloc[b][cols])
            pred += m.predict(xt[cols]) / len(folds)
            models.append(m)
            if (tag == "lean") and (not extra):
                gain_map = pd.Series(m.feature_importance(importance_type="gain"), index=cols)
                lgb_gain[k] += gain_map.reindex(sets["lean"]).fillna(0).values
        return oof, pred, models

    def fit_cat(cols_by_fold, tag, seed):
        oof = np.zeros(len(X_all))
        pred = np.zeros(len(X_test_all))
        for k, (a, b) in enumerate(folds):
            x, xt = frame(k)
            cols = cols_by_fold[k]
            x = x[cols].copy()
            xt = xt[cols].copy()
            cats = [c for c in x.columns if str(x[c].dtype) == "category"]
            for c in cats:
                x[c] = x[c].astype(str)
                xt[c] = xt[c].astype(str)
            m = CatBoostClassifier(
                iterations=int(70 / args.lr),
                learning_rate=2.5 * args.lr,
                depth=6,
                l2_leaf_reg=5,
                eval_metric="AUC",
                od_type="Iter",
                od_wait=300,
                random_seed=seed,
                cat_features=cats,
                verbose=0,
            )
            m.fit(x.iloc[a], y[a], eval_set=(x.iloc[b], y[b]))
            oof[b] = m.predict_proba(x.iloc[b])[:, 1]
            pred += m.predict_proba(xt)[:, 1] / len(folds)
        return oof, pred

    def fit_xgb(cols_by_fold, tag, seed):
        oof = np.zeros(len(X_all))
        pred = np.zeros(len(X_test_all))
        for k, (a, b) in enumerate(folds):
            x, xt = frame(k)
            cols = cols_by_fold[k]
            x = x[cols]
            xt = xt[cols]
            m = xgb.XGBClassifier(
                n_estimators=9000,
                learning_rate=args.lr,
                max_depth=4,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_lambda=20,
                tree_method="hist",
                max_bin=128,
                enable_categorical=True,
                eval_metric="auc",
                early_stopping_rounds=300,
                random_state=seed,
                n_jobs=-1,
            )
            m.fit(x.iloc[a], y[a], eval_set=[(x.iloc[b], y[b])], verbose=False)
            oof[b] = m.predict_proba(x.iloc[b])[:, 1]
            pred += m.predict_proba(xt)[:, 1] / len(folds)
        return oof, pred

    def fit_linear():
        num = [c for c in sets["lean"] if str(frame(0)[0][c].dtype) != "category"]
        flag = [c for c in num if frame(0)[0][c].isna().mean() > 0.01]

        def mat(x, med):
            return pd.concat(
                [x[num].astype(float).fillna(med), x[flag].isna().astype(float).add_suffix("_na")], axis=1
            )

        oof = np.zeros(len(X_all))
        pred = np.zeros(len(X_test_all))
        for k, (a, b) in enumerate(folds):
            x, xt = frame(k)
            med = x.iloc[a][num].median()
            m = make_pipeline(
                QuantileTransformer(n_quantiles=200, subsample=10**6, random_state=args.fold_seed),
                SplineTransformer(n_knots=4, degree=2),
                LogisticRegression(C=0.03, max_iter=3000),
            )
            m.fit(mat(x.iloc[a], med), y[a])
            oof[b] = m.predict_proba(mat(x.iloc[b], med))[:, 1]
            pred += m.predict_proba(mat(xt, med))[:, 1] / len(folds)
        return oof, pred

    def report(name, pred):
        auc = roc_auc_score(y, pred)
        ks = ks_score(y, pred)
        log(f"{name:22s} auc={auc:.6f} ks={ks:.6f}")
        return {"auc": float(auc), "ks": float(ks)}

    log("train: LGB full/lean")
    lgb_models_by_tag = {"full": {}, "lean": {}}
    for tag in ("full", "lean"):
        for s in lgb_seeds:
            key = f"lgb_{tag}_s{s}"
            oof, pred, models = fit_lgb([sets[tag]] * len(folds), tag, s, extra=False)
            res[key] = (oof, pred)
            lgb_models_by_tag[tag][s] = models
            report(key, oof)

    top_by_fold = []
    for k in range(len(folds)):
        g = pd.Series(lgb_gain[k], index=sets["lean"]).sort_values(ascending=False)
        top_by_fold.append(list(g.index[:90]))
    sets["top"] = top_by_fold[0]

    log("train: LGB top")
    for s in lgb_seeds:
        key = f"lgb_top_s{s}"
        oof, pred, _ = fit_lgb(top_by_fold, "top", s, extra=False)
        res[key] = (oof, pred)
        report(key, oof)

    log("train: CatBoost full/lean/top")
    for tag, cols in [("full", [sets["full"]] * len(folds)), ("lean", [sets["lean"]] * len(folds)), ("top", top_by_fold)]:
        key = f"cat_{tag}"
        res[key] = fit_cat(cols, tag, 0)
        report(key, res[key][0])

    log("train: XGBoost full/lean/top")
    for tag, cols in [("full", [sets["full"]] * len(folds)), ("lean", [sets["lean"]] * len(folds)), ("top", top_by_fold)]:
        key = f"xgb_{tag}"
        res[key] = fit_xgb(cols, tag, 0)
        report(key, res[key][0])

    log("train: extra LGB full/lean")
    for tag in ("full", "lean"):
        key = f"lgbx_{tag}"
        res[key] = fit_lgb([sets[tag]] * len(folds), tag, 99, extra=True)[:2]
        report(key, res[key][0])

    log("train: linear spline")
    res["linear"] = fit_linear()
    report("linear", res["linear"][0])

    def avg_rank(keys, idx):
        return np.mean([rank01(res[k][idx]) for k in keys], axis=0)

    lgb_full = avg_rank([f"lgb_full_s{s}" for s in lgb_seeds], 0)
    lgb_lean = avg_rank([f"lgb_lean_s{s}" for s in lgb_seeds], 0)
    lgb_top = avg_rank([f"lgb_top_s{s}" for s in lgb_seeds], 0)
    lgb_full_t = avg_rank([f"lgb_full_s{s}" for s in lgb_seeds], 1)
    lgb_lean_t = avg_rank([f"lgb_lean_s{s}" for s in lgb_seeds], 1)
    lgb_top_t = avg_rank([f"lgb_top_s{s}" for s in lgb_seeds], 1)

    cat_full, cat_lean, cat_top = rank01(res["cat_full"][0]), rank01(res["cat_lean"][0]), rank01(res["cat_top"][0])
    cat_full_t, cat_lean_t, cat_top_t = rank01(res["cat_full"][1]), rank01(res["cat_lean"][1]), rank01(res["cat_top"][1])
    xgb_full, xgb_lean, xgb_top = rank01(res["xgb_full"][0]), rank01(res["xgb_lean"][0]), rank01(res["xgb_top"][0])
    xgb_full_t, xgb_lean_t, xgb_top_t = rank01(res["xgb_full"][1]), rank01(res["xgb_lean"][1]), rank01(res["xgb_top"][1])
    lin, lin_t = rank01(res["linear"][0]), rank01(res["linear"][1])
    lgbx_full, lgbx_lean = rank01(res["lgbx_full"][0]), rank01(res["lgbx_lean"][0])
    lgbx_full_t, lgbx_lean_t = rank01(res["lgbx_full"][1]), rank01(res["lgbx_lean"][1])

    trees_full = (lgb_full + cat_full + xgb_full) / 3
    trees_lean = (lgb_lean + cat_lean + xgb_lean) / 3
    trees_top = (lgb_top + cat_top + xgb_top) / 3
    trees_full_t = (lgb_full_t + cat_full_t + xgb_full_t) / 3
    trees_lean_t = (lgb_lean_t + cat_lean_t + xgb_lean_t) / 3
    trees_top_t = (lgb_top_t + cat_top_t + xgb_top_t) / 3

    core = (trees_full + trees_lean + cat_full + cat_lean) / 4
    core_t = (trees_full_t + trees_lean_t + cat_full_t + cat_lean_t) / 4

    variants = {
        "top_mix": (0.8 * (core + trees_top) / 2 + 0.2 * lin, 0.8 * (core_t + trees_top_t) / 2 + 0.2 * lin_t),
        "linear30": (0.7 * (core + trees_top) / 2 + 0.3 * lin, 0.7 * (core_t + trees_top_t) / 2 + 0.3 * lin_t),
        "trees_only": ((core + trees_top) / 2, (core_t + trees_top_t) / 2),
        "extra31_mix": (0.75 * (core + trees_top) / 2 + 0.25 * rank01((lgbx_full + lgbx_lean) / 2),
                        0.75 * (core_t + trees_top_t) / 2 + 0.25 * rank01((lgbx_full_t + lgbx_lean_t) / 2)),
    }

    meta_cols = [
        trees_full, trees_lean, trees_top,
        cat_full, cat_lean, cat_top,
        xgb_full, xgb_lean, xgb_top,
        lin, lgbx_full, lgbx_lean,
    ]
    meta_cols_t = [
        trees_full_t, trees_lean_t, trees_top_t,
        cat_full_t, cat_lean_t, cat_top_t,
        xgb_full_t, xgb_lean_t, xgb_top_t,
        lin_t, lgbx_full_t, lgbx_lean_t,
    ]
    z = np.column_stack(meta_cols)
    zt = np.column_stack(meta_cols_t)

    meta_oof = np.zeros(len(y))
    meta_cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.fold_seed + 99)
    for a, b in meta_cv.split(z, y):
        m = LogisticRegression(C=0.05, max_iter=4000)
        m.fit(z[a], y[a])
        meta_oof[b] = m.predict_proba(z[b])[:, 1]
    m = LogisticRegression(C=0.05, max_iter=4000)
    m.fit(z, y)
    meta_pred = m.predict_proba(zt)[:, 1]
    variants["meta_logit"] = (meta_oof, meta_pred)

    best_name = None
    best_metrics = None
    metrics = {}
    for name, (oof, pred) in variants.items():
        stats = report(name, oof)
        metrics[name] = stats
        if (best_metrics is None) or (stats["ks"] > best_metrics["ks"]):
            best_name = name
            best_metrics = stats

    ranked = sorted(metrics.items(), key=lambda x: (x[1]["ks"], x[1]["auc"]), reverse=True)
    top1 = ranked[0][0]
    top2 = ranked[1][0]
    aggr_oof = rank01(1.45 * rank01(variants[top1][0]) - 0.45 * rank01(variants[top2][0]))
    aggr_pred = rank01(1.45 * rank01(variants[top1][1]) - 0.45 * rank01(variants[top2][1]))
    variants["aggressive_145"] = (aggr_oof, aggr_pred)
    aggr_stats = report("aggressive_145", aggr_oof)
    metrics["aggressive_145"] = aggr_stats
    if aggr_stats["ks"] > best_metrics["ks"]:
        best_name = "aggressive_145"
        best_metrics = aggr_stats

    test_ids = te.application_id.values

    for name, (_, pred) in variants.items():
        sub = pd.DataFrame({"application_id": test_ids, "default_probability": rank01(pred)})
        out_file = out_dir / f"submission_{args.out_prefix}_{name}.csv"
        sub.to_csv(out_file, index=False)

    best_file = out_dir / f"submission_{args.out_prefix}_best.csv"
    pd.DataFrame(
        {"application_id": test_ids, "default_probability": rank01(variants[best_name][1])}
    ).to_csv(best_file, index=False)

    summary = {
        "params": {
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "lr": args.lr,
            "lgb_seeds": lgb_seeds,
        },
        "best_variant": best_name,
        "best_metrics": best_metrics,
        "all_metrics": metrics,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "files": {
            "best_submission": str(best_file),
            "log": str(log_file),
            "feature_cache": str(cache_file),
        },
    }
    summary_file = out_dir / f"summary_{args.out_prefix}.json"
    with open(summary_file, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    log(f"best variant: {best_name}")
    log(f"best submission: {best_file}")
    log(f"summary: {summary_file}")
    log(f"elapsed_min: {summary['elapsed_minutes']:.2f}")


if __name__ == "__main__":
    main()
