"""v61 ensemble: 10-fold, three fold seeds, LightGBM / CatBoost / XGBoost /
spline-linear over the original feature frame plus the new blocks in
`feat_extra.py`.

Two feature variants are trained so the historically proven extrapolation
direction can be reproduced:

- `v4e`  : every feature, including the round 1 and round 2 extra blocks.
- `fac`  : the same frame without the round 2 blocks (`r2_`, `b2_`, `a2_`, `e2_`).

Every model's OOF and test prediction is written to
`agas/output/preds_v61.npz` so blends can be explored later without retraining.
Nothing outside `agas/` is modified and no label information leaks into a
validation fold.
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
def _add_shared():
    """Load agas/shared/bootstrap.py by path; agas is not an importable package."""
    import importlib.util as _il
    _here = Path(__file__).resolve().parent
    for _p in _here.parents:
        _c = _p / "shared" / "bootstrap.py"
        if _c.exists():
            _spec = _il.spec_from_file_location("_agas_bootstrap", _c)
            _m = _il.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            if str(_c.parent) not in sys.path:
                sys.path.insert(0, str(_c.parent))
            return _p
    raise RuntimeError(f"shared/bootstrap.py not found above {_here}")

_SHARED = _add_shared()
import paths  # noqa: E402
ROOT = paths.ROOT
OUT = HERE.parent / "output"

NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")
ROUND2_PREFIX = ("r2_", "b2_", "a2_", "e2_")


def rank01(v):
    return rankdata(v) / len(v)


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--lgb-seeds", type=str, default="0,1,2")
    ap.add_argument("--out-prefix", type=str, default="v61")
    ap.add_argument("--no-top", action="store_true")
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

    log_file = OUT / f"train_log_{args.out_prefix}.txt"
    if log_file.exists():
        log_file.unlink()
    t_start = time.time()

    def log(m):
        print(m, flush=True)
        with open(log_file, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    X, Xte, train_ids, test_ids = load_frames()
    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    all_ids = np.concatenate([train_ids, test_ids])
    p_all = pos[all_ids].values
    y_app = app.target.values

    n_tr = len(train_ids)
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]

    full = list(X.columns)
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    fac_cols = [c for c in full if not c.startswith(ROUND2_PREFIX)]
    log(f"feature counts: full={len(full)} lean={len(lean)} fac={len(fac_cols)}")

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

    log("computing factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor inputs ready")

    oof_all = {k: np.zeros(n_tr) for k in [
        "lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear", "fac_full"]}
    test_all = {k: np.zeros(len(Xte)) for k in oof_all}

    for fs in fold_seeds:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=fs)
        folds = list(skf.split(X, y_app[:n_tr]))
        fac_cols_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        fac_names = list(fac_cols_k[0].columns)
        y = y_app[:n_tr].astype(int)

        def frame(k):
            fk = fac_cols_k[k]
            a = pd.concat([X.reset_index(drop=True), fk.iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1)
            b = pd.concat([Xte.reset_index(drop=True), fk.iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1)
            return a, b

        cache_frames = {}

        def get_frame(k):
            if k not in cache_frames:
                cache_frames[k] = frame(k)
            return cache_frames[k]

        def run_lgb(cols_by_fold, seed, extra=False):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
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
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                cols = cols_by_fold[k]
                x = xk[cols].copy()
                xt2 = xt[cols].copy()
                cats = [c for c in cols if str(x[c].dtype) == "category"]
                for c in cats:
                    x[c] = x[c].astype(str)
                    xt2[c] = xt2[c].astype(str)
                m = CatBoostClassifier(iterations=4000, learning_rate=3.0 * args.lr, depth=6,
                                       l2_leaf_reg=5, eval_metric="AUC", od_type="Iter", od_wait=300,
                                       random_seed=seed, cat_features=cats, verbose=0)
                m.fit(x.iloc[a], y[a], eval_set=(x.iloc[b], y[b]))
                oof[b] = m.predict_proba(x.iloc[b])[:, 1]
                pred += m.predict_proba(xt2)[:, 1] / len(folds)
            return oof, pred

        def run_xgb(cols_by_fold, seed=0):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
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
            flag = [c for c in num if X[c].isna().mean() > 0.01]

            def mat(x, med):
                return pd.concat([x[num].astype(float).fillna(med),
                                  x[flag].isna().astype(float).add_suffix("_na")], axis=1)

            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                med = xk.iloc[a][num].median()
                m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10 ** 6, random_state=fs),
                                  SplineTransformer(n_knots=4, degree=2),
                                  LogisticRegression(C=0.03, max_iter=3000))
                m.fit(mat(xk.iloc[a], med), y[a])
                oof[b] = m.predict_proba(mat(xk.iloc[b], med))[:, 1]
                pred += m.predict_proba(mat(xt, med))[:, 1] / len(folds)
            return oof, pred

        full_k = [full] * len(folds)
        lean_k = [lean] * len(folds)
        fac_k = [fac_cols] * len(folds)
        jobs = [
            ("fac_full", fac_k, "lgb"),
            ("lgb_full", full_k, "lgb"),
            ("lgb_lean", lean_k, "lgb"),
            ("lgbx_full", full_k, "lgbx"),
            ("cat_full", full_k, "cat"),
            ("xgb_full", full_k, "xgb"),
            ("linear", lean, "lin"),
        ]
        for key, cols_arg, kind in jobs:
            if kind == "lgb":
                oo = [run_lgb(cols_arg, sd) for sd in lgb_seeds]
                o = np.mean([a for a, _ in oo], axis=0)
                p = np.mean([b for _, b in oo], axis=0)
            else:
                o, p = run_lgb(cols_arg, 99, True) if kind == "lgbx" else (
                    run_cat(cols_arg) if kind == "cat" else (
                        run_xgb(cols_arg) if kind == "xgb" else run_linear(cols_arg)))
            oof_all[key] += o / len(fold_seeds)
            test_all[key] += p / len(fold_seeds)
            log(f"seed {fs} {key:9s} auc={roc_auc_score(y, o):.6f} ks={ks_score(y, o):.6f} "
                f"| bagged auc={roc_auc_score(y, oof_all[key]):.6f}")

        np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                            test_ids=test_ids, train_ids=train_ids,
                            **{f"oof_{k}": v for k, v in oof_all.items()},
                            **{f"test_{k}": v for k, v in test_all.items()})
        log(f"saved partial preds, elapsed {(time.time() - t_start) / 60:.2f} min")

    y = y_app[:n_tr].astype(int)
    summary = {"params": vars(args), "minutes": (time.time() - t_start) / 60.0, "models": {}}
    for k in oof_all:
        s = {"auc": float(roc_auc_score(y, oof_all[k])), "ks": float(ks_score(y, oof_all[k]))}
        summary["models"][k] = s
        log(f"BAG {k:9s} auc={s['auc']:.6f} ks={s['ks']:.6f}")
    with open(OUT / f"summary_{args.out_prefix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"done in {(time.time() - t_start) / 60:.2f} min")


if __name__ == "__main__":
    main()
