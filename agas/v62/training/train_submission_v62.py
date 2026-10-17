"""v62 ensemble.

Motivated by two public results now on record: the original `v4 - fac`
extrapolation (`0.56171`) and the newer `topmix30 - lgb_lean` extrapolation
(`0.56312`). Both work the same way: take a model trained on a richer feature
set and push it away from the same model trained on a reduced set. So v62
lengthens that gradient to four steps and keeps every level, which lets the
extrapolation be aimed along a longer lever arm.

Feature richness ladder:

    full    555 columns, everything
    fac     drops the round 2 blocks (r2_, b2_, a2_, e2_)
    lean    drops the known-noise columns
    faclean drops both, the leanest level

Per-fold OOF for every leg is stored separately as well as bagged, so the
direction of each extrapolation axis can be checked for consistency across fold
seeds instead of being read off one noisy OOF vector.

Nothing outside `agas/` is modified. Factor loading comes only from the labels
of the training fold of each split, and every historical aggregate is cut at
`decision_datetime`.
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
    ap.add_argument("--fold-seeds", type=str, default="42,7,13,21")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--full-seeds", type=str, default="0,1")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--top-wide", type=int, default=350)
    ap.add_argument("--out-prefix", type=str, default="v62")
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
    t0 = time.time()

    def log(m):
        print(m, flush=True)
        with open(log_file, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {m}\n")

    X, Xte, train_ids, test_ids = load_frames()
    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([train_ids, test_ids])].values
    y_app = app.target.values
    n_tr = len(train_ids)

    full = list(X.columns)
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    fac = [c for c in full if not c.startswith(ROUND2_PREFIX)]
    faclean = [c for c in fac if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"ladder: full={len(full)} fac={len(fac)} lean={len(lean)} faclean={len(faclean)}")

    log("factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor ready")

    keys = ["lgb_full", "lgb_fac", "lgb_lean", "lgb_faclean", "lgb_top", "lgb_topwide",
            "lgbx_full", "cat_full", "xgb_full", "linear"]
    bag_o = {k: np.zeros(n_tr) for k in keys}
    bag_t = {k: np.zeros(len(Xte)) for k in keys}
    per_seed = {k: [] for k in keys}

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    full_seeds = [int(s) for s in args.full_seeds.split(",") if s.strip()]

    def params(seed, extra=False):
        if extra:
            return dict(objective="binary", learning_rate=args.lr * 1.2, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35, bagging_fraction=0.85,
                        bagging_freq=1, lambda_l2=args.l2 * 2.4, max_bin=255, seed=seed,
                        deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        return dict(objective="binary", learning_rate=args.lr, num_leaves=15,
                    min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=args.l2, max_bin=127, seed=seed,
                    deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)

    for fi, fs in enumerate(fold_seeds):
        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                     random_state=fs).split(X, y_app[:n_tr]))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        y = y_app[:n_tr].astype(int)
        cache = {}

        def get_frame(k):
            if k not in cache:
                fk = fac_k[k]
                cache[k] = (
                    pd.concat([X.reset_index(drop=True), fk.iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1),
                    pd.concat([Xte.reset_index(drop=True), fk.iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1),
                )
            return cache[k]

        def run_lgb(cols_by_fold, seed, extra=False, gather_gain=False):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            gains = []
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                cols = cols_by_fold[k]
                m = lgb.train(params(seed, extra), lgb.Dataset(xk.iloc[a][cols], y[a]),
                              num_boost_round=12000 if not extra else 8000,
                              valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                              callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(xk.iloc[b][cols])
                pred += m.predict(xt[cols]) / len(folds)
                if gather_gain:
                    gains.append(pd.Series(m.feature_importance(importance_type="gain"), index=cols))
            ranked = None
            if gather_gain:
                s = pd.concat(gains).groupby(level=0).sum().sort_values(ascending=False)
                ranked = list(s.index)
            return oof, pred, ranked

        def run_cat(cols_by_fold, seed=0):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                cols = cols_by_fold[k]
                x, xt2 = xk[cols].copy(), xt[cols].copy()
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

        rep = {name: [cols] * len(folds)
              for name, cols in (("full", full), ("fac", fac), ("lean", lean), ("faclean", faclean))}
        jobs = ["lgb_fac", "lgb_faclean", "lgb_lean", "lgbx_full", "cat_full",
                "xgb_full", "linear"]
        ranked = None
        for key in jobs:
            if key == "lgb_fac":
                oo = [run_lgb(rep["fac"], sd) for sd in [0]]
                o, p = oo[0][0], oo[0][1]
            elif key == "lgb_faclean":
                o, p, _ = run_lgb(rep["faclean"], 0)
            elif key == "lgb_lean":
                o, p, ranked = run_lgb(rep["lean"], 0, gather_gain=True)
            elif key == "lgbx_full":
                o, p, _ = run_lgb(rep["full"], 99, extra=True)
            elif key == "cat_full":
                o, p = run_cat(rep["full"])
            elif key == "xgb_full":
                o, p = run_xgb(rep["full"])
            else:
                o, p = run_linear(lean)
            bag_o[key] += o / len(fold_seeds)
            bag_t[key] += p / len(fold_seeds)
            per_seed[key].append(o)
            log(f"seed {fs} {key:11s} auc={roc_auc_score(y, o):.6f} ks={ks_score(y, o):.6f}")

        for key, cols, seeds in [("lgb_full", rep["full"], full_seeds)]:
            oo = [run_lgb(cols, sd) for sd in seeds]
            o = np.mean([a for a, _, _ in oo], axis=0)
            p = np.mean([b for _, b, _ in oo], axis=0)
            bag_o[key] += o / len(fold_seeds)
            bag_t[key] += p / len(fold_seeds)
            per_seed[key].append(o)
            log(f"seed {fs} {key:11s} auc={roc_auc_score(y, o):.6f} ks={ks_score(y, o):.6f}")

        ranked = ranked or lean
        for key, cols, nn in [("lgb_top", ranked[: args.top_n], args.top_n),
                              ("lgb_topwide", ranked[: args.top_wide], args.top_wide)]:
            o, p, _ = run_lgb([cols] * len(folds), 0)
            bag_o[key] += o / len(fold_seeds)
            bag_t[key] += p / len(fold_seeds)
            per_seed[key].append(o)
            log(f"seed {fs} {key:12s} auc={roc_auc_score(y, o):.6f} ks={ks_score(y, o):.6f} (top {nn})")

        done = fi + 1
        np.savez_compressed(
            OUT / f"preds_{args.out_prefix}.npz",
            test_ids=test_ids, train_ids=train_ids, seeds=np.array(fold_seeds),
            **{f"oof_{k}": bag_o[k] for k in keys},
            **{f"test_{k}": bag_t[k] for k in keys},
            **{f"oofseed_{k}_{j}": v for k in keys for j, v in enumerate(per_seed[k])},
        )
        log(f"saved after {done}/{len(fold_seeds)} fold seeds, elapsed {(time.time() - t0) / 60:.1f} min")

    y = y_app[:n_tr].astype(int)
    summary = {"params": vars(args), "minutes": (time.time() - t0) / 60.0, "models": {},
               "ladder": {"full": len(full), "fac": len(fac), "lean": len(lean), "faclean": len(faclean)}}
    for k in keys:
        s = {"auc": float(roc_auc_score(y, bag_o[k])), "ks": float(ks_score(y, bag_o[k]))}
        ps = [{"auc": float(roc_auc_score(y, v)), "ks": float(ks_score(y, v))} for v in per_seed[k]]
        s["per_seed"] = ps
        summary["models"][k] = s
        log(f"BAG {k:11s} auc={s['auc']:.6f} ks={s['ks']:.6f} "
            f"per-seed ks={[round(p['ks'], 5) for p in ps]}")
    with open(OUT / f"summary_{args.out_prefix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
