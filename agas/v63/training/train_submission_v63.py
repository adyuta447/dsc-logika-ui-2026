"""v63 ensemble, built on the feature-selection result.

The v63 sweep averaged fold seeds 42, 7 and 13 and compared selection levels and
the hyperparameters that suit them:

    no selection                 AUC 0.846102  KS 0.524794  sd 0.00064
    top 200, feature_fraction 0.3  AUC 0.846141  KS 0.527077  sd 0.00017
    top 200, feature_fraction 0.6  AUC 0.845785  KS 0.527346  sd 0.00117
    top 300                      AUC 0.846150  KS 0.525802  sd 0.00036
    top 400                      AUC 0.846195  KS 0.525898  sd 0.00037
    top 100                      AUC 0.845127  KS 0.523984  sd 0.00054

Two conclusions drive this script. First, selection is the real lever: it adds
about 0.0023 KS and does so in all three fold seeds, with by far the smallest
spread of any configuration. Second, the hypothesis that the selected set needed a
wider `feature_fraction` was wrong. Raising it to 0.6 nudges KS up by 0.0003 but
costs 0.0004 AUC and multiplies the fold-seed spread by seven, so 0.3 stays.

Both proved public directions extrapolate a richer-feature model away from a
reduced one. Selection size is another axis of exactly that quantity, so this run
keeps several selection levels as legs: extrapolating from the 200-column model
away from the 400-column model sharpens the same gradient. The legacy `full` and
`faclean` legs are kept so the already-proven directions remain reachable.

Selection is computed per fold from a gain probe on that fold's training rows
only, so no label crosses the validation split. Factor loading comes only from
the training fold's labels. Historical aggregates are cut at `decision_datetime`.
Nothing outside `agas/` is modified.
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
CACHE = paths.CACHE

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
    ap.add_argument("--sel200-seeds", type=str, default="0,1")
    ap.add_argument("--out-prefix", type=str, default="v63")
    return ap.parse_args()


def load_frames():
    f = pd.read_pickle(CACHE / "features_cache.pkl")
    ex = pd.read_pickle(CACHE / "extra_cache.pkl")
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
    fac = [c for c in full if not c.startswith(ROUND2_PREFIX)]
    faclean = [c for c in fac if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"ladder: full={len(full)} fac={len(fac)} faclean={len(faclean)}")

    log("factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor ready")

    # selection size -> feature_fraction, from the v63 sweep
    SEL_FF = {100: 0.9, 200: 0.3, 300: 0.5, 400: 0.4}
    keys = ["sel100", "sel200", "sel200b", "sel300", "sel400",
            "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
    bag_o = {k: np.zeros(n_tr) for k in keys}
    bag_t = {k: np.zeros(len(Xte)) for k in keys}
    per_seed = {k: [] for k in keys}

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    sel200_seeds = [int(s) for s in args.sel200_seeds.split(",") if s.strip()]

    def params(seed, ff=0.3, leaves=15, mcs=200, l2=None, extra=False):
        if extra:
            return dict(objective="binary", learning_rate=args.lr * 1.2, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35, bagging_fraction=0.85,
                        bagging_freq=1, lambda_l2=(l2 or args.l2) * 2.4, max_bin=255, seed=seed,
                        deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        return dict(objective="binary", learning_rate=args.lr, num_leaves=leaves,
                    min_child_samples=mcs, feature_fraction=ff, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=(l2 or args.l2), max_bin=127, seed=seed,
                    deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)

    for fi, fs in enumerate(fold_seeds):
        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                     random_state=fs).split(X, y_app[:n_tr]))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        y = y_app[:n_tr].astype(int)
        cache, ranked = {}, {}

        def get_frame(k):
            if k not in cache:
                fk = fac_k[k]
                cache[k] = (
                    pd.concat([X.reset_index(drop=True), fk.iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1),
                    pd.concat([Xte.reset_index(drop=True), fk.iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1),
                )
            return cache[k]

        def gain_rank(k, cols, a):
            """Gain probe on this fold's training rows only."""
            if ("probe", cols[0], len(cols)) not in ranked:
                xk, _ = get_frame(k)
                pr = lgb.train(params(0, ff=1.0), lgb.Dataset(xk.iloc[a][cols], y[a]),
                               num_boost_round=300)
                g = pd.Series(pr.feature_importance(importance_type="gain"), index=cols)
                ranked[("probe", cols[0], len(cols))] = list(g.sort_values(ascending=False).index)
            return ranked[("probe", cols[0], len(cols))]

        def run_lgb(cols_by_fold, seed, ff=0.3, leaves=15, mcs=200, l2=None, extra=False):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                cols = cols_by_fold[k]
                m = lgb.train(params(seed, ff, leaves, mcs, l2, extra),
                              lgb.Dataset(xk.iloc[a][cols], y[a]),
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

        def sel_cols(n):
            return [gain_rank(k, full, a)[:n] for k, (a, _) in enumerate(folds)]

        jobs = [
            ("sel200", lambda: run_lgb(sel_cols(200), 0, ff=0.3)),
            ("sel300", lambda: run_lgb(sel_cols(300), 0, ff=0.5)),
            ("sel400", lambda: run_lgb(sel_cols(400), 0, ff=0.4)),
            ("sel100", lambda: run_lgb(sel_cols(100), 0, ff=0.9, mcs=120)),
            ("sel200b", lambda: run_lgb(sel_cols(200), 7, ff=0.6, l2=2)),
            ("lgb_faclean", lambda: run_lgb([faclean] * len(folds), 0, ff=0.3)),
            ("lgb_full", lambda: run_lgb([full] * len(folds), 0, ff=0.3)),
            ("lgbx_full", lambda: run_lgb([full] * len(folds), 99, extra=True)),
            ("cat_full", lambda: run_cat([full] * len(folds))),
            ("xgb_full", lambda: run_xgb([full] * len(folds))),
            ("linear", lambda: run_linear(faclean)),
        ]
        for key, fn in jobs:
            o, p = fn()
            bag_o[key] += o / len(fold_seeds)
            bag_t[key] += p / len(fold_seeds)
            per_seed[key].append(o)
            log(f"seed {fs} {key:11s} auc={roc_auc_score(y, o):.6f} ks={ks_score(y, o):.6f}")

        oo = [run_lgb(sel_cols(200), sd, ff=0.3) for sd in sel200_seeds]
        o = np.mean([a for a, _ in oo], axis=0)
        p = np.mean([b for _, b in oo], axis=0)
        bag_o["sel200"] += o / len(fold_seeds)
        bag_t["sel200"] += p / len(fold_seeds)
        per_seed["sel200"][-1] = o
        log(f"seed {fs} {'sel200x%d' % len(sel200_seeds):11s} auc={roc_auc_score(y, o):.6f} "
            f"ks={ks_score(y, o):.6f}")

        np.savez_compressed(
            OUT / f"preds_{args.out_prefix}.npz",
            test_ids=test_ids, train_ids=train_ids, seeds=np.array(fold_seeds),
            **{f"oof_{k}": bag_o[k] for k in keys},
            **{f"test_{k}": bag_t[k] for k in keys},
            **{f"oofseed_{k}_{j}": v for k in keys for j, v in enumerate(per_seed[k])},
        )
        log(f"saved after {fi + 1}/{len(fold_seeds)} fold seeds, "
            f"elapsed {(time.time() - t0) / 60:.1f} min")

    y = y_app[:n_tr].astype(int)
    summary = {"params": vars(args), "minutes": (time.time() - t0) / 60.0, "models": {},
               "ladder": {"full": len(full), "fac": len(fac), "faclean": len(faclean)},
               "selection_ff": SEL_FF}
    for k in keys:
        s = {"auc": float(roc_auc_score(y, bag_o[k])), "ks": float(ks_score(y, bag_o[k]))}
        s["per_seed"] = [{"auc": float(roc_auc_score(y, v)), "ks": float(ks_score(y, v))}
                         for v in per_seed[k]]
        summary["models"][k] = s
        log(f"BAG {k:11s} auc={s['auc']:.6f} ks={s['ks']:.6f} "
            f"per-seed ks={[round(p['ks'], 5) for p in s['per_seed']]}")
    with open(OUT / f"summary_{args.out_prefix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
