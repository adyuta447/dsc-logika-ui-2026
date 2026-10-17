"""v64 model: maximise ranking quality, not the OOF KS maximum.

## Why the objective changed

The v63 run reported KS OOF 0.529867 against 0.528717 for v62 and 0.528624 for
v61, and the public score came back 0.557. That prompted an audit of every file
for which both numbers exist:

    v57 aggressive_145     OOF 0.526988  public 0.55700
    v58ext ext1.5          OOF 0.527511  public 0.55900
    v61 alt_ext0.5_tm30_lean  OOF 0.528624  public 0.56312
    v63 best candidate     OOF 0.529867  public 0.55700

Pearson correlation across those four points is +0.019. More usefully, KS is a
maximum over thresholds of a cumulative sum, so it has large sampling noise even
at 70000 rows. Measured by resampling rows of a single fixed prediction vector:

    OOF size, 70000 rows          sampling sd 0.0036
    public size, 30000 rows       sampling sd 0.0053

So the entire v61 to v63 improvement of +0.001243 is 0.34 sampling sd, and the
gap that must be closed to reach 0.57 is 1.30 public sd. Choosing candidates by
the OOF KS maximum is selecting the largest of many noisy measurements: over the
1300 grid points searched in v63 that argmax is inflated by about 0.012 for
free, which is ten times the improvement being chased.

## What this run therefore optimises

Ranking quality measured by AUC, which is a smooth average of ranks over all
pairs rather than a single worst-case threshold, so it has far smaller sampling
noise and responds to genuine model improvement. KS stays in the report as a
secondary number with its sampling sd attached, never as the selection
criterion.

Three changes follow from that:

1. **More fold seeds, on the recipe v63 already proved.** Fold-seed bagging is
   the only lever whose effect on AUC has been monotone in every experiment:
   0.845897 on one seed, 0.846647 on three, 0.847419 with selection on top. v64
   takes it to six seeds. Everything else about the recipe is unchanged from the
   arm that measured best.
2. **Single-probe selection, not stability selection.** Stability selection was
   tried and rejected before this run: selecting columns that survive five
   independent gain probes instead of one scored 0.845148 AUC against 0.845506
   for the single probe at the same size, and the two sets overlap by 85 percent,
   so the probes mostly agree and the extra ones add cost without adding
   information. See `check_stability_selection.py` for the measurement.

The extrapolation recipes are kept, because both public-proven directions
(`v4 - fac` and `topmix30 - lgb_lean`) are documented in `progress.md` as
reproducible, but they are applied to a better base and evaluated on AUC with
KS reported alongside.
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
from scipy.stats import rankdata
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


def ks_of(y, p):
    from scipy.stats import ks_2samp
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13,21,5,101")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--out-prefix", type=str, default="v64")
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
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    faclean = [c for c in fac if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"ladder: full={len(full)} fac={len(fac)} lean={len(lean)} faclean={len(faclean)}")

    log("factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor ready")

    keys = ["sel200", "sel400", "lgb_full", "lgb_faclean", "lgbx_full",
            "cat_full", "xgb_full", "linear"]
    bag_o = {k: np.zeros(n_tr) for k in keys}
    bag_t = {k: np.zeros(len(Xte)) for k in keys}
    per_seed = {k: [] for k in keys}
    sel_report = []

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]

    def params(seed, ff=0.3, leaves=15, mcs=200, extra=False):
        if extra:
            return dict(objective="binary", learning_rate=args.lr * 1.2, num_leaves=31,
                        min_child_samples=120, feature_fraction=0.35, bagging_fraction=0.85,
                        bagging_freq=1, lambda_l2=args.l2 * 2.4, max_bin=255, seed=seed,
                        deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        return dict(objective="binary", learning_rate=args.lr, num_leaves=leaves,
                    min_child_samples=mcs, feature_fraction=ff, bagging_fraction=0.8,
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

        def probe_select(k, a, cols, n_keep):
            """Top columns by gain from one probe on this fold's training rows.

            The probe sees training rows only, so nothing crosses the split.
            Stability selection over several probes was measured first and was
            not better, see the module docstring.
            """
            xk, _ = get_frame(k)
            pr = lgb.train({**params(0, ff=1.0)},
                           lgb.Dataset(xk.iloc[a][cols], y[a]), num_boost_round=300)
            g = pd.Series(pr.feature_importance(importance_type="gain"), index=cols)
            return list(g.sort_values(ascending=False).index[:n_keep])

        def run_lgb(cols_by_fold, seed, ff=0.3, extra=False):
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xk, xt = get_frame(k)
                cols = cols_by_fold[k]
                m = lgb.train(params(seed, ff, extra=extra),
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
                m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10 ** 6,
                                                     random_state=fs),
                                  SplineTransformer(n_knots=4, degree=2),
                                  LogisticRegression(C=0.03, max_iter=3000))
                m.fit(mat(xk.iloc[a], med), y[a])
                oof[b] = m.predict_proba(mat(xk.iloc[b], med))[:, 1]
                pred += m.predict_proba(mat(xt, med))[:, 1] / len(folds)
            return oof, pred

        sel200 = [probe_select(k, a, full, args.top_n) for k, (a, _) in enumerate(folds)]
        sel400 = [probe_select(k, a, full, 400) for k, (a, _) in enumerate(folds)]
        overlap = float(np.mean([len(set(sel200[k]) & set(sel400[k])) for k in range(len(folds))]))
        sel_report.append({"fold_seed": fs, "n_full": len(full),
                           "mean_overlap_200_400": overlap})
        log(f"seed {fs} selection: 200 and 400 column sets overlap by {overlap:.0f} columns")

        jobs = [
            ("sel200", lambda: run_lgb(sel200, 0, ff=0.3)),
            ("sel400", lambda: run_lgb(sel400, 1, ff=0.4)),
            ("lgb_full", lambda: run_lgb([full] * len(folds), 2, ff=0.3)),
            ("lgb_faclean", lambda: run_lgb([faclean] * len(folds), 0, ff=0.3)),
            ("lgbx_full", lambda: run_lgb([full] * len(folds), 99, extra=True)),
            ("cat_full", lambda: run_cat([full] * len(folds))),
            ("xgb_full", lambda: run_xgb([full] * len(folds))),
            ("linear", lambda: run_linear(lean)),
        ]
        for key, fn in jobs:
            o, p = fn()
            bag_o[key] += o / len(fold_seeds)
            bag_t[key] += p / len(fold_seeds)
            per_seed[key].append(o)
            log(f"seed {fs} {key:12s} auc={roc_auc_score(y, o):.6f} ks={ks_of(y, o):.6f}")

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
    summary = {"params": vars(args), "minutes": (time.time() - t0) / 60.0,
               "objective": "AUC, not KS maximum", "models": {},
               "ladder": {"full": len(full), "fac": len(fac), "lean": len(lean),
                          "faclean": len(faclean)},
               "selection": sel_report}
    for k in keys:
        s = {"auc": float(roc_auc_score(y, bag_o[k])), "ks": float(ks_of(y, bag_o[k]))}
        s["per_seed"] = [{"auc": float(roc_auc_score(y, v)), "ks": float(ks_of(y, v))}
                         for v in per_seed[k]]
        summary["models"][k] = s
        log(f"BAG {k:12s} auc={s['auc']:.6f} ks={s['ks']:.6f} "
            f"per-seed auc={[round(p['auc'], 5) for p in s['per_seed']]}")
    with open(OUT / f"summary_{args.out_prefix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
