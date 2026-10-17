"""Fast CV harness for feature / blend iteration.

Loads the cached feature frame, attaches optional extra feature blocks from
`agas/feat_extra.py`, rebuilds the fold-safe factor features, and trains a
single LightGBM 5-fold so a candidate block can be scored in a few minutes.
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
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
OUT = HERE / "harness_output"
CACHE = paths.FEATURES_CACHE
KEEP_V2 = ("price", "late", "ldpd", "bu_z")
EXTRA_CACHE = paths.EXTRA_CACHE


def ks_score(y_true, pred):
    return ks_2samp(pred[y_true == 1], pred[y_true == 0]).statistic


def rank01(values):
    return rankdata(values) / len(values)


def load_frames(use_extra=True):
    if CACHE.exists():
        f = pd.read_pickle(CACHE)
    else:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import features as ft

        f = ft.build()
        f.to_pickle(CACHE)

    if use_extra and EXTRA_CACHE.exists():
        ex = pd.read_pickle(EXTRA_CACHE)
        base_ids = list(f.application_id)
        missing = [c for c in ex.columns if c not in f.columns]
        f = f.merge(ex[missing + ["application_id"]], on="application_id", how="left")
        assert list(f.application_id) == base_ids

    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    return f, tr, te


def factor_block(tr, te, folds, app, version=1):
    import features as ft

    if version == 2:
        import factor2

        M, E = factor2.factor_inputs_v2(app)
        pos = pd.Series(np.arange(len(app)), index=app.application_id)
        p_tr = pos[tr.application_id].values
        p_te = pos[te.application_id].values
        fac = [factor2.factor_features_v2(M, E, app.target.values, p_tr[a], KEEP_V2)
               for a, _ in folds]
        return fac, p_tr, p_te, list(fac[0].columns)

    M, E = ft.factor_inputs(app)
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_tr = pos[tr.application_id].values
    p_te = pos[te.application_id].values
    fac = [ft.factor_features(M, E, app.target.values, p_tr[a]) for a, _ in folds]
    fac_cols = list(fac[0].columns)
    return fac, p_tr, p_te, fac_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tag", type=str, default="harness")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--no-factor", action="store_true")
    ap.add_argument("--drop", type=str, default="", help="comma separated prefixes to drop")
    ap.add_argument("--oof-out", type=str, default="")
    ap.add_argument("--ff", type=float, default=0.3)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--mcs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=20.0)
    ap.add_argument("--bag", type=float, default=0.8)
    ap.add_argument("--maxbin", type=int, default=127)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=0,
                    help="per-fold gain selection; 0 disables selection")
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import features as ft

    t0 = time.time()
    f, tr, te = load_frames(True)
    y = tr.target.astype(int).values
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    Xte = te.drop(columns=["application_id", "customer_id", "target"])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")

    app = ft.load_apps()
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.fold_seed)
    folds = list(skf.split(X, y))

    use_fac = not args.no_factor
    fac = [None] * len(folds)
    fac_cols = []
    if use_fac:
        fac, p_tr, p_te, fac_cols = factor_block(tr, te, folds, app, args.factor)
        X = pd.concat([X, fac[0].iloc[p_tr].reset_index(drop=True)], axis=1)
        Xte = pd.concat([Xte, fac[0].iloc[p_te].reset_index(drop=True)], axis=1)

    if args.drop:
        pref = tuple(args.drop.split(","))
        cols = [c for c in X.columns if not c.startswith(pref)]
    else:
        cols = list(X.columns)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gains = [None] * len(folds)
    oof_sum = np.zeros(len(y))
    pred_sum = np.zeros(len(Xte))
    for seed in seeds:
        params = dict(
            objective="binary",
            learning_rate=args.lr,
            num_leaves=args.leaves,
            min_child_samples=args.mcs,
            feature_fraction=args.ff,
            bagging_fraction=args.bag,
            bagging_freq=1,
            lambda_l2=args.l2,
            max_bin=args.maxbin,
            seed=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=-1,
            verbose=-1,
        )
        for k, (a, b) in enumerate(folds):
            xk = X if not use_fac else pd.concat(
                [X.drop(columns=fac_cols), fac[k].iloc[p_tr].reset_index(drop=True)], axis=1)
            use = cols
            if args.top_n:
                if gains[k] is None:
                    probe = lgb.train(
                        {**params, "feature_fraction": 1.0, "num_boost_round": 300,
                         "learning_rate": 0.1},
                        lgb.Dataset(xk.iloc[a][cols], y[a]),
                        num_boost_round=300)
                    g = pd.Series(probe.feature_importance(importance_type="gain"), index=cols)
                    gains[k] = list(g.sort_values(ascending=False).index[: args.top_n])
                use = gains[k]
            dtr = lgb.Dataset(xk.iloc[a][use], y[a])
            dva = lgb.Dataset(xk.iloc[b][use], y[b])
            m = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof_sum[b] += m.predict(xk.iloc[b][use]) / len(seeds)
            xk_t = Xte if not use_fac else pd.concat(
                [Xte.drop(columns=fac_cols), fac[k].iloc[p_te].reset_index(drop=True)], axis=1)
            pred_sum += m.predict(xk_t[use]) / (len(seeds) * len(folds))

    res = {
        "tag": args.tag,
        "n_features": len(cols),
        "top_n": args.top_n,
        "auc": float(roc_auc_score(y, oof_sum)),
        "ks": float(ks_score(y, oof_sum)),
        "params": vars(args),
        "minutes": (time.time() - t0) / 60.0,
    }
    print(json.dumps(res, indent=2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"harness_{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    if args.oof_out:
        np.save(OUT / args.oof_out, oof_sum)
        np.save(OUT / args.oof_out.replace(".npy", "_test.npy"), rank01(pred_sum))


if __name__ == "__main__":
    main()