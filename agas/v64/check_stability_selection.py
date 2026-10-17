"""Does stability selection actually beat a single-probe selection at 10 folds?

The v64 smoke test at 3 folds put `stable200` below `stable_full` on AUC, which
would make the whole idea pointless. But 3 folds trains on a third of the rows
and the selection behaves very differently there, so the comparison is redone at
the real 10-fold setting before spending four hours on it.

Everything else is held fixed: same parameters, same fold seed, same feature
frame, one LightGBM seed per arm.

Run: python3 agas/v64/check_stability_selection.py
"""

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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
OUT = paths.version_output_of("v64")
FOLDS = 10
SEED = 42
TOP_N = 200
PROBES = 5


def rank01(v):
    return rankdata(v) / len(v)


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import features as ft

    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_pickle(paths.FEATURES_CACHE)
    ex = pd.read_pickle(paths.EXTRA_CACHE)
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    tr = f[f.target.notna()].reset_index(drop=True)
    y = tr.target.astype(int).values
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
    full = list(X.columns)

    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_tr = pos[tr.application_id].values
    fM, fE = ft.factor_inputs(app)
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))
    fac = [ft.factor_features(fM, fE, app.target.values, p_tr[a]) for a, _ in folds]

    def frame(k):
        fk = fac[k]
        return pd.concat([X.reset_index(drop=True), fk.iloc[p_tr].reset_index(drop=True)], axis=1)

    base = dict(objective="binary", learning_rate=0.01, num_leaves=15, min_child_samples=200,
                feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                max_bin=127, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)

    t0 = time.time()
    singles, stables = [], []
    for k, (a, b) in enumerate(folds):
        xk = frame(k)
        # single probe, as v62 and v63 did it
        pr = lgb.train({**base, "feature_fraction": 1.0}, lgb.Dataset(xk.iloc[a][full], y[a]),
                       num_boost_round=300)
        g = pd.Series(pr.feature_importance(importance_type="gain"), index=full)
        singles.append(list(g.sort_values(ascending=False).index[:TOP_N]))

        # stability selection over PROBES subsamples of the same training rows
        rng = np.random.default_rng(1000 * k + SEED)
        votes = pd.Series(0.0, index=full)
        rows = np.arange(len(a))
        for p in range(PROBES):
            sub = rng.choice(rows, int(0.8 * len(rows)), replace=False)
            q = lgb.train({**base, "feature_fraction": 0.5, "bagging_fraction": 0.6},
                          lgb.Dataset(xk.iloc[a[sub]][full], y[a[sub]]), num_boost_round=250)
            gq = pd.Series(q.feature_importance(importance_type="gain"), index=full)
            top = set(gq.sort_values(ascending=False).index[:TOP_N])
            votes += [1.0 if c in top else 0.0 for c in full]
        votes /= PROBES
        stables.append(list(votes[votes > 0].sort_values(ascending=False).index[:TOP_N]))
        print(f"  fold {k}: single probe kept {len(singles[-1])}, stable kept "
              f"{len(stables[-1])}, columns in 80%+ of probes "
              f"{int((votes >= 0.8).sum())}, in any probe {int((votes > 0).sum())}", flush=True)

    overlap = np.mean([len(set(singles[k]) & set(stables[k])) for k in range(FOLDS)])
    print(f"\nmean overlap between single and stable selection: {overlap:.1f} of {TOP_N} "
          f"({100 * overlap / TOP_N:.0f} percent)")

    arms = {"single_top200": singles, "stable_top200": stables, "full": [full] * FOLDS}
    res = {}
    for name, cols_by_fold in arms.items():
        oof = np.zeros(len(y))
        for k, (a, b) in enumerate(folds):
            xk = frame(k)
            cols = cols_by_fold[k]
            m = lgb.train(base, lgb.Dataset(xk.iloc[a][cols], y[a]), num_boost_round=12000,
                          valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[b] = m.predict(xk.iloc[b][cols])
        from scipy.stats import ks_2samp
        res[name] = {"auc": float(roc_auc_score(y, oof)),
                     "ks": float(ks_2samp(oof[y == 1], oof[y == 0]).statistic)}
        print(f"{name:16s} auc={res[name]['auc']:.6f} ks={res[name]['ks']:.6f}", flush=True)

    print(f"\nverdict: stability selection changes AUC by "
          f"{res['stable_top200']['auc'] - res['single_top200']['auc']:+.6f} against a "
          f"single-probe selection at the same size.")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
    import json
    (OUT / "summary_stability_check.json").write_text(json.dumps(
        {"results": res, "mean_overlap": float(overlap), "folds": FOLDS,
         "top_n": TOP_N, "probes": PROBES, "seed": SEED}, indent=2))


if __name__ == "__main__":
    main()
