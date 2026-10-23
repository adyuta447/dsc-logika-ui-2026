"""v70 main run: new cross-source blocks on top of the v69 lean recipe.

The v66/v69 legs stay fixed and are reused from `preds_v66.npz` and
`preds_v69.npz`; this run trains only the new legs that the blend can add or
swap in. Every variant uses the same fold construction, the fold-safe factor
columns and the v70 feature cache, so the difference between variants is the
LightGBM parameter preset, not the data.

Partials are saved after every fold seed, so a run can be stopped and resumed.
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
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
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

CONFIGS = {
    "base": dict(learning_rate=0.01, num_leaves=15, min_child_samples=200,
                 feature_fraction=0.30, bagging_fraction=0.80, lambda_l2=5.0,
                 max_bin=127),
    "l2_20": dict(learning_rate=0.01, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.30, bagging_fraction=0.80, lambda_l2=20.0,
                  max_bin=127),
    "stiff7": dict(learning_rate=0.01, num_leaves=7, min_child_samples=400,
                   feature_fraction=0.20, bagging_fraction=0.70, lambda_l2=40.0,
                   max_bin=127),
    "extra": dict(learning_rate=0.01, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.45, bagging_fraction=0.80, lambda_l2=20.0,
                  max_bin=127, extra_trees=True),
    "extra_stiff": dict(learning_rate=0.01, num_leaves=7, min_child_samples=300,
                        feature_fraction=0.30, bagging_fraction=0.75, lambda_l2=40.0,
                        max_bin=127, extra_trees=True),
    "wide31": dict(learning_rate=0.01, num_leaves=31, min_child_samples=100,
                   feature_fraction=0.35, bagging_fraction=0.85, lambda_l2=20.0,
                   max_bin=255),
}


def ks_score(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(OUT / "features_v70.pkl"))
    ap.add_argument("--variants", default="l2_20,extra,extra_stiff")
    ap.add_argument("--folds", type=int, default=15)
    ap.add_argument("--fold-seeds", default="42,7,13,21,5,101,3,17,11,23")
    ap.add_argument("--lgb-seeds", default="0")
    ap.add_argument("--drop-prefixes", default="cf_,pv_")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--no-factor", action="store_true")
    ap.add_argument("--out-prefix", default="v70")
    return ap.parse_args()


def main():
    args = parse()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        assert v in CONFIGS, f"unknown variant {v}"
    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
    use_factor = not args.no_factor
    drops = tuple(s.strip() for s in args.drop_prefixes.split(",") if s.strip())

    log_path = OUT / f"train_log_{args.out_prefix}.txt"
    if log_path.exists():
        log_path.unlink()
    t0 = time.time()

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    f = pd.read_pickle(args.features)
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X, Xte = tr.drop(columns=ids), te.drop(columns=ids)
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")

    full = list(X.columns)
    base_lean = [c for c in full
                 if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)
                 and not c.startswith(drops)]
    y = tr.target.astype(int).values
    n_tr, n_te = len(X), len(Xte)
    log(f"features={Path(args.features).name} full={len(full)} lean={len(base_lean)} "
        f"drops={drops} variants={variants} folds={args.folds} seeds={fold_seeds} "
        f"lgb_seeds={lgb_seeds} factor={use_factor}")

    import features as ft

    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([tr.application_id.values, te.application_id.values])].values
    y_app = app.target.values
    log("computing factor inputs")
    fM, fE = ft.factor_inputs(app)
    log("factor inputs ready")

    partial_path = OUT / f"preds_{args.out_prefix}_partial.npz"
    seed_oof = {}
    seed_test = {}
    done = []
    if partial_path.exists():
        p = np.load(partial_path, allow_pickle=True)
        same = (int(p["folds"]) == args.folds
                and [int(s) for s in p["fold_seeds"]] == fold_seeds
                and [int(s) for s in p["lgb_seeds"]] == lgb_seeds
                and list(p["variants"]) == variants
                and bool(p["use_factor"]) == use_factor)
        if not same:
            raise SystemExit(f"{partial_path} belongs to a different configuration")
        done = [int(s) for s in p["done"]]
        for fs in done:
            seed_oof[fs] = {v: p[f"seed_{fs}_oof_{v}"] for v in variants}
            seed_test[fs] = {v: p[f"seed_{fs}_test_{v}"] for v in variants}
        log(f"resumed with {done}")

    for fs in fold_seeds:
        if fs in done:
            log(f"skip fold seed {fs}")
            continue
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(args.folds, shuffle=True,
                                      random_state=fs).split(X, y))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        factor_cols = list(fac_k[0].columns)
        cols = base_lean + (factor_cols if use_factor else [])
        log(f"  train columns={len(cols)} factor_added={len(factor_cols) if use_factor else 0}")

        oof = {v: np.zeros(n_tr) for v in variants}
        test = {v: np.zeros(n_te) for v in variants}

        for k, (a, b) in enumerate(folds):
            fk = fac_k[k]
            xk = pd.concat([X.reset_index(drop=True),
                            fk.iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1)
            xt = pd.concat([Xte.reset_index(drop=True),
                            fk.iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1)
            for v in variants:
                for sd in lgb_seeds:
                    params = dict(objective="binary", bagging_freq=1, seed=sd,
                                  deterministic=True, force_row_wise=True, n_jobs=args.jobs,
                                  verbose=-1, **CONFIGS[v])
                    m = lgb.train(params, lgb.Dataset(xk.iloc[a][cols], y[a]),
                                  num_boost_round=12000,
                                  valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                                  callbacks=[lgb.early_stopping(300, verbose=False)])
                    oof[v][b] += m.predict(xk.iloc[b][cols]) / len(lgb_seeds)
                    test[v] += m.predict(xt[cols]) / (len(folds) * len(lgb_seeds))

        seed_oof[fs] = oof
        seed_test[fs] = test
        done.append(fs)
        for v in variants:
            log(f"  fs={fs} {v:12s} auc={roc_auc_score(y, oof[v]):.6f} "
                f"ks={ks_score(y, oof[v]):.6f}")
        np.savez_compressed(
            partial_path,
            folds=np.array(args.folds), fold_seeds=np.array(fold_seeds),
            lgb_seeds=np.array(lgb_seeds), variants=np.array(variants),
            use_factor=np.array(use_factor), done=np.array(done),
            **{f"seed_{fs2}_oof_{v}": seed_oof[fs2][v]
               for fs2 in done for v in variants},
            **{f"seed_{fs2}_test_{v}": seed_test[fs2][v]
               for fs2 in done for v in variants})
        log(f"  saved after {len(done)}/{len(fold_seeds)} seeds, "
            f"{(time.time() - t0) / 60:.1f} min")

    oof_final = {v: np.mean([seed_oof[fs][v] for fs in fold_seeds], axis=0)
                 for v in variants}
    test_final = {v: np.mean([seed_test[fs][v] for fs in fold_seeds], axis=0)
                  for v in variants}
    summary = {"features": str(args.features), "variants": variants,
               "configs": {v: CONFIGS[v] for v in variants},
               "folds": args.folds, "fold_seeds": fold_seeds,
               "lgb_seeds": lgb_seeds, "use_factor": use_factor,
               "n_full": len(full), "n_lean": len(base_lean), "drops": drops,
               "minutes": (time.time() - t0) / 60.0, "per_seed": {}}
    for fs in fold_seeds:
        summary["per_seed"][str(fs)] = {
            v: {"auc": float(roc_auc_score(y, seed_oof[fs][v])),
                "ks": ks_score(y, seed_oof[fs][v])} for v in variants}
    for v in variants:
        summary[v] = {"auc": float(roc_auc_score(y, oof_final[v])),
                      "ks": ks_score(y, oof_final[v])}
        log(f"BAG {v:12s} auc={summary[v]['auc']:.6f} ks={summary[v]['ks']:.6f}")

    np.savez_compressed(
        OUT / f"preds_{args.out_prefix}.npz",
        train_ids=tr.application_id.values, test_ids=te.application_id.values,
        y=y, **{f"oof_{v}": a for v, a in oof_final.items()},
        **{f"test_{v}": a for v, a in test_final.items()},
        **{f"seed_{fs}_oof_{v}": seed_oof[fs][v]
           for fs in fold_seeds for v in variants},
        **{f"seed_{fs}_test_{v}": seed_test[fs][v]
           for fs in fold_seeds for v in variants})
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
