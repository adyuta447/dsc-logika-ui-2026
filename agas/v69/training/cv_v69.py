"""Small, reusable CV harness for v69 experiments.

It keeps the v66/v68 recipe fixed and varies one thing at a time:

* `--drop-prefixes` removes feature blocks, so a new block can be compared
  against the same model without it.
* `--folds` changes the number of stratified folds.
* `--config` selects a LightGBM parameter preset.

The point is screening, not a final submission. The final lean run uses the
same columns and fold construction but more folds and more fold seeds.
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
    "stiff3": dict(learning_rate=0.012, num_leaves=3, min_child_samples=600,
                   feature_fraction=0.15, bagging_fraction=0.70, lambda_l2=80.0,
                   max_bin=127),
    "wide31": dict(learning_rate=0.01, num_leaves=31, min_child_samples=100,
                   feature_fraction=0.35, bagging_fraction=0.85, lambda_l2=20.0,
                   max_bin=255),
    "extra": dict(learning_rate=0.01, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.45, bagging_fraction=0.80, lambda_l2=20.0,
                  max_bin=127, extra_trees=True),
}


def ks_score(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(OUT / "features_v69.pkl"))
    ap.add_argument("--drop-prefixes", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", default="42,7,13")
    ap.add_argument("--lgb-seeds", default="0")
    ap.add_argument("--config", default="base")
    ap.add_argument("--use-factor", action="store_true",
                    help="append fold-safe factor_z/sd/fs_* columns to the lean set")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out-prefix", default="cv_v69")
    return ap.parse_args()


def main():
    args = parse()
    cfg = CONFIGS[args.config]
    folds_n = args.folds
    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
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
    lean = [c for c in full
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)
            and not c.startswith(drops)]
    y = tr.target.astype(int).values
    log(f"features={Path(args.features).name} rows={len(X)} full={len(full)} "
        f"lean={len(lean)} drops={drops} folds={folds_n} config={args.config}")

    import features as ft

    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([tr.application_id.values, te.application_id.values])].values
    y_app = app.target.values
    n_tr = len(tr)
    fM, fE = ft.factor_inputs(app)
    factor_cols = []

    oof_by_fs, test_by_fs = {}, {}
    results = {}
    for fs in fold_seeds:
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(folds_n, shuffle=True, random_state=fs).split(X, y))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]
        factor_cols = list(fac_k[0].columns)
        cols = lean + (factor_cols if args.use_factor else [])
        log(f"  train columns={len(cols)} factor={len(factor_cols) if args.use_factor else 0}")

        def frame(k):
            return (pd.concat([X.reset_index(drop=True),
                               fac_k[k].iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1),
                    pd.concat([Xte.reset_index(drop=True),
                               fac_k[k].iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1))

        oof_sum = np.zeros(n_tr)
        test_sum = np.zeros(len(Xte))
        seed_aucs = []
        for sd in lgb_seeds:
            oof = np.zeros(n_tr)
            pred = np.zeros(len(Xte))
            params = dict(objective="binary", bagging_freq=1, seed=sd,
                          deterministic=True, force_row_wise=True, n_jobs=args.jobs,
                          verbose=-1, **cfg)
            for k, (a, b) in enumerate(folds):
                xk, xt = frame(k)
                m = lgb.train(params, lgb.Dataset(xk.iloc[a][cols], y[a]),
                              num_boost_round=12000,
                              valid_sets=[lgb.Dataset(xk.iloc[b][cols], y[b])],
                              callbacks=[lgb.early_stopping(300, verbose=False)])
                oof[b] = m.predict(xk.iloc[b][cols])
                pred += m.predict(xt[cols]) / len(folds)
            auc = roc_auc_score(y, oof)
            seed_aucs.append(float(auc))
            log(f"  lgb{sd} auc={auc:.6f} ks={ks_score(y, oof):.6f}")
            oof_sum += oof / len(lgb_seeds)
            test_sum += pred / len(lgb_seeds)

        auc = roc_auc_score(y, oof_sum)
        ks = ks_score(y, oof_sum)
        results[str(fs)] = {"auc": float(auc), "ks": ks,
                            "auc_lgb_mean": float(np.mean(seed_aucs)),
                            "auc_lgb_sd": float(np.std(seed_aucs))}
        oof_by_fs[f"oof_{fs}"] = oof_sum
        test_by_fs[f"test_{fs}"] = test_sum
        log(f"fs={fs} auc={auc:.6f} ks={ks:.6f}")

    vals = [results[str(fs)]["auc"] for fs in fold_seeds]
    ksv = [results[str(fs)]["ks"] for fs in fold_seeds]
    summary = {
        "features": str(args.features), "drops": drops, "folds": folds_n,
        "fold_seeds": fold_seeds, "lgb_seeds": lgb_seeds, "config": args.config,
        "n_full": len(full), "n_lean": len(lean), "use_factor": bool(args.use_factor),
        "n_factor": len(factor_cols),
        "mean_auc": float(np.mean(vals)), "sd_auc": float(np.std(vals)),
        "mean_ks": float(np.mean(ksv)), "minutes": (time.time() - t0) / 60.0,
        "per_fold_seed": results,
    }
    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        train_ids=tr.application_id.values,
                        test_ids=te.application_id.values, y=y,
                        **oof_by_fs, **test_by_fs)
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"SUMMARY auc={summary['mean_auc']:.6f} sd={summary['sd_auc']:.6f} "
        f"ks={summary['mean_ks']:.6f} in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
