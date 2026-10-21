"""v69 lean sweep: regularization and capacity variants of the winning leg.

The public winner `ext0.5|tm30|lgb_lean` is built by pushing `lgb_lean` away
from `tm30`. This sweep keeps the feature set and the blend target fixed and
varies only the LightGBM leg, using three fold seeds so a variant has to be
good on average rather than on one partition.

The sweep is deliberately small: five configurations, two LightGBM seeds, five
folds. It is a screening step, not a final model. The selected variants are
retrained with ten folds and more fold seeds by `train_lean_v69.py`.
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
    "wide31": dict(learning_rate=0.01, num_leaves=31, min_child_samples=100,
                   feature_fraction=0.35, bagging_fraction=0.85, lambda_l2=20.0,
                   max_bin=255),
    "extra": dict(learning_rate=0.01, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.45, bagging_fraction=0.80, lambda_l2=20.0,
                  max_bin=127, extra_trees=True),
}


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


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
    return X, Xte, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", type=str, default="42,7,13")
    ap.add_argument("--lgb-seeds", type=str, default="0,1")
    ap.add_argument("--out-prefix", type=str, default="v69sweep")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_path = OUT / f"train_log_{args.out_prefix}.txt"
    if log_path.exists():
        log_path.unlink()

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    X, Xte, tr, te = load_frames()
    y = tr.target.astype(int).values
    full = list(X.columns)
    lean = [c for c in full if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    log(f"rows={len(X)} lean={len(lean)} full={len(full)}")

    import features as ft
    app = ft.load_apps()
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    p_all = pos[np.concatenate([tr.application_id.values, te.application_id.values])].values
    y_app = app.target.values
    n_tr = len(tr)
    fM, fE = ft.factor_inputs(app)
    log("factor inputs ready")

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    lgb_seeds = [int(s) for s in args.lgb_seeds.split(",") if s.strip()]
    oof_store = {}
    test_store = {}
    results = {name: {"per_seed": {}, "fold_seed": {}} for name in CONFIGS}

    for fs in fold_seeds:
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=fs).split(X, y))
        fac_k = [ft.factor_features(fM, fE, y_app, p_all[a]) for a, _ in folds]

        def frame(k):
            return (pd.concat([X.reset_index(drop=True),
                               fac_k[k].iloc[p_all[:n_tr]].reset_index(drop=True)], axis=1),
                    pd.concat([Xte.reset_index(drop=True),
                               fac_k[k].iloc[p_all[n_tr:]].reset_index(drop=True)], axis=1))

        for name, cfg in CONFIGS.items():
            params = dict(
                objective="binary",
                bagging_freq=1,
                seed=0,
                deterministic=True,
                force_row_wise=True,
                n_jobs=-1,
                verbose=-1,
                **cfg,
            )
            oof_sum = np.zeros(n_tr)
            test_sum = np.zeros(len(Xte))
            seed_aucs = []
            for sd in lgb_seeds:
                oof = np.zeros(n_tr)
                pred = np.zeros(len(Xte))
                p = dict(params, seed=sd)
                for k, (a, b) in enumerate(folds):
                    xk, xt = frame(k)
                    m = lgb.train(
                        p, lgb.Dataset(xk.iloc[a][lean], y[a]),
                        num_boost_round=12000,
                        valid_sets=[lgb.Dataset(xk.iloc[b][lean], y[b])],
                        callbacks=[lgb.early_stopping(300, verbose=False)],
                    )
                    oof[b] = m.predict(xk.iloc[b][lean])
                    pred += m.predict(xt[lean]) / len(folds)
                oof_sum += oof / len(lgb_seeds)
                test_sum += pred / len(lgb_seeds)
                auc = roc_auc_score(y, oof)
                seed_aucs.append(float(auc))
                log(f"  {name:6s} lgb{sd} auc={auc:.6f} ks={ks_score(y, oof):.6f}")
            auc = roc_auc_score(y, oof_sum)
            ks = ks_score(y, oof_sum)
            results[name]["per_seed"][str(fs)] = seed_aucs
            results[name]["fold_seed"][str(fs)] = {
                "auc": float(auc), "ks": float(ks),
                "auc_mean_lgb": float(np.mean(seed_aucs)),
                "auc_sd_lgb": float(np.std(seed_aucs)),
            }
            oof_store[f"oof_{name}_{fs}"] = oof_sum
            test_store[f"test_{name}_{fs}"] = test_sum
            log(f"{name:6s} fs={fs} auc={auc:.6f} ks={ks:.6f} "
                f"(lgb seed mean={np.mean(seed_aucs):.6f} sd={np.std(seed_aucs):.6f})")

    for name in CONFIGS:
        vals = [results[name]["fold_seed"][str(fs)]["auc"] for fs in fold_seeds]
        results[name]["mean_auc"] = float(np.mean(vals))
        results[name]["sd_auc"] = float(np.std(vals))
        results[name]["mean_ks"] = float(np.mean(
            [results[name]["fold_seed"][str(fs)]["ks"] for fs in fold_seeds]))
        log(f"SUMMARY {name:6s} auc={results[name]['mean_auc']:.6f} "
            f"sd={results[name]['sd_auc']:.6f} ks={results[name]['mean_ks']:.6f}")

    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        train_ids=tr.application_id.values,
                        test_ids=te.application_id.values,
                        y=y, **oof_store, **test_store)
    summary = {"folds": args.folds, "fold_seeds": fold_seeds,
               "lgb_seeds": lgb_seeds, "n_lean": len(lean),
               "minutes": (time.time() - t0) / 60.0,
               "configs": CONFIGS, "results": results}
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
