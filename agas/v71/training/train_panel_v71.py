"""Train the v71 historical panel model and cross-fit its score.

The panel is trained only on historical loans whose 12-month outcome window is
fully observed.  Folds are grouped by customer so a customer's prior loans and
current application never sit on opposite sides of a validation boundary.

The current train/test rows are scored with the same cross-fitting rule: a
customer's current row is scored by the fold model that did not see that
customer's historical labels.  This makes the panel score a leak-free feature
for the v71 current model.
"""

import argparse
import gc
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
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"

DROP = {
    "row_id", "customer_id", "cut_id", "cut", "cur_cut", "is_hist", "label",
    "own_application_id", "loan_id", "loan_purpose", "hist_first_due",
    "hist_window_end", "hist_maxdpd12",
}
CATS = ["region", "employment_type", "housing_status"]

CONFIGS = {
    "extra": dict(learning_rate=0.02, num_leaves=15, min_child_samples=150,
                  feature_fraction=0.40, bagging_fraction=0.80,
                  lambda_l2=20.0, max_bin=127, extra_trees=True),
    "stiff": dict(learning_rate=0.02, num_leaves=7, min_child_samples=300,
                  feature_fraction=0.25, bagging_fraction=0.75,
                  lambda_l2=40.0, max_bin=127),
}


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "panel_dataset.pkl"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default="extra,stiff")
    ap.add_argument("--out-prefix", default="panel_v71")
    args = ap.parse_args()
    t0 = time.time()
    f = pd.read_pickle(args.data)
    hist = f[f.is_hist == 1].reset_index(drop=True)
    cur = f[f.is_hist == 0].reset_index(drop=True)
    y = hist.label.astype(int).values
    y_cur = cur.label.astype(float).values
    cur_has_label = np.isfinite(y_cur)
    feat = [c for c in f.columns if c.startswith("pn_")]
    feat += [c for c in CATS if c in f.columns]
    for c in CATS:
        if c in f.columns:
            f[c] = f[c].astype("category")
    # keep only columns with some variation; drop constant and all-missing
    keep = []
    for c in feat:
        s = f[c]
        if s.notna().sum() == 0:
            continue
        if s.dtype.name == "category":
            if s.nunique(dropna=True) > 1:
                keep.append(c)
        elif s.std(skipna=True) > 1e-12:
            keep.append(c)
    feat = keep
    log(f"panel hist={len(hist)} current={len(cur)} features={len(feat)} "
        f"positive={y.mean():.4f}")

    # customer folds: one assignment covers historical and current rows
    cust = hist[["customer_id"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=args.folds)
    cust_fold = np.zeros(len(cust), dtype=int)
    for k, (_, b) in enumerate(gkf.split(cust, groups=cust.customer_id)):
        cust_fold[b] = k
    fold_map = pd.Series(cust_fold, index=cust.customer_id)
    hist["cust_fold"] = hist.customer_id.map(fold_map).fillna(-1).astype(int)
    cur["cust_fold"] = cur.customer_id.map(fold_map).fillna(-1).astype(int)
    # customers with no historical loan (mostly current-only) get a deterministic
    # fold so they still receive a cross-fitted score
    missing = cur.cust_fold < 0
    if missing.any():
        cur.loc[missing, "cust_fold"] = cur.loc[missing, "customer_id"].map(
            lambda s: hash(s) % args.folds)
    log(f"hist fold counts={hist.cust_fold.value_counts().sort_index().to_dict()}")
    log(f"current fold counts={cur.cust_fold.value_counts().sort_index().to_dict()}")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # OOF accumulators for historical rows
    hist_oof = np.zeros(len(hist))
    hist_oof_cnt = np.zeros(len(hist))
    # current score: one cross-fitted prediction per customer fold, averaged over configs/seeds
    cur_score = np.zeros(len(cur))
    cur_score_cnt = np.zeros(len(cur))
    fold_models = {}

    per_fold = {}
    for k in range(args.folds):
        tr_idx = np.where(hist.cust_fold != k)[0]
        va_idx = np.where(hist.cust_fold == k)[0]
        x_tr = hist.iloc[tr_idx][feat]
        x_va = hist.iloc[va_idx][feat]
        log(f"fold {k}: train={len(tr_idx)} valid={len(va_idx)} "
            f"valid_pos={y[va_idx].mean():.4f}")
        models = []
        pred_va = np.zeros(len(va_idx))
        for cfg_name in configs:
            for sd in seeds:
                params = dict(objective="binary", bagging_freq=1, seed=sd,
                              deterministic=True, force_row_wise=True,
                              n_jobs=-1, verbose=-1, **CONFIGS[cfg_name])
                m = lgb.train(params, lgb.Dataset(x_tr, y[tr_idx]),
                              num_boost_round=6000,
                              valid_sets=[lgb.Dataset(x_va, y[va_idx])],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                pred_va += m.predict(x_va) / (len(configs) * len(seeds))
                models.append(m)
        auc = roc_auc_score(y[va_idx], pred_va)
        ksv = ks(y[va_idx], pred_va)
        per_fold[str(k)] = {"auc": float(auc), "ks": ksv,
                            "n": int(len(va_idx)), "pos": float(y[va_idx].mean())}
        log(f"  fold {k} auc={auc:.6f} ks={ksv:.6f}")
        hist_oof[va_idx] += pred_va
        hist_oof_cnt[va_idx] += 1
        # current rows in this customer fold are scored by exactly this model set
        ci = np.where(cur.cust_fold.values == k)[0]
        if len(ci):
            pred_c = np.zeros(len(ci))
            for m in models:
                pred_c += m.predict(cur.iloc[ci][feat]) / len(models)
            cur_score[ci] = pred_c
            cur_score_cnt[ci] = 1
        fold_models[k] = models
        gc.collect()

    hist_oof = hist_oof / np.maximum(hist_oof_cnt, 1)
    assert (hist_oof_cnt == 1).all(), "historical row received more than one OOF prediction"
    assert (cur_score_cnt == 1).all(), "current row did not receive exactly one cross-fit prediction"

    panel_auc = roc_auc_score(y, hist_oof)
    panel_ks = ks(y, hist_oof)
    cur_auc = roc_auc_score(y_cur[cur_has_label], cur_score[cur_has_label])
    cur_ks = ks(y_cur[cur_has_label].astype(int), cur_score[cur_has_label])
    log(f"PANEL hist OOF auc={panel_auc:.6f} ks={panel_ks:.6f}")
    log(f"PANEL current train auc={cur_auc:.6f} ks={cur_ks:.6f}")

    np.savez_compressed(
        OUT / f"preds_{args.out_prefix}.npz",
        hist_row_id=hist.row_id.values,
        hist_customer_id=hist.customer_id.values,
        hist_y=y,
        hist_oof=hist_oof,
        hist_cust_fold=hist.cust_fold.values,
        current_row_id=cur.row_id.values,
        current_application_id=cur.own_application_id.values,
        current_customer_id=cur.customer_id.values,
        current_is_train=cur.label.notna().values,
        current_y=y_cur,
        current_score=cur_score,
        current_cust_fold=cur.cust_fold.values,
    )
    summary = {
        "data": str(args.data),
        "folds": args.folds,
        "seeds": seeds,
        "configs": {c: CONFIGS[c] for c in configs},
        "hist_rows": int(len(hist)),
        "current_rows": int(len(cur)),
        "features": len(feat),
        "hist_positive_rate": float(y.mean()),
        "panel_hist_auc": float(panel_auc),
        "panel_hist_ks": float(panel_ks),
        "panel_current_auc": float(cur_auc),
        "panel_current_ks": float(cur_ks),
        "per_fold": per_fold,
        "minutes": (time.time() - t0) / 60.0,
        "feature_names": feat,
        "note": "GroupKFold by customer. Current train rows are scored by the fold "
                "model that did not see that customer's historical labels. "
                "No current target is used anywhere in panel training or scoring.",
    }
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"saved preds_{args.out_prefix}.npz in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
