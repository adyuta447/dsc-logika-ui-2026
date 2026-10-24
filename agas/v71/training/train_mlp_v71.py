"""Small tabular MLP diversity leg for v71.

The LightGBM/CatBoost family shares a tree-based inductive bias.  This leg uses
the same leak-free lean columns but a standardised MLP, so its errors are more
decorrelated.  It is deliberately small and regularised; the goal is a blend
leg, not a standalone winner.

Preprocessing is fitted inside each fold: median imputation, 0.1/99.9 percentile
clipping, standardisation, plus a missing-count feature.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"
NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Net(nn.Module):
    def __init__(self, n_in, hidden=256, hidden2=64, dropout=0.30, dropout2=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2), nn.BatchNorm1d(hidden2), nn.SiLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_predict(x_tr, y_tr, x_va, y_va, x_te, seed, epochs, batch,
                hidden, hidden2, dropout, dropout2, lr, wd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_in = x_tr.shape[1]
    model = Net(n_in, hidden, hidden2, dropout, dropout2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(x_va)
    xe = torch.from_numpy(x_te)
    n = len(xt)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).numpy()
        auc = roc_auc_score(y_va, pv)
        sched.step(auc)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(xv)).numpy()
        pt = torch.sigmoid(model(xe)).numpy()
    return pv, pt, best_auc, ep + 1


def prep(train_df, valid_df, test_df, cols):
    a = train_df[cols].to_numpy(dtype=np.float32, copy=True)
    b = valid_df[cols].to_numpy(dtype=np.float32, copy=True)
    c = test_df[cols].to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    miss_a = np.isnan(a).sum(1, keepdims=True)
    miss_b = np.isnan(b).sum(1, keepdims=True)
    miss_c = np.isnan(c).sum(1, keepdims=True)
    for x in (a, b, c):
        idx = np.where(np.isnan(x))
        x[idx] = np.take(med, idx[1])
    lo = np.nanquantile(a, 0.001, axis=0)
    hi = np.nanquantile(a, 0.999, axis=0)
    a = np.clip(a, lo, hi)
    b = np.clip(b, lo, hi)
    c = np.clip(c, lo, hi)
    mu = a.mean(0)
    sd = a.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    a = (a - mu) / sd
    b = (b - mu) / sd
    c = (c - mu) / sd
    a = np.concatenate([a, miss_a], axis=1)
    b = np.concatenate([b, miss_b], axis=1)
    c = np.concatenate([c, miss_c], axis=1)
    return (a.astype(np.float32), b.astype(np.float32), c.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(OUT / "features_v71.pkl"))
    ap.add_argument("--drops", default="pn_,rf_")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", default="42")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--dropout2", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--out-prefix", default="mlp_v71")
    args = ap.parse_args()
    torch.set_num_threads(10)
    t0 = time.time()
    drops = tuple(s.strip() for s in args.drops.split(",") if s.strip())
    f = pd.read_pickle(args.features)
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X = tr.drop(columns=ids)
    Xte = te.drop(columns=ids)
    lean = [c for c in X.columns
            if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)
            and not c.startswith(drops)]
    # all lean columns are numeric; guard anyway
    lean = [c for c in lean if pd.api.types.is_numeric_dtype(X[c])]
    y = tr.target.astype(int).values
    log(f"features={Path(args.features).name} rows={len(X)} lean={len(lean)}")

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    oof_by_fs, test_by_fs, results = {}, {}, {}
    for fs in fold_seeds:
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=fs).split(X, y))
        oof_sum = np.zeros(len(X))
        test_sum = np.zeros(len(Xte))
        seed_aucs = []
        for sd in seeds:
            oof = np.zeros(len(X))
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                xa, xb, xc = prep(X.iloc[a], X.iloc[b], Xte, lean)
                pv, pt, auc, ep = fit_predict(
                    xa, y[a], xb, y[b], xc, sd * 100 + k, args.epochs, args.batch,
                    args.hidden, args.hidden2, args.dropout, args.dropout2,
                    args.lr, args.wd)
                oof[b] = pv
                pred += pt / len(folds)
                log(f"  fs={fs} sd={sd} fold={k} ep={ep} auc={auc:.6f}")
            auc = roc_auc_score(y, oof)
            seed_aucs.append(float(auc))
            log(f"  fs={fs} sd={sd} auc={auc:.6f} ks={ks(y, oof):.6f}")
            oof_sum += oof / len(seeds)
            test_sum += pred / len(seeds)
        auc = roc_auc_score(y, oof_sum)
        results[str(fs)] = {"auc": float(auc), "ks": ks(y, oof_sum),
                            "auc_seed_mean": float(np.mean(seed_aucs)),
                            "auc_seed_sd": float(np.std(seed_aucs))}
        oof_by_fs[f"oof_{fs}"] = oof_sum
        test_by_fs[f"test_{fs}"] = test_sum
        log(f"fs={fs} auc={auc:.6f} ks={ks(y, oof_sum):.6f}")
        np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                            train_ids=tr.application_id.values,
                            test_ids=te.application_id.values, y=y,
                            **oof_by_fs, **test_by_fs)
    vals = [results[str(fs)]["auc"] for fs in fold_seeds]
    ksv = [results[str(fs)]["ks"] for fs in fold_seeds]
    summary = {
        "features": str(args.features), "drops": drops, "folds": args.folds,
        "fold_seeds": fold_seeds, "seeds": seeds, "n_lean": len(lean),
        "params": {"hidden": args.hidden, "hidden2": args.hidden2,
                   "dropout": args.dropout, "dropout2": args.dropout2,
                   "lr": args.lr, "wd": args.wd, "epochs": args.epochs},
        "mean_auc": float(np.mean(vals)), "sd_auc": float(np.std(vals)),
        "mean_ks": float(np.mean(ksv)), "per_fold_seed": results,
        "minutes": (time.time() - t0) / 60.0,
    }
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"SUMMARY auc={summary['mean_auc']:.6f} sd={summary['sd_auc']:.6f} "
        f"ks={summary['mean_ks']:.6f} in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
