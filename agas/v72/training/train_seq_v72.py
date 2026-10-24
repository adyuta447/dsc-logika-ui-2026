"""Supervised monthly-sequence model for v72.

This is the structural test that v71 deferred: instead of compressing account,
repayment, e-commerce, bureau and application history into static aggregates, it
feeds the last 24 months as a sequence to a small bidirectional GRU.  The static
tabular block from v70 is optional and is concatenated with the sequence
representation, so the run can answer two questions separately:

* does the sequence add anything over the same static features (``--static``)?
* is a pure sequence leg useful as a decorrelated blend component?

The model is small on purpose.  The goal is a blend leg on 70k labels, not a
leaderboard single-model winner.  All preprocessing is fitted on training rows
only and all sequences are point-in-time at ``decision_datetime``.
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


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


class SeqNet(nn.Module):
    def __init__(self, c_in, c_mask, static_dim, hidden=64, static_hidden=256,
                 dropout=0.25, static_dropout=0.35):
        super().__init__()
        self.gru = nn.GRU(c_in + c_mask, hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        seq_repr = 4 * 2 * hidden
        self.static = None
        if static_dim:
            self.static = nn.Sequential(
                nn.Linear(static_dim, static_hidden), nn.BatchNorm1d(static_hidden), nn.SiLU(),
                nn.Dropout(static_dropout),
                nn.Linear(static_hidden, 64), nn.BatchNorm1d(64), nn.SiLU(),
            )
            seq_repr += 64
        self.head = nn.Sequential(
            nn.Linear(seq_repr, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x, m, s=None):
        # m: [B,L,Cmask]; x is already masked to zero at missing cells
        lengths = m.max(-1).values.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            torch.cat([x, m], dim=-1), lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        att = self.att(h).squeeze(-1).masked_fill(~valid, -1e9)
        w = torch.softmax(att, dim=1)
        ctx = (h * w.unsqueeze(-1)).sum(1)
        last = h[torch.arange(h.shape[0], device=h.device), lengths - 1]
        denom = lengths.float().unsqueeze(-1)
        mean = (h * valid.unsqueeze(-1)).sum(1) / denom
        mx = h.masked_fill(~valid.unsqueeze(-1), -1e9).max(1).values
        z = torch.cat([ctx, last, mean, mx], dim=-1)
        if self.static is not None:
            z = torch.cat([z, self.static(s)], dim=-1)
        return self.head(self.drop(z)).squeeze(1)


def load_static(path, app_ids):
    f = pd.read_pickle(path)
    f = f.set_index("application_id").reindex(app_ids)
    ids = ["customer_id", "target"]
    drop = [c for c in ids if c in f.columns]
    X = f.drop(columns=drop)
    # Keep only numeric columns; categorical encodings are handled by the tree legs.
    keep = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    keep = [c for c in keep if c not in NOISE_EXACT
            and not any(c.startswith(p) for p in NOISE_PREFIX)]
    return X[keep].to_numpy(np.float32), keep


def fit_predict(x_tr, m_tr, s_tr, y_tr, x_va, m_va, s_va, y_va, x_te, m_te, s_te,
                seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    c_in = x_tr.shape[2]
    c_mask = m_tr.shape[2]
    static_dim = s_tr.shape[1] if s_tr is not None else 0
    model = SeqNet(c_in, c_mask, static_dim, args.hidden, args.static_hidden,
                   args.dropout, args.static_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    xt = torch.from_numpy(x_tr)
    mt = torch.from_numpy(m_tr)
    st = torch.from_numpy(s_tr) if s_tr is not None else None
    yt = torch.from_numpy(y_tr.astype(np.float32))
    xv = torch.from_numpy(x_va)
    mv = torch.from_numpy(m_va)
    sv = torch.from_numpy(s_va) if s_va is not None else None
    xe = torch.from_numpy(x_te)
    me = torch.from_numpy(m_te)
    se = torch.from_numpy(s_te) if s_te is not None else None
    n = len(xt)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            xb = xt[idx].to(device); mb = mt[idx].to(device)
            sb = st[idx].to(device) if st is not None else None
            yb = yt[idx].to(device)
            loss = lossf(model(xb, mb, sb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv.to(device), mv.to(device),
                                     sv.to(device) if sv is not None else None)).cpu().numpy()
        auc = roc_auc_score(y_va, pv)
        sched.step(auc)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(xv.to(device), mv.to(device),
                                 sv.to(device) if sv is not None else None)).cpu().numpy()
        pt = torch.sigmoid(model(xe.to(device), me.to(device),
                                 se.to(device) if se is not None else None)).cpu().numpy()
    return pv, pt, best_auc, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=str(OUT / "sequences_v72.npz"))
    ap.add_argument("--features", default=str(ROOT / "agas/v70/output/features_v70.pkl"))
    ap.add_argument("--static", action="store_true", help="append the v70 static tabular block")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seeds", default="42")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--static-hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--static-dropout", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-prefix", default="seq_v72")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(10)
    t0 = time.time()
    z = np.load(args.sequence, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = z["M"].astype(np.float32)
    y_all = z["y"].astype(np.float32)
    app_ids = z["application_id"].astype(str)
    tr_idx_all = np.where(~np.isnan(y_all))[0]
    te_idx_all = np.where(np.isnan(y_all))[0]
    y = y_all[tr_idx_all].astype(int)
    Xtr, Mtr = X[tr_idx_all], M[tr_idx_all]
    Xte, Mte = X[te_idx_all], M[te_idx_all]
    log(f"sequence={Path(args.sequence).name} X={Xtr.shape} train={len(y)} test={len(Xte)} device={args.device}")

    S = None
    static_cols = []
    if args.static:
        S_all, static_cols = load_static(args.features, app_ids)
        S = S_all[tr_idx_all]
        Ste = S_all[te_idx_all]
        log(f"static columns={len(static_cols)}")
    else:
        Ste = None

    fold_seeds = [int(s) for s in args.fold_seeds.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    oof_by, test_by, results = {}, {}, {}
    for fs in fold_seeds:
        log(f"fold seed {fs}")
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=fs).split(Xtr, y))
        oof_sum = np.zeros(len(y))
        test_sum = np.zeros(len(Xte))
        seed_aucs = []
        for sd in seeds:
            oof = np.zeros(len(y))
            pred = np.zeros(len(Xte))
            for k, (a, b) in enumerate(folds):
                # unsupervised channel scaling fitted on this training fold
                xx_tr = Xtr[a]
                mu = xx_tr.mean(axis=(0, 1)); sdv = xx_tr.std(axis=(0, 1))
                sdv = np.where(sdv > 1e-8, sdv, 1.0)
                src = np.repeat(np.arange(Mtr.shape[2]), [10, 13, 16, 8, 9])
                xa = np.clip((Xtr[a] - mu) / sdv, -8, 8).astype(np.float32) * Mtr[a][:, :, src]
                xb = np.clip((Xtr[b] - mu) / sdv, -8, 8).astype(np.float32) * Mtr[b][:, :, src]
                xc = np.clip((Xte - mu) / sdv, -8, 8).astype(np.float32) * Mte[:, :, src]
                if S is not None:
                    araw = S[a].astype(np.float32, copy=True)
                    med = np.nanmedian(araw, axis=0); med = np.where(np.isfinite(med), med, 0.0)
                    miss_tr = np.isnan(S).sum(1, keepdims=True).astype(np.float32)
                    miss_te = np.isnan(Ste).sum(1, keepdims=True).astype(np.float32)
                    lo = np.nanquantile(araw, 0.001, axis=0)
                    hi = np.nanquantile(araw, 0.999, axis=0)
                    def tf(raw, miss):
                        raw = raw.copy()
                        ii = np.where(np.isnan(raw)); raw[ii] = np.take(med, ii[1])
                        raw = np.clip(raw, lo, hi)
                        raw = (raw - araw.mean(0)) / np.where(araw.std(0) > 1e-8, araw.std(0), 1.0)
                        return np.concatenate([raw, miss], axis=1).astype(np.float32)
                    sa = tf(S[a], miss_tr[a]); sb = tf(S[b], miss_tr[b]); sc = tf(Ste, miss_te)
                else:
                    sa = sb = sc = None
                pv, pt, auc, ep = fit_predict(
                    xa, Mtr[a], sa, y[a], xb, Mtr[b], sb, y[b], xc, Mte, sc, sd * 1000 + k, args)
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
        oof_by[f"oof_{fs}"] = oof_sum
        test_by[f"test_{fs}"] = test_sum
        log(f"fs={fs} auc={auc:.6f} ks={ks(y, oof_sum):.6f}")
        np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                            train_ids=z["application_id"][tr_idx_all],
                            test_ids=z["application_id"][te_idx_all], y=y,
                            **oof_by, **test_by)
    vals = [results[str(fs)]["auc"] for fs in fold_seeds]
    ksv = [results[str(fs)]["ks"] for fs in fold_seeds]
    summary = {
        "sequence": str(args.sequence), "features": str(args.features),
        "static": bool(args.static), "n_static": len(static_cols),
        "folds": args.folds, "fold_seeds": fold_seeds, "seeds": seeds,
        "params": {"hidden": args.hidden, "static_hidden": args.static_hidden,
                   "dropout": args.dropout, "static_dropout": args.static_dropout,
                   "lr": args.lr, "wd": args.wd, "epochs": args.epochs,
                   "batch": args.batch, "pos_weight": args.pos_weight,
                   "patience": args.patience, "device": args.device},
        "mean_auc": float(np.mean(vals)), "sd_auc": float(np.std(vals)),
        "mean_ks": float(np.mean(ksv)), "per_fold_seed": results,
        "minutes": (time.time() - t0) / 60.0,
    }
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"SUMMARY auc={summary['mean_auc']:.6f} sd={summary['sd_auc']:.6f} "
        f"ks={summary['mean_ks']:.6f} in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
