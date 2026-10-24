"""Build v72 submission artifacts and record the sequence gate.

v72 tested a genuinely different representation: the last 24 months of account,
repayment, e-commerce, bureau and application events as a sequence.  The
sequence leg is a real signal (pure OOF AUC 0.7833), but it is far below the
static family (0.8484) and does not improve the blend's KS.  Therefore no v72
candidate is promoted over the v66_04 anchor.  The files below exist so the
negative result is reproducible and so the anchor control is explicit.

Candidates:

* ``01_anchor``     exact v66_04 rank vector, public 0.56424 control.
* ``02_anchor_seq05``   95% anchor + 5% pure sequence; diagnostic only.
* ``03_anchor_seqst15`` 85% anchor + 15% static-fused sequence; rejected.

Only the anchor is recommended.  The two sequence blends are kept because a
future run may want to test whether *any* sequence signal transfers to public;
their OOF gain is below the noise floor and their KS is not higher.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"
ANCHOR = ROOT / "agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv"
Z66 = ROOT / "agas/v66/output/preds_v66.npz"
Z72P = OUT / "preds_seq_v72_pure.npz"
Z72S = OUT / "preds_seq_v72_static.npz"
SIGMA_PUB = 0.0053


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def anchor_oof():
    z = np.load(Z66, allow_pickle=True)
    train_ids = z["train_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv",
                    usecols=["application_id", "target"]).set_index(
        "application_id").loc[train_ids].target.values.astype(int)
    ro = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    trees = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    core = (2 * trees + ro["lgbx_full"]) / 3
    tm30 = 0.7 * core + 0.3 * ro["linear"]
    p = rank01(tm30 + 0.5 * (tm30 - ro["lgb_lean"]))
    return train_ids, y, p


def bootstrap_dks(y, base, cand, n_boot=2000, n=30000, seed=20261024):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    npos = int(round(n * y.mean()))
    nneg = n - npos
    out = []
    for _ in range(n_boot):
        ii = np.concatenate([rng.choice(pos, npos, replace=True),
                             rng.choice(neg, nneg, replace=True)])
        yy = y[ii]
        kb = ks_2samp(base[ii][yy == 1], base[ii][yy == 0]).statistic
        kc = ks_2samp(cand[ii][yy == 1], cand[ii][yy == 0]).statistic
        out.append(kc - kb)
    a = np.asarray(out)
    return {"mean_dks": float(a.mean()), "sd": float(a.std()),
            "p_gt0": float((a > 0).mean()),
            "p_gt_0.001": float((a > 0.001).mean())}


def main():
    train_ids, y, a_oof = anchor_oof()
    z66 = np.load(Z66, allow_pickle=True)
    test_ids = z66["test_ids"]
    anchor_csv = pd.read_csv(ANCHOR)
    assert list(anchor_csv.columns) == ["application_id", "default_probability"]
    assert anchor_csv.application_id.tolist() == list(test_ids)
    a_test = rank01(anchor_csv.default_probability.values)

    zp = np.load(Z72P, allow_pickle=True)
    zs = np.load(Z72S, allow_pickle=True)
    assert np.array_equal(zp["train_ids"], train_ids)
    assert np.array_equal(zs["train_ids"], train_ids)
    assert np.array_equal(zp["test_ids"], test_ids)
    seq_o, seq_t = rank01(zp["oof_42"]), rank01(zp["test_42"])
    seqs_o, seqs_t = rank01(zs["oof_42"]), rank01(zs["test_42"])

    specs = [
        ("01_anchor", 0.0, None, None, "recommended anchor control; public 0.56424"),
        ("02_anchor_seq05", 0.05, seq_o, seq_t,
         "diagnostic only: 5% pure sequence; OOF gain is below noise"),
        ("03_anchor_seqst15", 0.15, seqs_o, seqs_t,
         "rejected: static-fused sequence lowers OOF KS"),
    ]
    rows = []
    for name, w, oo, tt, note in specs:
        if w == 0:
            co, ct = a_oof, a_test
        else:
            co = rank01((1 - w) * a_oof + w * oo)
            ct = rank01((1 - w) * a_test + w * tt)
        path = OUT / f"submission_v72_{name}.csv"
        if w == 0:
            # Keep the control byte-identical to the public 0.56424 file.
            shutil.copyfile(ANCHOR, path)
        else:
            pd.DataFrame({"application_id": test_ids, "default_probability": ct}).to_csv(path, index=False)
        ok = (len(ct) == 30000 and len(set(test_ids)) == 30000
              and np.isfinite(ct).all() and (ct >= 0).all() and (ct <= 1).all())
        rows.append({
            "name": name, "file": path.name, "weight": w,
            "oof_auc": float(roc_auc_score(y, co)), "oof_ks": ks(y, co),
            "spearman_anchor": float(spearmanr(ct, a_test).statistic),
            "valid": bool(ok), "recommended": name == "01_anchor", "note": note,
        })

    # Bootstrap only the non-anchor candidates, against the anchor on the same rows.
    boot = {}
    for name, w, oo, tt, _ in specs[1:]:
        co = rank01((1 - w) * a_oof + w * oo)
        boot[name] = bootstrap_dks(y, a_oof, co)

    summary = {
        "anchor": {"file": str(ANCHOR), "oof_auc": float(roc_auc_score(y, a_oof)),
                   "oof_ks": ks(y, a_oof), "n": int(len(y))},
        "sequence": {
            "pure": {"oof_auc": float(roc_auc_score(y, seq_o)), "oof_ks": ks(y, seq_o),
                     "spearman_anchor": float(spearmanr(seq_o, a_oof).statistic)},
            "static_fused": {"oof_auc": float(roc_auc_score(y, seqs_o)), "oof_ks": ks(y, seqs_o),
                             "spearman_anchor": float(spearmanr(seqs_o, a_oof).statistic)},
        },
        "bootstrap_dks_vs_anchor": boot,
        "sigma_public": SIGMA_PUB,
        "candidates": rows,
        "decision": "no sequence candidate promoted; keep v66_04 anchor",
    }
    (OUT / "summary_v72_candidates.json").write_text(json.dumps(summary, indent=2))

    print("v72 candidate gate")
    for r in rows:
        print(f"  {r['name']:22s} w={r['weight']:.2f} auc={r['oof_auc']:.6f} "
              f"ks={r['oof_ks']:.6f} rho={r['spearman_anchor']:.6f} "
              f"valid={r['valid']} recommended={r['recommended']}")
    for k, v in boot.items():
        print(f"  bootstrap {k}: dKS={v['mean_dks']:+.6f} "
              f"P(dKS>0)={v['p_gt0']:.3f}")
    print(f"saved {OUT / 'summary_v72_candidates.json'}")


if __name__ == "__main__":
    main()
