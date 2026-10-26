"""Fase 2 candidates: the dropped-columns leg as its own lever.

The leg exists because of a hypothesis that turned out to be wrong - that
raising the extrapolation ratio amplifies the 108 columns `lean` drops. The
direction test refuted it: raising `r` is anti-correlated with adding this leg
(Pearson -0.68) and with adding any leg at all, because raising `r` subtracts.

The leg is still worth keeping, for a different reason than the one it was
built for. It reaches OOF AUC 0.836393 on 108 columns alone - dominated by the
66 `gn_` columns rather than the raw demographics - and it has never been part
of the chain. So it is an independent second lever, ordered behind the `r` line
search because `r` has a measured gradient and this does not.

Candidates are built on top of `r=0.75`, the best measured point, so they ask
only one question: does adding this leg to the current best help. No
out-of-fold gate is applied, because out-of-fold is measured anti-correlated
with public in this region.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"
R_BEST = 0.75
PUBLIC_R075 = 0.56520


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    tr, te = pd.Index(z["train_ids"]), pd.Index(z["test_ids"])
    o = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}
    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[tr].target.values.astype(int))

    d = np.load(OUT / "preds_dropped_v77.npz", allow_pickle=True)
    si = pd.Index(d["train_ids"]).get_indexer(tr)
    ti = pd.Index(d["test_ids"]).get_indexer(te)
    assert (si >= 0).all() and (ti >= 0).all(), "id mismatch with the dropped leg"
    o["dropped"] = rank01(d["oof_dropped"][si])
    t["dropped"] = rank01(d["test_dropped"][ti])

    def base_of(dd, tm=0.30, cw=2.0):
        trees4 = (dd["lgb_full"] + dd["lgb_lean"] + dd["cat_full"] + dd["xgb_full"]) / 4
        core = (trees4 * cw + dd["lgbx_full"]) / (cw + 1.0)
        return (1 - tm) * core + tm * dd["linear"]

    def chain(dd, r=R_BEST):
        b = base_of(dd)
        return rank01(b + r * (b - dd["lgb_lean"]))

    best_o, best_t = chain(o), chain(t)
    b_auc, b_ks = float(roc_auc_score(y, best_o)), ks(best_o, y)
    print(f"dropped leg       : OOF AUC {roc_auc_score(y, o['dropped']):.6f} "
          f"KS {ks(o['dropped'], y):.6f}")
    print(f"r=0.75 reference  : OOF AUC {b_auc:.6f} KS {b_ks:.6f}, "
          f"public {PUBLIC_R075:.5f}")

    # the refutation, recorded in the artefact rather than only in prose
    dir_r = chain(t, 1.00) - chain(t, 0.50)
    refute = {}
    for leg in ("dropped", "lgb_lean", "lgb_full", "linear", "cat_full",
                "xgb_full", "lgbx_full", "fac_full"):
        dir_leg = rank01(base_of(t) + 0.20 * t[leg]) - chain(t, 0.50)
        refute[leg] = round(float(pearsonr(dir_r, dir_leg).statistic), 4)
    print("\ndirection of raising r vs adding 20% of a leg (Pearson):")
    for k, v in sorted(refute.items(), key=lambda kv: kv[1]):
        print(f"  {k:12s} {v:+.4f}")
    print("all negative: raising r subtracts, so it cannot be 'amplifying' any leg")

    cands = {}
    for u in (0.10, 0.20, 0.35):
        cands[f"r0.75_plus_dropped{int(u*100)}"] = (
            rank01((1 - u) * best_o + u * o["dropped"]),
            rank01((1 - u) * best_t + u * t["dropped"]))
    # the dropped leg inside trees4 instead of mixed on top, so it is averaged
    # with the other trees rather than layered over the finished chain
    def chain_with_dropped(dd, r=R_BEST, tm=0.30, cw=2.0):
        trees5 = (dd["lgb_full"] + dd["lgb_lean"] + dd["cat_full"]
                  + dd["xgb_full"] + dd["dropped"]) / 5
        core = (trees5 * cw + dd["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * dd["linear"]
        return rank01(base + r * (base - dd["lgb_lean"]))
    cands["r0.75_dropped_in_trees5"] = (chain_with_dropped(o), chain_with_dropped(t))

    rows, vecs = [], {}
    for name, (vo, vt) in cands.items():
        rows.append({"name": name,
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - b_auc, 6),
                     "d_oof_ks": round(ks(vo, y) - b_ks, 6),
                     "spearman_vs_r075": round(
                         float(spearmanr(vt, best_t).statistic), 6)})
        vecs[name] = vt

    print(f"\n{'candidate':30s} {'OOF AUC':>10} {'dAUC':>10} {'dKS':>10} {'rho':>9}")
    for r in rows:
        print(f"{r['name']:30s} {r['oof_auc']:10.6f} {r['d_oof_auc']:+10.6f} "
              f"{r['d_oof_ks']:+10.6f} {r['spearman_vs_r075']:9.6f}")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v77d_*.csv")):
        f.unlink()
    for r in rows:
        p = OUT / f"submission_v77d_{r['name'].replace('.', 'p')}.csv"
        frame = pd.DataFrame({"application_id": te.values,
                              "default_probability": rank01(vecs[r["name"]])})
        frame.to_csv(p, index=False)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.default_probability.isna().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name
        print(f"  wrote {p.name}")

    (OUT / "summary_v77_dropped.json").write_text(json.dumps(
        {"dropped_leg_oof_auc": float(roc_auc_score(y, o["dropped"])),
         "dropped_leg_oof_ks": ks(o["dropped"], y),
         "reference_r": R_BEST, "reference_public": PUBLIC_R075,
         "reference_oof_auc": b_auc, "reference_oof_ks": b_ks,
         "refuted_hypothesis": "raising r amplifies the 108 dropped columns",
         "direction_pearson_vs_adding_leg": refute,
         "candidates": rows}, indent=2))
    print(f"\nsaved {OUT / 'summary_v77_dropped.json'}")


if __name__ == "__main__":
    main()
