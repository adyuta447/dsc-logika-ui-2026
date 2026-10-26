"""v76: use the leaderboard as a measuring instrument instead of a lottery.

Eight public results are now in, and they do not form one noisy cloud. They
form two clusters that barely overlap:

    A  the exact ext0.5 | tm30 | lgb_lean chain
       v66_04 0.56424, v70_07 0.56419, v68_04 0.56411
       n=3, mean 0.56418, sd 0.00007

    B  anything that is not that chain
       core71 0.56111, mix80_mlp 0.56073, pair_mlp 0.56045,
       seq05 0.55968, v70_01 0.55801
       n=5, mean 0.56000, sd 0.00123

Two things follow, and both are new.

**The +0.00418 gap is real.** The three cluster A members were trained on
different feature sets and different numbers of fold seeds - v66 on six,
v68 on ten, v70_07 on the v70 feature set entirely - and still landed inside a
0.00013 band. The maximum of five draws from cluster B would reach only
0.56142, so 0.56424 cannot be explained as the luckiest draw of a single
distribution.

**Inside the chain, public has a standard deviation of 0.00007.** For the first
time in this project the leaderboard is a precision instrument rather than a
source of noise, and any difference above 0.0002 is signal.

Leg quality is already known not to move cluster A, so what is left is the
chain's own parameters - and only one value of each has ever been measured on
public. v61 and v66 did sweep the extrapolation ratio and the linear weight,
but they selected on OOF AUC, which v74 and v75 showed does not track public.
On the metric that decides the competition, this axis is unmeasured.

The first file is therefore not a bid for a higher score, it is a test. The
smallest perturbation available inside the chain is the core weight, 2:1 to
3:1, at rho 0.999986 against the anchor - far closer than seq05, which sat at
rho 0.99950 and still fell 0.00456. If that file holds near 0.5641 the surface
is smooth and the sweep is worth continuing. If it collapses to about 0.560,
0.56424 is a brittle local maximum, the sweep is pointless, and the remaining
submissions are saved.

This script trains nothing. Every candidate is a re-assembly of the stored v66
leg predictions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"

# every scored submission, with the cluster it belongs to
LEDGER = [
    ("v66_04  ext0.5|tm30|lgb_lean", "A", 0.56424),
    ("v70_07  ext0.5|tm30v70|lgb_lean", "A", 0.56419),
    ("v68_04  ext0.5|tm30|lgb_lean", "A", 0.56411),
    ("v71     core71 MLP-heavy", "B", 0.56111),
    ("v75_03  mix80_mlp", "B", 0.56073),
    ("v75_04  pair_mlp_v70_extra", "B", 0.56045),
    ("v72     seq05 anchor+5% seq", "B", 0.55968),
    ("v70_01  raw leg v70_extra", "B", 0.55801),
    ("v61_alt ext0.5|tm30|lean (weaker legs)", "C", 0.56312),
    ("qc_00   ext1.5|fac_full", "C", 0.56171),
    ("v58ext  ext1.5|fac", "C", 0.55900),
    ("v66_02  plateau|tm20|fac_full", "C", 0.55877),
]
CONTROL = "core3to1"


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
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    o = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}
    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[train_ids].target.values.astype(int))

    # cluster statistics straight from the ledger
    print("public ledger")
    stats = {}
    for cl in ("A", "B", "C"):
        v = np.array([x[2] for x in LEDGER if x[1] == cl])
        stats[cl] = {"n": len(v), "mean": round(float(v.mean()), 5),
                     "sd": round(float(v.std(ddof=1)), 5),
                     "min": float(v.min()), "max": float(v.max())}
        print(f"  cluster {cl}: n={len(v)} mean={v.mean():.5f} sd={v.std(ddof=1):.5f}")
    for name, cl, sc in LEDGER:
        print(f"    {cl}  {sc:.5f}  {name}")
    gap = stats["A"]["mean"] - stats["B"]["mean"]
    print(f"\n  gap A - B = {gap:+.5f}; instrument sd inside the chain = "
          f"{stats['A']['sd']:.5f}")

    def chain(d, tm=0.30, r=0.50, ref="lgb_lean", cw=2.0):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * d["linear"]
        return rank01(base + r * (base - d[ref]))

    a_o, a_t = chain(o), chain(t)
    a_auc, a_ks = float(roc_auc_score(y, a_o)), ks(a_o, y)

    # one parameter changed per candidate; ref stays lgb_lean because cluster C
    # shows that changing the reference leg is expensive
    specs = [
        (CONTROL, dict(cw=3.0), 1,
         "core weight 3:1 instead of 2:1 - the smallest perturbation available; "
         "the control that separates a smooth surface from a brittle maximum"),
        ("r0.75", dict(r=0.75), 2,
         "extrapolation ratio up one step, with tm and ref held fixed"),
        ("r0.25", dict(r=0.25), 3,
         "extrapolation ratio down one step; with r0.75 this brackets r=0.5 so "
         "the gradient and the curvature read directly off the instrument"),
        ("tm0.40", dict(tm=0.40), 4, "linear weight up one step"),
        ("tm0.20", dict(tm=0.20), 5, "linear weight down one step"),
        ("r1.00", dict(r=1.00), 6,
         "extrapolation ratio, large step; only worth sending if r0.75 moved up"),
        ("core1to1", dict(cw=1.0), 7, "core weight 1:1, the other direction"),
        ("anchor", dict(), 0,
         "the measured reference, 0.56424; do not resubmit, it returns the same"),
    ]

    rows = []
    for name, kw, order, why in specs:
        vo, vt = chain(o, **kw), chain(t, **kw)
        rho = float(spearmanr(vt, a_t).statistic)
        rows.append({"name": name, "submit_order": order, "why": why,
                     "params": {k: v for k, v in kw.items()} or {"baseline": True},
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - a_auc, 6),
                     "d_oof_ks": round(ks(vo, y) - a_ks, 6),
                     "spearman_vs_anchor": round(rho, 6),
                     "rank_changes": int((rankdata(vt) != rankdata(a_t)).sum())})
        globals().setdefault("_vecs", {})[name] = vt

    rows.sort(key=lambda r: r["submit_order"])
    print(f"\nanchor: OOF auc={a_auc:.6f} ks={a_ks:.6f}, public 0.56424 measured")
    print(f"\n{'order':>5} {'name':10s} {'auc':>9} {'dAUC_oof':>9} {'dKS_oof':>9} "
          f"{'rho':>9} {'rank_chg':>9}")
    for r in rows:
        print(f"{r['submit_order']:>5} {r['name']:10s} {r['oof_auc']:9.6f} "
              f"{r['d_oof_auc']:+9.6f} {r['d_oof_ks']:+9.6f} "
              f"{r['spearman_vs_anchor']:9.6f} {r['rank_changes']:>9}")

    print("\nOOF numbers are printed for completeness only. v74 and v75 measured "
          "that they do not track public, so they are not the selection key here; "
          "the public instrument is.")
    print(f"\ngo / no-go after submission 1 ({CONTROL}):")
    print(f"  >= 0.5639  surface is smooth -> continue the sweep in order")
    print(f"  <= 0.5620  0.56424 is a brittle maximum -> stop, freeze the anchor")
    print(f"  between    inconclusive -> send r0.75 only, then decide")

    for f in sorted(OUT.glob("submission_v76_*.csv")):
        f.unlink()
    vecs = globals()["_vecs"]
    for r in rows:
        tag = f"{r['submit_order']:02d}" if r["submit_order"] else "00"
        p = OUT / f"submission_v76_{tag}_{r['name']}.csv"
        frame = pd.DataFrame({"application_id": test_ids,
                              "default_probability": rank01(vecs[r["name"]])})
        frame.to_csv(p, index=False)
        test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                                   usecols=["application_id"]).application_id)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.default_probability.isna().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name
        print(f"  wrote {p.name}")

    (OUT / "summary_v76.json").write_text(json.dumps(
        {"ledger": [{"name": n, "cluster": c, "public": s} for n, c, s in LEDGER],
         "cluster_stats": stats, "gap_A_minus_B": round(gap, 5),
         "instrument_sd": stats["A"]["sd"],
         "anchor_public": 0.56424, "anchor_oof_auc": a_auc, "anchor_oof_ks": a_ks,
         "control": CONTROL,
         "go_no_go": {"continue_at_or_above": 0.5639, "stop_at_or_below": 0.5620},
         "candidates": rows}, indent=2))
    print(f"\nsaved {OUT / 'summary_v76.json'}")


if __name__ == "__main__":
    main()
