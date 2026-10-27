"""v78: the reference leg, measured at matched subtraction strength.

The `r` axis is now mapped and closed. Three public points on it -
r=0.50 -> 0.56418 (mean of three), r=0.75 -> 0.56520, r=1.00 -> 0.56464 - fit a
quadratic with curvature -0.01264, peak at r=0.7864 and peak value 0.565217.
That is +0.000017 over the measured r=0.75, a quarter of the instrument's
0.00007 spread, so no further point on this axis is worth a submission. The
registered rule said to send r=0.875 next; the third point made that rule
obsolete and the quadratic puts r=0.875 at 0.56512, below the current best.
Net gain from the axis: +0.00102.

What the axis bought besides the score is the mechanism. Extrapolation is
`base - w*lgb_lean` with `w = r/(1+r)`, so the chain subtracts a leg, and `r`
only sets how hard. The leg being subtracted was never chosen - it is `lgb_lean`
because v61 wrote it that way. That is the remaining degree of freedom.

**Matched subtraction strength.** Comparing references at a fixed `r` would
compare different things, because each leg already carries its own weight
inside `base`: 0.11667 for a `trees4` member, 0.23333 for `lgbx_full`, 0.30 for
`linear`, and 0 for `fac_full` and the dropped leg, which are not in `base` at
all. So each reference here gets the `r` that puts its **net** coefficient at
-0.31190, the value at the measured optimum:

    w = weight_in_base(ref) + 0.31190,    r = w / (1 - w)

That makes the comparison about *which* leg is worth subtracting, holding *how
much* fixed. The fixed-`r` variants are built too, so the difference between
the two framings is on the record rather than assumed.

No out-of-fold gate is applied. On the `r` axis out-of-fold fell monotonically
while public rose and then fell, so it does not track public here; it is
reported for completeness only.

This script trains nothing.
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

TM, CW = 0.30, 2.0
R_BEST = 0.75
TARGET_NET = -0.31190        # net coefficient on the reference at r=0.75
MEASURED = {0.50: 0.56418, 0.75: 0.56520, 1.00: 0.56464}
INSTRUMENT_SD = 0.00007


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

    dpath = AGAS / "v77/output/preds_dropped_v77.npz"
    if dpath.exists():
        d = np.load(dpath, allow_pickle=True)
        si = pd.Index(d["train_ids"]).get_indexer(tr)
        ti = pd.Index(d["test_ids"]).get_indexer(te)
        o["dropped"] = rank01(d["oof_dropped"][si])
        t["dropped"] = rank01(d["test_dropped"][ti])

    # trees4 is a synthetic reference: the average the chain already forms
    for dd in (o, t):
        dd["trees4"] = (dd["lgb_full"] + dd["lgb_lean"]
                        + dd["cat_full"] + dd["xgb_full"]) / 4

    # weight each leg already carries inside base
    IN_BASE = {"lgb_full": (1 - TM) * (CW / (CW + 1)) * 0.25,
               "lgb_lean": (1 - TM) * (CW / (CW + 1)) * 0.25,
               "cat_full": (1 - TM) * (CW / (CW + 1)) * 0.25,
               "xgb_full": (1 - TM) * (CW / (CW + 1)) * 0.25,
               "lgbx_full": (1 - TM) * (1 / (CW + 1)),
               "linear": TM,
               "fac_full": 0.0,
               "dropped": 0.0,
               "trees4": (1 - TM) * (CW / (CW + 1))}

    def base_of(dd):
        trees4 = dd["trees4"]
        core = (trees4 * CW + dd["lgbx_full"]) / (CW + 1.0)
        return (1 - TM) * core + TM * dd["linear"]

    def chain(dd, r, ref="lgb_lean", tm=TM, cw=CW):
        trees4 = dd["trees4"]
        core = (trees4 * cw + dd["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * dd["linear"]
        return rank01(base + r * (base - dd[ref]))

    best_o, best_t = chain(o, R_BEST), chain(t, R_BEST)
    b_auc, b_ks = float(roc_auc_score(y, best_o)), ks(best_o, y)

    rr = np.array(sorted(MEASURED)); yy = np.array([MEASURED[k] for k in rr])
    c = np.polyfit(rr, yy, 2)
    vx = float(-c[1] / (2 * c[0]))
    print(f"r axis: curvature {c[0]:+.5f}, peak at r={vx:.4f} "
          f"value {np.poly1d(c)(vx):.6f}, measured best {MEASURED[0.75]:.5f}")
    print(f"remaining gain on the r axis: {np.poly1d(c)(vx) - MEASURED[0.75]:+.6f} "
          f"against an instrument sd of {INSTRUMENT_SD:.5f} -> axis closed\n")
    print(f"reference r={R_BEST}: OOF AUC {b_auc:.6f} KS {b_ks:.6f}, "
          f"public {MEASURED[0.75]:.5f}\n")

    cands, rows = {}, []

    def add(name, vo, vt, group, why):
        cands[name] = (vo, vt)
        rows.append({"name": name, "group": group, "why": why,
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - b_auc, 6),
                     "spearman_vs_r075": round(
                         float(spearmanr(vt, best_t).statistic), 6)})

    refs = [k for k in ("lgb_full", "cat_full", "xgb_full", "fac_full",
                        "lgbx_full", "linear", "trees4", "dropped") if k in o]
    for ref in refs:
        w = IN_BASE[ref] + abs(TARGET_NET)
        if not (0 < w < 0.95):
            print(f"  skip {ref}: matched w={w:.3f} out of range")
            continue
        r_m = w / (1 - w)
        add(f"ref_{ref}_matched", chain(o, r_m, ref), chain(t, r_m, ref),
            "reference at matched net strength",
            f"subtract {ref} at r={r_m:.4f} so its net coefficient is {TARGET_NET}")
        add(f"ref_{ref}_r075", chain(o, R_BEST, ref), chain(t, R_BEST, ref),
            "reference at fixed r=0.75",
            f"subtract {ref} at the same r as the measured optimum")

    for tm in (0.20, 0.40):
        add(f"tm{int(tm*100)}_r075",
            chain(o, R_BEST, tm=tm), chain(t, R_BEST, tm=tm),
            "linear weight", f"tm={tm} with r held at the measured optimum")
    for cw in (1.0, 3.0):
        add(f"core{int(cw)}to1_r075",
            chain(o, R_BEST, cw=cw), chain(t, R_BEST, cw=cw),
            "core weight", f"trees4:lgbx = {cw:.0f}:1 with r held at 0.75")

    rows.sort(key=lambda r: (r["group"], -r["spearman_vs_r075"]))
    print(f"{'name':26s} {'OOF AUC':>10} {'dAUC':>10} {'rho_vs_r075':>12}  group")
    for r in rows:
        print(f"{r['name']:26s} {r['oof_auc']:10.6f} {r['d_oof_auc']:+10.6f} "
              f"{r['spearman_vs_r075']:12.6f}  {r['group']}")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v78_*.csv")):
        f.unlink()
    for r in rows:
        p = OUT / f"submission_v78_{r['name']}.csv"
        frame = pd.DataFrame({"application_id": te.values,
                              "default_probability": rank01(cands[r["name"]][1])})
        frame.to_csv(p, index=False)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.default_probability.isna().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name

    order = [r["name"] for r in rows
             if r["group"] == "reference at matched net strength"][:3]
    print(f"\nwrote {len(rows)} candidates")
    print(f"submit order, one at a time: {order}")
    print("the reference axis goes first: it is the only structural freedom left "
          "after r closed, and the matched-strength framing makes the comparison "
          "about which leg to subtract rather than how hard.")

    (OUT / "summary_v78.json").write_text(json.dumps(
        {"r_axis": {"measured": MEASURED, "curvature": float(c[0]),
                    "peak_r": vx, "peak_value": float(np.poly1d(c)(vx)),
                    "remaining_gain": float(np.poly1d(c)(vx) - MEASURED[0.75]),
                    "closed": True},
         "instrument_sd": INSTRUMENT_SD, "target_net_coefficient": TARGET_NET,
         "weight_in_base": IN_BASE, "reference_oof_auc": b_auc,
         "reference_oof_ks": b_ks, "reference_public": MEASURED[0.75],
         "submit_order": order, "candidates": rows}, indent=2))
    print(f"saved {OUT / 'summary_v78.json'}")


if __name__ == "__main__":
    main()
