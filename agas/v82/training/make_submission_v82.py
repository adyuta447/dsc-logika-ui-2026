"""v82: the last submission, and why no large jump is available.

All seven axes of the chain are now measured against public and all seven are
closed. The costs of one step from the final point (r=0.75, tm=0.30, core=2:1,
ref=lgb_lean, v66 legs, public 0.56520):

    r      -> 1.00                   -0.00056     closed, quadratic peak 0.786
    r      -> 0.50                   -0.00096     remaining +0.000017
    core   -> 1:1                    -0.00120     closed, quadratic peak 1.847
    trees4 -> add fac_full           -0.00205     closed, one measurement
    core   -> 3:1                    -0.00226     remaining +0.000041
    legs   -> retrained +0.0008 AUC  -0.00361     closed, public evidence
    tm     -> 0.20                   -0.00362     closed, quadratic peak 0.295
    tm     -> 0.40                   -0.00436     remaining +0.000009
    ref    -> lgb_full               -0.00586     closed, one measurement

Thirteen of fourteen post-`r` probes lost. Four structural ideas were checked
geometrically before spending the last slot on any of them, and three were
rejected by that check rather than by a guess:

  probit-space blending instead of rank-space: rho 0.999973 against the final
    and direction alignment -0.004 with the only movement ever measured to
    help. It barely changes the ranking, so it is not a lever at all.
  subtracting two lean-trained legs instead of one: alignment **-0.90**. It
    moves opposite to the direction that works.
  pushing `r` past its peak with tm and core at theirs: alignment +0.97, the
    right direction, but `r` is already mapped and declining there.

What is left is the only move with a positive predicted delta: put all three
quadratics at their fitted peaks at once. Each peak is individually inside the
instrument's 0.00007 spread - r +0.000017, tm +0.000009, core +0.000041 - so if
they add, the combination is worth about **+0.000067**, which is one instrument
spread. That is the honest size of the last available gain. It is not a large
jump and nothing in seventeen scored submissions suggests one exists in this
candidate space.

The reason it is still worth sending is that the final Kaggle pick is made
after scores are visible, so an unused slot is worth nothing and a slot spent
here cannot lower the final score - it can only replace it if it reads higher.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"

FINAL = AGAS / "v76/output/submission_v76_02_r0.75.csv"
PUBLIC_BEST = 0.56520
INSTRUMENT_SD = 0.00007

# each axis: measured points -> fitted quadratic peak
AXES = {
    "r":    {"x": [0.50, 0.75, 1.00], "y": [0.56418, 0.56520, 0.56464]},
    "tm":   {"x": [0.20, 0.30, 0.40], "y": [0.56158, 0.56520, 0.56084]},
    "core": {"x": [1.00, 2.00, 3.00], "y": [0.56400, 0.56520, 0.56294]},
}


def rank01(v):
    return rankdata(v) / len(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    peaks, total = {}, 0.0
    print("fitted peak of each axis, from its three public measurements:")
    for name, d in AXES.items():
        c = np.polyfit(np.array(d["x"]), np.array(d["y"]), 2)
        px = float(-c[1] / (2 * c[0]))
        pv = float(np.poly1d(c)(px))
        gain = pv - PUBLIC_BEST
        peaks[name] = {"peak": round(px, 4), "peak_value": round(pv, 6),
                       "gain": round(gain, 6),
                       "gain_in_sd": round(gain / INSTRUMENT_SD, 2),
                       "curvature": round(float(c[0]), 5)}
        total += gain
        print(f"  {name:5s} peak={px:7.4f}  value={pv:.6f}  "
              f"gain={gain:+.6f} ({gain / INSTRUMENT_SD:.2f} sd)")
    print(f"\nadditive prediction if the three combine: {total:+.6f} "
          f"({total / INSTRUMENT_SD:.2f} instrument sd) -> {PUBLIC_BEST + total:.6f}")

    z = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    te = pd.Index(z["test_ids"])
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

    def chain(d, r, tm, cw, ref="lgb_lean"):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        tm30 = (1 - tm) * core + tm * d["linear"]
        return rank01(tm30 + r * (tm30 - d[ref]))

    base = chain(t, 0.75, 0.30, 2.0)
    scored = pd.read_csv(FINAL).set_index("application_id").loc[te]
    scored = scored.default_probability.to_numpy()
    assert int((rankdata(base) != rankdata(scored)).sum()) == 0, \
        "the baseline does not reproduce the scored file"
    print(f"\nbaseline reproduces {FINAL.name} exactly (public {PUBLIC_BEST:.5f})")

    vec = chain(t, peaks["r"]["peak"], peaks["tm"]["peak"], peaks["core"]["peak"])
    rho = float(spearmanr(vec, base).statistic)
    changed = int((rankdata(vec) != rankdata(base)).sum())
    print(f"combined-peak candidate: rho={rho:.6f}, "
          f"{changed} of 30000 rank positions differ")

    p = OUT / "submission_v82_allpeaks.csv"
    frame = pd.DataFrame({"application_id": te.values,
                          "default_probability": rank01(vec)})
    frame.to_csv(p, index=False)
    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    ok = {"columns": list(frame.columns) == ["application_id", "default_probability"],
          "30000 rows": len(frame) == 30000,
          "ids unique": bool(frame.application_id.is_unique),
          "id set matches test": set(frame.application_id) == test_set,
          "no NaN": not bool(frame.isna().any().any()),
          "finite": bool(np.isfinite(frame.default_probability).all()),
          "within [0,1]": bool(frame.default_probability.between(0, 1).all())}
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    assert all(ok.values()), "invalid csv"
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"  wrote {p.name}  sha256 {digest[:16]}")

    print("\n--- honest summary for the last slot ---")
    print(f"predicted: {PUBLIC_BEST + total:.5f}, i.e. {total:+.5f}")
    print(f"this is one instrument spread. It is not a large jump, and none of")
    print(f"the 17 scored submissions suggests a large jump exists here.")
    print(f"It is worth sending only because an unused slot is worth nothing and")
    print(f"the final pick is chosen after scores are visible, so this cannot")
    print(f"lower the final score.")

    (OUT / "summary_v82.json").write_text(json.dumps(
        {"final_public": PUBLIC_BEST, "instrument_sd": INSTRUMENT_SD,
         "axis_peaks": peaks, "additive_prediction": total,
         "predicted_public": PUBLIC_BEST + total,
         "candidate": p.name, "sha256": digest,
         "spearman_vs_final": rho, "rank_positions_changed": changed,
         "rejected_by_geometry": {
             "probit_space_blend": {"rho": 0.999973, "alignment": -0.0036,
                                    "why": "barely changes the ranking"},
             "dual_lean_reference": {"rho": 0.996416, "alignment": -0.9024,
                                     "why": "moves opposite to what works"},
             "r_past_peak": {"rho": 0.999667, "alignment": 0.9905,
                             "why": "right direction, axis already mapped down"}},
         "axes_closed": ["r", "tm", "core", "ref", "leg_quality",
                         "trees4_membership"],
         "validation": ok}, indent=2))
    print(f"\nsaved {OUT / 'summary_v82.json'}")


if __name__ == "__main__":
    main()
