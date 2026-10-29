"""v83: the one slot in the chain whose occupant was never questioned.

`submission_v82_allpeaks.csv` scored 0.56481, **down** 0.00039, against an
additive prediction of +0.000066. So the three per-axis quadratics do not
compose, and the interaction is about -0.00046, six instrument spreads - real,
not noise.

Decomposing that failure against the two directions public has actually
measured - `GOOD` = the r 0.5 -> 0.75 move that gained +0.00102, and `BAD` =
the allpeaks move that lost 0.00039 - shows exactly which coordinate did the
damage:

    coordinate moved alone      alignment with GOOD   alignment with BAD
    r    -> 0.786                     +0.977                +0.517
    tm   -> 0.295                     -0.772                -0.030
    core -> 1.847                     -0.185                +0.569

Only `r` moves the way the one measured gain moves. The fitted `tm` and `core`
peaks point the other way: they fit their own three points and still move
against the direction that works, which is what a peak located inside the
instrument's own spread looks like when it is an artefact rather than a signal.

That closes parameter tuning. What it leaves is a **slot identity** that
twenty-three versions never questioned. The chain is

    core = (2 * trees4 + lgbx_full) / 3

so `lgbx_full` carries 1/3 of `core` on its own - twice the weight of any
single `trees4` member, which gets 2/3 * 1/4 = 1/6. v61 put `lgbx_full` in that
slot and every version since has tuned its *weight* (the core axis: 1:1, 2:1,
3:1, all measured) without ever asking whether it is the right *leg*. That is
the same kind of question as the reference axis, on a different slot, and the
reference axis is the only slot question ever asked.

Geometry of the alternatives, before spending the slot:

    second slot              rho        GOOD      BAD
    cat_full               0.998157    +0.337    -0.205
    cat_full+lgbx mix      0.999946    -0.022    -0.453
    fac_full               0.999787    -0.021    -0.455
    lgb_full               0.999851    -0.041    -0.595

`cat_full` is the only one with positive alignment to the direction that gained
and negative alignment to the direction that lost, and it is also the largest
step of the four, so it carries the most upside. Everything with rho above
0.9999 sits inside the +/-0.0001 band that eighteen measurements have
established as this chain's noise, and cannot produce a higher score whatever
its sign.

**Honest expected value: slightly negative.** Eighteen scored submissions, seven
closed axes, and the one composition attempt all say the final point is a sharp
peak. This is a gamble taken because the final Kaggle pick is made after scores
are visible, so an unused slot is worth nothing and a spent one cannot lower the
result.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"

FINAL = AGAS / "v76/output/submission_v76_02_r0.75.csv"
PUBLIC_BEST = 0.56520
INSTRUMENT_SD = 0.00007


def rank01(v):
    return rankdata(v) / len(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    te = pd.Index(z["test_ids"])
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

    def chain(d, r=0.75, tm=0.30, cw=2.0, ref="lgb_lean", second="lgbx_full"):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d[second]) / (cw + 1.0)
        tm30 = (1 - tm) * core + tm * d["linear"]
        return rank01(tm30 + r * (tm30 - d[ref]))

    final = chain(t)
    scored = pd.read_csv(FINAL).set_index("application_id").loc[te]
    assert int((rankdata(final) !=
                rankdata(scored.default_probability.to_numpy())).sum()) == 0, \
        "the baseline does not reproduce the scored file"
    print(f"baseline reproduces {FINAL.name} exactly (public {PUBLIC_BEST:.5f})\n")

    good = chain(t, r=0.75) - chain(t, r=0.50)                 # gained +0.00102
    bad = chain(t, r=0.7864, tm=0.2954, cw=1.8468) - final      # lost  -0.00039

    # weight the second slot carries on its own, for the record
    cw = 2.0
    print(f"weight of the second core slot      : {1 / (cw + 1):.5f}")
    print(f"weight of one trees4 member         : {(cw / (cw + 1)) / 4:.5f}")
    print("so the occupant of that slot carries twice any single tree leg\n")

    cands = {}
    for name, second in [("cat_full", "cat_full"), ("fac_full", "fac_full"),
                         ("lgb_full", "lgb_full"), ("xgb_full", "xgb_full")]:
        cands[f"slot2_{name}"] = chain(t, second=second)
    d2 = dict(t)
    d2["mix_lgbx_fac"] = 0.5 * t["lgbx_full"] + 0.5 * t["fac_full"]
    cands["slot2_lgbx_fac_mix"] = chain(d2, second="mix_lgbx_fac")
    cands["r0p786_only"] = chain(t, r=0.7864)

    rows = []
    print(f"{'candidate':22s} {'rho':>9} {'GOOD':>8} {'BAD':>8} {'rank chg':>9}")
    for name, v in cands.items():
        rho = float(spearmanr(v, final).statistic)
        g = float(pearsonr(v - final, good).statistic)
        b = float(pearsonr(v - final, bad).statistic)
        rows.append({"name": name, "spearman_vs_final": round(rho, 6),
                     "align_good": round(g, 3), "align_bad": round(b, 3),
                     "rank_changed": int((rankdata(v) != rankdata(final)).sum()),
                     "inside_noise_band": bool(rho > 0.9999)})
        print(f"{name:22s} {rho:9.6f} {g:+8.3f} {b:+8.3f} "
              f"{rows[-1]['rank_changed']:9d}")

    pick = "slot2_cat_full"
    print(f"\nrecommended: {pick}")
    print("  the only alternative with positive alignment to the direction that")
    print("  gained and negative alignment to the one that lost, and the largest")
    print("  step of the four, so the most upside. Expected value is still")
    print("  slightly negative; the downside is truncated by the final pick.")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v83_*.csv")):
        f.unlink()
    for r in rows:
        p = OUT / f"submission_v83_{r['name']}.csv"
        frame = pd.DataFrame({"application_id": te.values,
                              "default_probability": rank01(cands[r["name"]])})
        frame.to_csv(p, index=False)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.isna().any().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name
        r["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"\nwrote {len(rows)} candidates, all validated")
    for r in rows:
        print(f"  {r['file']}  sha256 {r['sha256']}")

    (OUT / "summary_v83.json").write_text(json.dumps(
        {"final_public": PUBLIC_BEST, "instrument_sd": INSTRUMENT_SD,
         "allpeaks_result": {"public": 0.56481, "predicted": 0.565266,
                             "delta": -0.00039,
                             "finding": "the three per-axis quadratics do not "
                                        "compose; interaction about -0.00046"},
         "decomposition": {"r_0.786_alone": {"align_good": 0.977, "align_bad": 0.517},
                           "tm_0.295_alone": {"align_good": -0.772, "align_bad": -0.030},
                           "core_1.847_alone": {"align_good": -0.185, "align_bad": 0.569}},
         "second_slot_weight": 1 / (cw + 1),
         "trees4_member_weight": (cw / (cw + 1)) / 4,
         "candidates": rows, "recommended": pick,
         "expected_value": "slightly negative; downside truncated by the final pick",
         "axes_closed": ["r", "tm", "core", "ref", "leg_quality",
                         "trees4_membership", "axis_composition"]}, indent=2))
    print(f"\nsaved {OUT / 'summary_v83.json'}")


if __name__ == "__main__":
    main()
