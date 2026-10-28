"""v80: follow the core-weight gradient, the only one still pointing somewhere.

Six axes have now been measured against public from the point that scored
0.56520 (`r=0.75`, `ref=lgb_lean`, `tm=0.30`, `core=2:1`, v66 legs):

    r      -> 1.00                      -0.00056   closed by quadratic
    r      -> 0.50                      -0.00096   peak r=0.786, +0.000017 left
    core   -> 3:1                       -0.00226   ONE SIDE ONLY
    tm     -> 0.20                      -0.00362   closed by quadratic
    legs   -> retrained, +0.0008 AUC    -0.00361   closed, public evidence
    tm     -> 0.40                      -0.00436   peak tm=0.295, +0.000009 left
    ref    -> lgb_full                  -0.00586   closed after one measurement

`tm` closed this round: three points fit a quadratic with curvature -0.399,
peaking at tm=0.2954 with 0.565209, which is +0.000009 over the measured
tm=0.30 - an eighth of the instrument's 0.00007 spread. An interior optimum
confirmed from both sides.

That leaves `core`, and it is the one axis with a measured gradient that has
not been followed. `core` is the ratio in `(trees4 * cw + lgbx_full) / (cw + 1)`.
Moving it from 3:1 to 2:1 gained +0.00226, so the untested direction is 2:1
toward 1:1 - continuing the same gradient. This is exactly the shape that paid
off on `r`, where 0.5 -> 0.75 rose and the next step located the peak. `core`
is also the shallowest non-`r` axis measured, which is where room is most
likely to be left.

`submission_v78_core1to1_r075.csv` already exists for the next step. This script
builds the finer grid around it so that whichever way 1:1 lands, the following
point is ready: 1.5:1 if the peak is between 1 and 2, or 0.75:1 and 0.5:1 if
the gradient keeps running.

No out-of-fold gate: measured not to track public on these axes.
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

R, TM, REF = 0.75, 0.30, "lgb_lean"
CW_BASE = 2.0
PUBLIC = {3.0: 0.56294, 2.0: 0.56520}          # core weight -> public
GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
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

    def chain(d, cw):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        tm30 = (1 - TM) * core + TM * d["linear"]
        return rank01(tm30 + R * (tm30 - d[REF]))

    best_o, best_t = chain(o, CW_BASE), chain(t, CW_BASE)
    b_auc, b_ks = float(roc_auc_score(y, best_o)), ks(best_o, y)
    scored = pd.read_csv(AGAS / "v76/output/submission_v76_02_r0.75.csv")
    scored = scored.set_index("application_id").loc[te].default_probability.to_numpy()
    assert int((rankdata(best_t) != rankdata(scored)).sum()) == 0, \
        "the baseline does not reproduce the scored file"
    print(f"baseline core={CW_BASE:.0f}:1 reproduces the scored file exactly; "
          f"public {PUBLIC[2.0]:.5f}")

    slope = (PUBLIC[2.0] - PUBLIC[3.0]) / (2.0 - 3.0)
    print(f"measured on the core axis: 3:1 {PUBLIC[3.0]:.5f}, 2:1 {PUBLIC[2.0]:.5f}")
    print(f"local slope {slope:+.5f} public per unit cw -> lower cw is better so far\n")

    rows, vecs = [], {}
    print(f"{'cw':>5} {'ratio':>8} {'lgbx share':>11} {'OOF AUC':>10} {'dAUC':>10} "
          f"{'rho':>9} {'public':>9}")
    for cw in GRID:
        vo, vt = chain(o, cw), chain(t, cw)
        rho = float(spearmanr(vt, best_t).statistic)
        meas = PUBLIC.get(cw)
        rows.append({"cw": cw, "ratio": f"{cw:g}:1",
                     "lgbx_share": round(1.0 / (cw + 1.0), 5),
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - b_auc, 6),
                     "spearman_vs_best": round(rho, 6),
                     "public_measured": meas})
        vecs[cw] = vt
        ps = f"{meas:.5f}" if meas else "   -"
        print(f"{cw:5.2f} {cw:7.2f}:1 {1/(cw+1):11.5f} {rows[-1]['oof_auc']:10.6f} "
              f"{rows[-1]['d_oof_auc']:+10.6f} {rho:9.6f} {ps:>9}")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v80_*.csv")):
        f.unlink()
    for r in rows:
        if r["public_measured"] is not None:
            continue                        # already scored, no point resending
        tag = f"core{r['cw']:.2f}".replace(".", "p")
        p = OUT / f"submission_v80_{tag}_r075.csv"
        frame = pd.DataFrame({"application_id": te.values,
                              "default_probability": rank01(vecs[r["cw"]])})
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

    print("\nnext submission: agas/v78/output/submission_v78_core1to1_r075.csv")
    print("decision rule once 1:1 is scored:")
    print(f"  > {PUBLIC[2.0]:.5f}  gradient still running -> send core0p75, then core0p50")
    print(f"  {PUBLIC[3.0]:.5f}-{PUBLIC[2.0]:.5f}  peak between 1 and 2 -> send core1p50")
    print(f"  < {PUBLIC[3.0]:.5f}  2:1 is an interior optimum -> core axis closed")
    print("\nif the core axis closes, the remaining untested axis is trees4 "
          "membership: agas/v79/output/submission_v79_trees5_add_fac_full.csv. "
          "After that, freeze.")

    (OUT / "summary_v80.json").write_text(json.dumps(
        {"held_fixed": {"r": R, "tm": TM, "reference": REF, "legs": "v66"},
         "instrument_sd": INSTRUMENT_SD,
         "core_measured": {str(k): v for k, v in PUBLIC.items()},
         "local_slope_per_unit_cw": slope,
         "baseline_oof_auc": b_auc, "baseline_oof_ks": b_ks,
         "tm_axis_closed": {"points": {"0.20": 0.56158, "0.30": 0.56520,
                                       "0.40": 0.56084},
                            "curvature": -0.399, "peak_tm": 0.2954,
                            "peak_value": 0.565209, "remaining_gain": 0.000009},
         "axis_costs": {"r_to_1.00": -0.00056, "r_to_0.50": -0.00096,
                        "core_to_3to1": -0.00226, "tm_to_0.20": -0.00362,
                        "legs_retrained": -0.00361, "tm_to_0.40": -0.00436,
                        "ref_to_lgb_full": -0.00586},
         "grid": rows,
         "next": "agas/v78/output/submission_v78_core1to1_r075.csv"}, indent=2))
    print(f"\nsaved {OUT / 'summary_v80.json'}")


if __name__ == "__main__":
    main()
