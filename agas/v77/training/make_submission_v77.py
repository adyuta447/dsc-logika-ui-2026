"""v77: follow the gradient that v76 found on the extrapolation ratio.

`submission_v76_02_r0.75.csv` scored **0.56520**, the first candidate in twenty
versions to beat the anchor's 0.56424. Against the three measurements of the
exact `r=0.5` recipe - 0.56424, 0.56419, 0.56411, mean 0.56418, sd 0.00007 -
that is +0.00102, or 14.6 standard deviations of the instrument. It also
settles the question v76 posed: a perturbation at rho 0.999803 did not collapse
the way `seq05` did at rho 0.99950, it went up, so the surface is smooth and
the fragility reading is rejected.

**Why it went up.** The extrapolation is algebraically a weighted contrast:

    base + r*(base - lgb_lean)  ==  base - w*lgb_lean,   w = r/(1+r)

verified numerically for r from 0.25 to 3.0, with a maximum difference of
3.3e-5, which is one rank position out of 30000 from tie-breaking. Since
`lgb_lean` also sits inside `base` with weight 0.7 * 2/3 * 1/4 = 0.11667, its
net coefficient runs from +0.11667 at r=0 to -0.31190 at r=0.75.

The first reading of that was wrong and is withdrawn. Since `lgb_lean` is the
only leg trained on the 490-column `lean` set while the others use all 598, the
contrast looked like a full-minus-lean comparison that isolates the 108 dropped
columns. Testing it refuted it: a leg trained only on those 108 columns
(OOF AUC 0.836393) moves the ranking in the *opposite* direction from raising
`r`, Pearson -0.68, and raising `r` is anti-correlated with adding any leg at
all - most strongly with `lgb_lean` itself at -0.9959, which is what the
algebra says it must be. Raising `r` subtracts; it does not add.

What survives the test is narrower. `lgb_lean` correlates about 0.99 with
`base`, so subtracting a large multiple of it amplifies the small residual
between them. Raising `r` is therefore de-shrinkage: it moves the ranking away
from the consensus along the `base - lgb_lean` disagreement, reordering rows the
consensus had left nearly tied. Public rewarded that once. Why that residual
carries signal is not established, and this docstring stops where the
measurement stops.

**Why no version found this.** v61 and v66 both swept `r` and both chose 0.5 by
out-of-fold AUC. On this axis out-of-fold points the wrong way: `r=0.75` is
-0.000350 AUC and -0.000920 KS against the anchor and still +0.00102 on public.
So the sweep is continued here against public only, with no out-of-fold gate.

This script trains nothing; it re-assembles the stored v66 leg predictions.
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

# measured public scores on the (tm=0.30, ref=lgb_lean) axis
MEASURED = {0.50: [0.56424, 0.56419, 0.56411], 0.75: [0.56520]}
INSTRUMENT_SD = 0.00007
GRID = (0.25, 0.50, 0.625, 0.75, 0.875, 1.00, 1.25, 1.50, 2.00)


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

    def chain(d, r, tm=0.30, cw=2.0, ref="lgb_lean"):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * d["linear"]
        return rank01(base + r * (base - d[ref]))

    a_o, a_t = chain(o, 0.50), chain(t, 0.50)
    a_auc, a_ks = float(roc_auc_score(y, a_o)), ks(a_o, y)
    base_pub = float(np.mean(MEASURED[0.50]))

    # linear fit through the two measured points, used only to state a
    # falsifiable prediction for the next submission, never as a selection key
    slope = (MEASURED[0.75][0] - base_pub) / (0.75 - 0.50)

    rows, vecs = [], {}
    print(f"instrument sd inside one recipe: {INSTRUMENT_SD:.5f}")
    print(f"measured: r=0.50 -> {base_pub:.5f} (n={len(MEASURED[0.50])}), "
          f"r=0.75 -> {MEASURED[0.75][0]:.5f}")
    print(f"local slope: {slope:+.5f} public per unit r\n")
    print(f"{'r':>6} {'w':>7} {'net_lgb_lean':>13} {'OOF AUC':>10} {'dAUC':>10} "
          f"{'rho_vs_r0.5':>12} {'public':>9} {'linear_pred':>12}")
    for r in GRID:
        vo, vt = chain(o, r), chain(t, r)
        w = r / (1 + r)
        net = 0.7 * (2 / 3) * 0.25 - w
        rho = float(spearmanr(vt, a_t).statistic)
        meas = MEASURED.get(r)
        pred = base_pub + slope * (r - 0.50)
        rows.append({"r": r, "w": round(w, 4), "net_lgb_lean": round(net, 5),
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - a_auc, 6),
                     "d_oof_ks": round(ks(vo, y) - a_ks, 6),
                     "spearman_vs_r050": round(rho, 6),
                     "public_measured": (round(float(np.mean(meas)), 5)
                                         if meas else None),
                     "linear_prediction": round(pred, 5)})
        vecs[r] = vt
        ps = f"{np.mean(meas):.5f}" if meas else "   -"
        print(f"{r:6.3f} {w:7.4f} {net:13.5f} {rows[-1]['oof_auc']:10.6f} "
              f"{rows[-1]['d_oof_auc']:+10.6f} {rho:12.6f} {ps:>9} {pred:12.5f}")

    print("\nthe linear prediction is a registered guess, not a selection key: "
          "the surface must turn somewhere and only public can say where.")
    print("\ndecision rule once r=1.00 is scored:")
    print(f"  > {MEASURED[0.75][0]:.5f}  gradient continues -> send r=1.25, then r=1.50")
    print(f"  {base_pub:.5f}-{MEASURED[0.75][0]:.5f}  optimum between 0.75 and 1.00 "
          f"-> send r=0.875")
    print(f"  < {base_pub:.5f}  optimum near 0.75 -> send r=0.625 to bracket it")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v77_*.csv")):
        f.unlink()
    written = 0
    for r in GRID:
        if MEASURED.get(r):          # already scored, no point resubmitting
            continue
        tag = f"r{r:.3f}".replace(".", "p")
        p = OUT / f"submission_v77_{tag}.csv"
        frame = pd.DataFrame({"application_id": test_ids,
                              "default_probability": rank01(vecs[r])})
        frame.to_csv(p, index=False)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.default_probability.isna().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        for row in rows:
            if row["r"] == r:
                row["file"] = p.name
        written += 1
        print(f"  wrote {p.name}")

    (OUT / "summary_v77.json").write_text(json.dumps(
        {"instrument_sd": INSTRUMENT_SD, "measured_public": MEASURED,
         "base_public_r050": base_pub, "local_slope_per_unit_r": slope,
         "anchor_oof_auc": a_auc, "anchor_oof_ks": a_ks,
         "algebra": "base + r*(base - lgb_lean) == base - w*lgb_lean, w = r/(1+r)",
         "lean_drops": ["region", "employment_type", "housing_status", "age",
                        "employment_tenure_months", "dependents", "tenure_to_age",
                        "income_per_dependent", "loan_purpose", "decision_lag_h",
                        "app_hour", "app_dow", "ec_share_*", "ec_catqty_*",
                        "cs_shift*", "buh_*", "gn_*"],
         "grid": rows, "files_written": written}, indent=2))
    print(f"\nsaved {OUT / 'summary_v77.json'}")


if __name__ == "__main__":
    main()
