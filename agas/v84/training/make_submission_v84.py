"""v84: branch candidates on the slot-2 identity axis, built before any slot is spent.

    second = (1 - a) * lgbx_full + a * LEG        a = 0 is the scored final

v83 built a = 1 for every leg. This adds the step past it (a = 1.5) for the two
legs whose geometry favours them, and the cat/xgb average for the branch where
both a = 1 points lose. Usage: python3 make_submission_v84.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"
FINAL = AGAS / "v76/output/submission_v76_02_r0.75.csv"


def rank01(v):
    return rankdata(v) / len(v)


def chain(d, r=0.75, tm=0.30, cw=2.0, ref="lgb_lean", leg="lgbx_full", a=0.0):
    trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
    second = (1 - a) * d["lgbx_full"] + a * d[leg]
    core = (trees4 * cw + second) / (cw + 1.0)
    tm30 = (1 - tm) * core + tm * d["linear"]
    return rank01(tm30 + r * (tm30 - d[ref]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    te = pd.Index(z["test_ids"])
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}
    t["catxgb"] = (t["cat_full"] + t["xgb_full"]) / 2

    final = chain(t)
    scored = pd.read_csv(FINAL).set_index("application_id").loc[te].default_probability
    assert int((rankdata(final) != rankdata(scored.to_numpy())).sum()) == 0
    # v83 a=1 files must be what this chain gives, so the axis is one axis
    for leg in ("cat_full", "xgb_full"):
        v83 = pd.read_csv(AGAS / f"v83/output/submission_v83_slot2_{leg}.csv")
        v83 = v83.set_index("application_id").loc[te].default_probability.to_numpy()
        assert np.allclose(v83, chain(t, leg=leg, a=1.0)), leg

    good = chain(t) - chain(t, r=0.50)
    bad = chain(t, r=0.7864, tm=0.2954, cw=1.8468) - final

    cands = {"slot2_cat_a1p5": chain(t, leg="cat_full", a=1.5),
             "slot2_xgb_a1p5": chain(t, leg="xgb_full", a=1.5),
             "slot2_catxgb": chain(t, leg="catxgb", a=1.0)}
    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    rows = []
    print(f"{'candidate':18s} {'rho':>9} {'GOOD':>7} {'BAD':>7}")
    for name, v in cands.items():
        p = OUT / f"submission_v84_{name}.csv"
        frame = pd.DataFrame({"application_id": te.values, "default_probability": v})
        frame.to_csv(p, index=False)
        assert (list(frame.columns) == ["application_id", "default_probability"]
                and len(frame) == 30000 and frame.application_id.is_unique
                and set(frame.application_id) == test_set
                and not frame.isna().any().any()
                and frame.default_probability.between(0, 1).all()), p
        row = {"name": name, "file": p.name,
               "spearman_vs_final": round(float(spearmanr(v, final).statistic), 6),
               "align_good": round(float(pearsonr(v - final, good).statistic), 3),
               "align_bad": round(float(pearsonr(v - final, bad).statistic), 3),
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
        rows.append(row)
        print(f"{name:18s} {row['spearman_vs_final']:9.6f} "
              f"{row['align_good']:+7.3f} {row['align_bad']:+7.3f}")
    (OUT / "summary_v84.json").write_text(json.dumps(
        {"final": FINAL.name, "final_public": 0.56524, "candidates": rows}, indent=2))
    print(f"wrote {len(rows)} validated candidates to {OUT}")


if __name__ == "__main__":
    main()
