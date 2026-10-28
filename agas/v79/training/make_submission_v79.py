"""The last unmeasured axis: which legs are inside trees4.

Five axes of the chain have now been measured against public, and the result is
a sharp peak. One step away from `r=0.75, ref=lgb_lean, tm=0.30, core=2:1` on
v66 legs costs:

    r      -> 0.50                      -0.00096
    r      -> 1.00                      -0.00056
    tm     -> 0.20                      -0.00362
    ref    -> lgb_full                  -0.00586
    legs   -> retrained, +0.0008 AUC    -0.00361

Only `r` is shallow. Ten of the eleven post-`r` probes lost.

What has never been built, let alone measured, is the **membership of
`trees4`** - the four legs that get rank-averaged before `lgbx_full` is folded
in. v61 wrote `lgb_full, lgb_lean, cat_full, xgb_full` and no version since has
asked whether four is the right number or those the right four. That matters
more than it looks, because `lgb_lean` is simultaneously a `trees4` member and
the subtracted reference, so dropping a member changes what the subtraction is
measured against.

Candidates hold `r=0.75`, `tm=0.30`, `core=2:1` and `ref=lgb_lean` fixed and
change only the membership: drop one member at a time, or add `fac_full`, which
is the one leg trained on a third column set and currently used nowhere in the
chain. `lgb_lean` is never dropped, since it is the reference and removing it
would change two things at once.

No out-of-fold gate: on these axes out-of-fold is measured not to track public,
and the recipe that set the record had been ranked 7th of 9 by an out-of-fold
score.
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

R, TM, CW, REF = 0.75, 0.30, 2.0, "lgb_lean"
PUBLIC_BEST = 0.56520
BASE_MEMBERS = ("lgb_full", "lgb_lean", "cat_full", "xgb_full")


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

    def chain(d, members):
        trees = sum(d[m] for m in members) / len(members)
        core = (trees * CW + d["lgbx_full"]) / (CW + 1.0)
        tm30 = (1 - TM) * core + TM * d["linear"]
        return rank01(tm30 + R * (tm30 - d[REF]))

    best_o, best_t = chain(o, BASE_MEMBERS), chain(t, BASE_MEMBERS)
    b_auc, b_ks = float(roc_auc_score(y, best_o)), ks(best_o, y)

    scored = pd.read_csv(AGAS / "v76/output/submission_v76_02_r0.75.csv")
    scored = scored.set_index("application_id").loc[te].default_probability.to_numpy()
    assert int((rankdata(best_t) != rankdata(scored)).sum()) == 0, \
        "the baseline does not reproduce the scored file"
    print(f"baseline trees4 = {list(BASE_MEMBERS)} reproduces the scored file "
          f"exactly; public {PUBLIC_BEST:.5f}")
    print(f"  OOF AUC {b_auc:.6f} KS {b_ks:.6f}\n")

    variants = {}
    for drop in BASE_MEMBERS:
        if drop == REF:
            continue                      # the reference must stay a member
        members = tuple(m for m in BASE_MEMBERS if m != drop)
        variants[f"trees3_drop_{drop}"] = (members, f"drop {drop}, leaving three")
    variants["trees5_add_fac_full"] = (
        BASE_MEMBERS + ("fac_full",),
        "add fac_full, the only leg on a third column set and used nowhere else")

    rows = {}
    print(f"{'candidate':24s} {'members':>8} {'OOF AUC':>10} {'dAUC':>10} "
          f"{'dKS':>10} {'rho':>9}")
    for name, (members, why) in variants.items():
        vo, vt = chain(o, members), chain(t, members)
        rho = float(spearmanr(vt, best_t).statistic)
        rows[name] = {"name": name, "members": list(members), "why": why,
                      "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                      "oof_ks": round(ks(vo, y), 6),
                      "d_oof_auc": round(float(roc_auc_score(y, vo)) - b_auc, 6),
                      "d_oof_ks": round(ks(vo, y) - b_ks, 6),
                      "spearman_vs_best": round(rho, 6), "vec": vt}
        r = rows[name]
        print(f"{name:24s} {len(members):>8} {r['oof_auc']:10.6f} "
              f"{r['d_oof_auc']:+10.6f} {r['d_oof_ks']:+10.6f} {rho:9.6f}")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v79_*.csv")):
        f.unlink()
    for name, r in rows.items():
        p = OUT / f"submission_v79_{name}.csv"
        frame = pd.DataFrame({"application_id": te.values,
                              "default_probability": rank01(r.pop("vec"))})
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

    queue = ["agas/v78/output/submission_v78_tm40_r075.csv",
             "agas/v78/output/submission_v78_core3to1_r075.csv",
             f"agas/v79/output/{rows['trees5_add_fac_full']['file']}"]
    print(f"\nremaining submission queue, one at a time:")
    for i, q in enumerate(queue, 1):
        print(f"  {i}. {q}")
    print("\nstop rule: if none of these beats 0.56520, freeze "
          "submission_v76_02_r0.75.csv as final and stop probing.")

    (OUT / "summary_v79.json").write_text(json.dumps(
        {"held_fixed": {"r": R, "tm": TM, "core_weight": CW, "reference": REF,
                        "legs": "v66"},
         "baseline_public": PUBLIC_BEST, "baseline_oof_auc": b_auc,
         "baseline_oof_ks": b_ks,
         "axis_costs": {"r_to_0.50": -0.00096, "r_to_1.00": -0.00056,
                        "tm_to_0.20": -0.00362, "ref_to_lgb_full": -0.00586,
                        "legs_retrained": -0.00361},
         "candidates": list(rows.values()), "queue": queue}, indent=2))
    print(f"saved {OUT / 'summary_v79.json'}")


if __name__ == "__main__":
    main()
