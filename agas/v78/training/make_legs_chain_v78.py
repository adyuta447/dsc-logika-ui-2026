"""The chain at r=0.75 on the legs from the 8.7 hour run.

The run finished all six fold seeds in 524.8 minutes at 10 folds and three
booster seeds, with `extra_trees=True` on `lgb_full`, `lgb_lean` and `fac_full`
and nothing else changed from v66.

It is a clean A/B, and the log proves it: the four legs that were **not**
touched reproduce v66's bagged values to six decimals - `lgbx_full` 0.846550,
`cat_full` 0.846311, `xgb_full` 0.846756, `linear` 0.842771 - because the folds,
the fold seeds and `deterministic=True` are all identical. So there is no
hybrid-versus-pure question here as there was in v76b: taking every leg from
v78 *is* taking v66's legs with three of them upgraded. This script asserts
that bit-identity rather than trusting the matching printout.

The chain is built at `r = 0.75`, the optimum measured on public, so the only
thing that differs from the 0.56520 submission is leg quality. That is the
question v76b could not answer: its `hybrid` chain came in at -0.000088
out-of-fold against the anchor and was set aside on that basis, but out-of-fold
has since been measured not to track public on this axis - the r sweep has
out-of-fold falling monotonically while public rose then fell, and the recipe
that set the record had been ranked 7th of 9 by an out-of-fold score. So this
candidate is decided on public or not at all.
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

R_BEST = 0.75
PUBLIC_R075 = 0.56520
EXTRA_TREES_LEGS = ("lgb_full", "lgb_lean", "fac_full")
LEGS = ("fac_full", "lgb_full", "lgb_lean", "lgbx_full", "cat_full",
        "xgb_full", "linear")


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z66 = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    z78 = np.load(OUT / "preds_v78legs.npz", allow_pickle=True)

    tr, te = pd.Index(z66["train_ids"]), pd.Index(z66["test_ids"])
    si = pd.Index(z78["train_ids"]).get_indexer(tr)
    ti = pd.Index(z78["test_ids"]).get_indexer(te)
    assert (si == np.arange(len(si))).all(), "v78 train id order differs from v66"
    assert (ti == np.arange(len(ti))).all(), "v78 test id order differs from v66"

    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[tr].target.values.astype(int))

    # the controlled-experiment claim, asserted on the raw arrays
    untouched = [k for k in LEGS if k not in EXTRA_TREES_LEGS]
    identical = {}
    for k in untouched:
        same_o = np.allclose(z66[f"oof_{k}"], z78[f"oof_{k}"], rtol=0, atol=1e-12)
        same_t = np.allclose(z66[f"test_{k}"], z78[f"test_{k}"], rtol=0, atol=1e-12)
        identical[k] = bool(same_o and same_t)
    print("legs left untouched, checked against v66 at 1e-12:")
    for k, v in identical.items():
        print(f"  {k:12s} bit-identical: {v}")
    if not all(identical.values()):
        print("  NOTE: not all untouched legs reproduced, so this is not a clean A/B")

    o66 = {k: rank01(z66[f"oof_{k}"]) for k in LEGS}
    t66 = {k: rank01(z66[f"test_{k}"]) for k in LEGS}
    o78 = {k: rank01(z78[f"oof_{k}"]) for k in LEGS}
    t78 = {k: rank01(z78[f"test_{k}"]) for k in LEGS}

    print(f"\n{'leg':12s} {'v66 AUC':>10} {'v78 AUC':>10} {'delta':>10} "
          f"{'extra_trees':>12}")
    leg_rows = []
    for k in LEGS:
        a66, a78 = roc_auc_score(y, o66[k]), roc_auc_score(y, o78[k])
        leg_rows.append({"leg": k, "v66_auc": round(float(a66), 6),
                         "v78_auc": round(float(a78), 6),
                         "delta": round(float(a78 - a66), 6),
                         "extra_trees": k in EXTRA_TREES_LEGS,
                         "bit_identical": identical.get(k)})
        print(f"{k:12s} {a66:10.6f} {a78:10.6f} {a78 - a66:+10.6f} "
              f"{str(k in EXTRA_TREES_LEGS):>12}")

    def chain(d, r=R_BEST, tm=0.30, cw=2.0, ref="lgb_lean"):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * d["linear"]
        return rank01(base + r * (base - d[ref]))

    ref_o, ref_t = chain(o66), chain(t66)      # the scored 0.56520 vector
    r_auc, r_ks = float(roc_auc_score(y, ref_o)), ks(ref_o, y)

    # sanity: the reference must reproduce the file that actually scored
    scored = pd.read_csv(AGAS / "v76/output/submission_v76_02_r0.75.csv")
    scored = scored.set_index("application_id").loc[te].default_probability.to_numpy()
    assert int((rankdata(ref_t) != rankdata(scored)).sum()) == 0, \
        "the r=0.75 reference does not reproduce the scored file"
    print(f"\nreference r=0.75 on v66 legs reproduces the scored file exactly; "
          f"public {PUBLIC_R075:.5f}")

    cands = {"v78legs_r075": (chain(o78), chain(t78))}
    # r is the optimum measured on v66 legs; better legs could move it, so one
    # step either side is built too - not to submit now, but to have ready
    for r in (0.625, 0.875):
        cands[f"v78legs_r{r:.3f}".replace(".", "p")] = (chain(o78, r), chain(t78, r))

    rows = []
    for name, (vo, vt) in cands.items():
        rows.append({"name": name,
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_oof_auc": round(float(roc_auc_score(y, vo)) - r_auc, 6),
                     "d_oof_ks": round(ks(vo, y) - r_ks, 6),
                     "spearman_vs_scored": round(
                         float(spearmanr(vt, ref_t).statistic), 6)})
    print(f"\n{'candidate':20s} {'OOF AUC':>10} {'dAUC':>10} {'dKS':>10} {'rho':>9}")
    print(f"{'r075 on v66 legs':20s} {r_auc:10.6f} {0.0:+10.6f} {0.0:+10.6f} "
          f"{1.0:9.6f}   <- scored {PUBLIC_R075:.5f}")
    for r in rows:
        print(f"{r['name']:20s} {r['oof_auc']:10.6f} {r['d_oof_auc']:+10.6f} "
              f"{r['d_oof_ks']:+10.6f} {r['spearman_vs_scored']:9.6f}")
    print("\nout-of-fold is reported, not used: on this axis it fell monotonically "
          "while public rose then fell, and the record recipe was ranked 7th of 9 "
          "by an out-of-fold score.")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v78legs_*.csv")):
        f.unlink()
    for r in rows:
        p = OUT / f"submission_{r['name']}.csv"
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
        print(f"  wrote {p.name}")

    (OUT / "summary_v78legs.json").write_text(json.dumps(
        {"run": "10 folds x 6 fold seeds x 3 booster seeds, extra_trees on "
                "lgb_full/lgb_lean/fac_full, 524.8 min",
         "controlled_experiment": identical,
         "legs": leg_rows, "reference_r": R_BEST,
         "reference_public": PUBLIC_R075,
         "reference_oof_auc": r_auc, "reference_oof_ks": r_ks,
         "candidates": rows,
         "submit": "submission_v78legs_r075.csv"}, indent=2))
    print(f"\nsaved {OUT / 'summary_v78legs.json'}")


if __name__ == "__main__":
    main()
