"""Final v66 candidate builder.

The blend formula is not new. It is v61's, verified before any new training: the
v61 chain

    trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
    core   = (2 * trees4 + lgbx_full) / 3
    tm20   = 0.8 * core + 0.2 * linear
    tm30   = 0.7 * core + 0.3 * linear
    alt    = rank01(tm30 + r * (tm30 - lgb_lean))

rebuilt from the stored v61 predictions gives KS 0.5286237, matching
`summary_v61_final.json` to seven decimals. So the object being extrapolated
here is the same object that scored 0.56312, only trained on six fold seeds and
five LightGBM seeds instead of three and three.

What is new is only the neighbourhood. v61 searched this axis at two ratios
(0.5 and 1.0) using `tm20` and `tm30` against `fac_full` and `lgb_lean`. Here
the same ratios are searched against the same legs, plus a consensus across
neighbouring ratios, because the ratio plateau measured in v61 is broad and a
single ratio is the one choice most likely to be a noise draw.

Candidate selection does not use KS as the key, for the reason established in
v64: KS is a maximum statistic with a sampling spread of about 0.0036 at OOF
size, so the argmax of a few hundred vectors measures noise. AUC is the key, KS
is reported with its spread subtracted, and vectors that rank almost identically
are dropped.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# This script sits in agas/v66/, so HERE is the version folder and the repo root
# is two levels up: HERE -> agas -> repo. Getting this off by one is the same
# class of bug that hit check_stability_selection.py and make_submission_v64.py,
# so the path is asserted rather than trusted.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "agas/shared/paths.py").exists(), f"repo root wrong: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE / "output"
ANCHOR = paths.version_output_of("v61") / "submission_v61_alt_ext0.5_tm30_lean.csv"
SIGMA_OOF, SIGMA_PUB = 0.0036, 0.0053
# Deduplication is deliberately loose here. v61's winning file sits at Spearman
# 0.9859 against its own anchor and scored 0.56312, so candidates correlated at
# 0.998 with each other are still meaningfully different submissions. A tight
# threshold first pass over this pool threw away the v61-equivalent candidate
# itself, because it ranked below two near-ties on AUC.
TAU = 0.9995
FLOOR = 0.0025

# The candidate the run exists to produce: v61's exact recipe on better-trained
# legs. It is force-kept so a dedup pass can never drop it.
MUST_KEEP = "ext0.5|tm30|lgb_lean"


def rank01(v):
    return rankdata(v) / len(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "preds_v66.npz", allow_pickle=True)
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv",
                    usecols=["application_id", "target"]).set_index(
        "application_id").loc[train_ids].target.values.astype(int)

    ro = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    rt = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

    trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o, core_t = (trees_o * 2 + ro["lgbx_full"]) / 3, (trees_t * 2 + rt["lgbx_full"]) / 3
    tm20_o, tm20_t = 0.8 * core_o + 0.2 * ro["linear"], 0.8 * core_t + 0.2 * rt["linear"]
    tm30_o, tm30_t = 0.7 * core_o + 0.3 * ro["linear"], 0.7 * core_t + 0.3 * rt["linear"]

    def ext(lo, lt, mo, mt, r):
        return rank01(lo + r * (lo - mo)), rank01(lt + r * (lt - mt))

    cands = {}
    for k in ro:
        cands[f"leg|{k}"] = (ro[k], rt[k])
    cands["trees4"] = (trees_o, trees_t)
    cands["core"] = (core_o, core_t)
    cands["tm20"], cands["tm30"] = (tm20_o, tm20_t), (tm30_o, tm30_t)

    for base, bo, bt in (("tm20", tm20_o, tm20_t), ("tm30", tm30_o, tm30_t)):
        for leg in ("lgb_lean", "fac_full"):
            for r in (0.25, 0.5, 0.75, 1.0, 1.5):
                cands[f"ext{r}|{base}|{leg}"] = ext(bo, bt, ro[leg], rt[leg], r)

    # the plateau, averaged in rank space, so no single ratio is load-bearing
    for leg in ("lgb_lean", "fac_full"):
        for base, bo, bt in (("tm20", tm20_o, tm20_t), ("tm30", tm30_o, tm30_t)):
            acc_o, acc_t = 0.0, 0.0
            for r in (0.25, 0.5, 0.75):
                acc_o += ext(bo, bt, ro[leg], rt[leg], r)[0]
                acc_t += ext(bo, bt, ro[leg], rt[leg], r)[1]
            cands[f"plateau|{base}|{leg}"] = (rank01(acc_o), rank01(acc_t))

    scored = []
    for name, (o, t) in cands.items():
        scored.append({"name": name, "oof": o, "test": t,
                       "auc": float(roc_auc_score(y, o)),
                       "ks": float(ks_2samp(o[y == 1], o[y == 0]).statistic)})
    scored.sort(key=lambda r: -r["auc"])
    best = scored[0]["auc"]
    eligible = [r for r in scored if r["auc"] >= best - FLOOR]
    print(f"  pool {len(scored)}, AUC floor {FLOOR} below {best:.6f}: {len(eligible)}")

    anchor_rank = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                         .loc[test_ids].default_probability.values)

    kept = [r for r in eligible if r["name"] == MUST_KEEP]
    dropped = [r for r in eligible if r["name"] == MUST_KEEP]
    for r in eligible:
        if r in dropped:
            continue
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -r["auc"])
    if MUST_KEEP not in [r["name"] for r in eligible]:
        print(f"  warning: {MUST_KEEP} fell outside the AUC floor and was not in the pool")
    print(f"  after deduplicating at rank correlation {TAU}: {len(kept)}")

    stale = sorted(OUT.glob("submission_v66_*.csv"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"  removed {len(stale)} stale file(s) from an earlier run")

    rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v66_{i:02d}_{k['name'].replace('|', '_')}.csv"
        d = pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                     "ks": round(k["ks"], 6),
                     "ks_minus_sigma": round(k["ks"] - SIGMA_OOF, 6),
                     "spearman_v61_anchor": round(
                         float(spearmanr(k["test"], anchor_rank).statistic), 6),
                     "file": p.name, "valid": bool(ok)})

    print(f"\n=== candidates, ranked by AUC (KS shown with OOF sampling sd removed) ===")
    print(f"{'slot':>4} {'name':30s} {'auc':>9} {'ks':>9} {'ks-sd':>9} {'vs v61':>9}")
    for r in rows:
        print(f"{r['slot']:>4} {r['name']:30s} {r['auc']:9.6f} {r['ks']:9.6f} "
              f"{r['ks_minus_sigma']:9.6f} {r['spearman_v61_anchor']:9.6f}")

    for n in sorted({1, 3, len(rows)}):
        mu = np.array([r["auc"] for r in rows[:n]])
        rng = np.random.default_rng(5)
        emax = 0.0 if n == 1 else float(np.mean(np.max(rng.standard_normal((40000, n)), axis=1)))
        print(f"  {n} diverse submits: expected maximum "
              f"+{emax * SIGMA_PUB:.5f} public ({emax:.2f} sd), independence upper bound")

    summary = {"sigma_oof": SIGMA_OOF, "sigma_public": SIGMA_PUB,
               "dedupe_threshold": TAU, "auc_floor_gap": FLOOR, "best_auc": best,
               "pool_size": len(scored), "eligible": len(eligible),
               "kept": len(kept), "candidates": rows}
    (OUT / "summary_v66_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v66_candidates.json'}")


if __name__ == "__main__":
    main()