"""Final v68 candidate builder.

The blend chain is the v61/v66 chain, unchanged:

    trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
    core   = (2 * trees4 + lgbx_full) / 3
    tm20   = 0.8 * core + 0.2 * linear
    tm30   = 0.7 * core + 0.3 * linear
    alt    = rank01(tm30 + r * (tm30 - lgb_lean))

The only difference from v66 is that the legs are averaged over ten fold seeds
instead of six. The ratio grid is denser around 0.5, the ratio that produced
the best public score (0.56424), and a narrow plateau is added so no single
ratio is load-bearing. Candidate selection still keys on AUC, not KS: KS is a
maximum statistic with OOF sampling spread around 0.0036, and the public
leaderboard has its own noise.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
assert (ROOT / "agas/shared/paths.py").exists(), f"repo root wrong: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE / "output"
ANCHOR = paths.version_output_of("v66") / "submission_v66_04_ext0.5_tm30_lgb_lean.csv"
SIGMA_OOF, SIGMA_PUB = 0.0036, 0.0053
TAU = 0.9995
FLOOR = 0.0025
MUST_KEEP = "ext0.5|tm30|lgb_lean"

RATIOS = (0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 1.00, 1.50)


def rank01(v):
    return rankdata(v) / len(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "preds_v68.npz", allow_pickle=True)
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv",
                    usecols=["application_id", "target"]).set_index(
        "application_id").loc[train_ids].target.values.astype(int)

    ro = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    rt = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

    trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
    core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
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
            for r in RATIOS:
                cands[f"ext{r:g}|{base}|{leg}"] = ext(bo, bt, ro[leg], rt[leg], r)

    for leg in ("lgb_lean", "fac_full"):
        for base, bo, bt, name in (("tm20", tm20_o, tm20_t, "tm20"),
                                   ("tm30", tm30_o, tm30_t, "tm30")):
            wide_o, wide_t = 0.0, 0.0
            for r in (0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75):
                o, t = ext(bo, bt, ro[leg], rt[leg], r)
                wide_o += o
                wide_t += t
            cands[f"plateau|{name}|{leg}"] = (rank01(wide_o), rank01(wide_t))
            narrow_o, narrow_t = 0.0, 0.0
            for r in (0.45, 0.50, 0.55):
                o, t = ext(bo, bt, ro[leg], rt[leg], r)
                narrow_o += o
                narrow_t += t
            cands[f"plateau_narrow|{name}|{leg}"] = (rank01(narrow_o), rank01(narrow_t))

    scored = []
    for name, (o, t) in cands.items():
        scored.append({"name": name, "oof": o, "test": t,
                       "auc": float(roc_auc_score(y, o)),
                       "ks": float(ks_2samp(o[y == 1], o[y == 0]).statistic)})
    scored.sort(key=lambda r: -r["auc"])
    best = scored[0]["auc"]
    eligible = [r for r in scored if r["auc"] >= best - FLOOR]
    print(f"  pool {len(scored)}, AUC floor {FLOOR} below {best:.6f}: {len(eligible)}")

    must = next((r for r in scored if r["name"] == MUST_KEEP), None)
    if must is None:
        raise SystemExit(f"{MUST_KEEP} missing from candidate pool")
    if must not in eligible:
        print(f"  force-keeping {MUST_KEEP} below the AUC floor")
        eligible.append(must)

    anchor_rank = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                         .loc[test_ids].default_probability.values)

    kept = [must]
    for r in eligible:
        if r is must:
            continue
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -r["auc"])
    print(f"  after deduplicating at rank correlation {TAU}: {len(kept)}")

    stale = sorted(OUT.glob("submission_v68_*.csv"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"  removed {len(stale)} stale file(s) from an earlier run")

    rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v68_{i:02d}_{k['name'].replace('|', '_')}.csv"
        d = pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                     "ks": round(k["ks"], 6),
                     "ks_minus_sigma": round(k["ks"] - SIGMA_OOF, 6),
                     "spearman_v66_anchor": round(
                         float(spearmanr(k["test"], anchor_rank).statistic), 6),
                     "file": p.name, "valid": bool(ok)})

    print("\n=== candidates, ranked by AUC (KS shown with OOF sampling sd removed) ===")
    print(f"{'slot':>4} {'name':32s} {'auc':>9} {'ks':>9} {'ks-sd':>9} {'vs v66':>9}")
    for r in rows:
        print(f"{r['slot']:>4} {r['name']:32s} {r['auc']:9.6f} {r['ks']:9.6f} "
              f"{r['ks_minus_sigma']:9.6f} {r['spearman_v66_anchor']:9.6f}")

    for n in sorted({1, 3, len(rows)}):
        rng = np.random.default_rng(5)
        emax = 0.0 if n == 1 else float(np.mean(np.max(rng.standard_normal((40000, n)), axis=1)))
        print(f"  {n} diverse submits: expected maximum "
              f"+{emax * SIGMA_PUB:.5f} public ({emax:.2f} sd), independence upper bound")

    summary = {"sigma_oof": SIGMA_OOF, "sigma_public": SIGMA_PUB,
               "dedupe_threshold": TAU, "auc_floor_gap": FLOOR, "best_auc": best,
               "pool_size": len(scored), "eligible": len(eligible),
               "kept": len(kept), "candidates": rows}
    (OUT / "summary_v68_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v68_candidates.json'}")


if __name__ == "__main__":
    main()
