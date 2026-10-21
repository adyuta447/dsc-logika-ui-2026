"""v69 candidate builder: swap the lean leg into the proven v61/v66 chain.

The other six legs stay at their v68 ten-seed values. For every v69 lean variant
we rebuild:

    trees4 = mean(rank(lgb_full), rank(lean_v), rank(cat_full), rank(xgb_full))
    core   = (2 * trees4 + rank(lgbx_full)) / 3
    tm20   = 0.8 * core + 0.2 * rank(linear)
    tm30   = 0.7 * core + 0.3 * rank(linear)
    ext(r) = rank01(tm30 + r * (tm30 - rank(lean_v)))

and also a rank-stacked logistic meta-model over all legs. Candidates are
ranked by AUC, deduplicated at rank correlation 0.9995, and forced to include
the proven `ext0.5|tm30|<best lean>` direction.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
RATIOS = (0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 1.00, 1.50)


def rank01(v):
    return rankdata(v) / len(v)


def ks_score(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def align(v, ids):
    return pd.Series(np.asarray(v), index=ids).reindex(ids).values


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(OUT / "preds_v69.npz"))
    return ap.parse_args()


def main():
    args = parse()
    OUT.mkdir(parents=True, exist_ok=True)
    z68 = np.load(paths.version_output_of("v68") / "preds_v68.npz",
                  allow_pickle=True)
    z69 = np.load(args.preds, allow_pickle=True)
    train_ids = z69["train_ids"]
    test_ids = z69["test_ids"]
    assert set(train_ids) == set(z68["train_ids"])
    assert set(test_ids) == set(z68["test_ids"])
    y = z69["y"].astype(int)

    ro, rt = {}, {}
    for k in z68.files:
        if k.startswith("oof_"):
            ro[k[4:]] = rank01(align(z68[k], train_ids))
        elif k.startswith("test_"):
            rt[k[5:]] = rank01(align(z68[k], test_ids))
    leans = []
    for k in z69.files:
        if k.startswith("oof_"):
            name = k[4:]
            if name.startswith("seed_"):
                continue
            ro[name] = rank01(z69[k])
            rt[name] = rank01(z69[f"test_{name}"])
            leans.append(name)
    assert leans, "no v69 lean arrays"

    lean_avg_o = np.mean([ro[k] for k in leans], axis=0)
    lean_avg_t = np.mean([rt[k] for k in leans], axis=0)
    ro["lean_avg"], rt["lean_avg"] = rank01(lean_avg_o), rank01(lean_avg_t)

    chain_names = leans + ["lean_avg"]
    best_lean = max(chain_names, key=lambda k: roc_auc_score(y, ro[k]))
    print(f"lean variants: {leans}; best={best_lean} "
          f"auc={roc_auc_score(y, ro[best_lean]):.6f}")

    cands = {}
    for name in chain_names:
        trees_o = (ro["lgb_full"] + ro[name] + ro["cat_full"] + ro["xgb_full"]) / 4
        trees_t = (rt["lgb_full"] + rt[name] + rt["cat_full"] + rt["xgb_full"]) / 4
        core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
        core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
        tm20_o = 0.8 * core_o + 0.2 * ro["linear"]
        tm20_t = 0.8 * core_t + 0.2 * rt["linear"]
        tm30_o = 0.7 * core_o + 0.3 * ro["linear"]
        tm30_t = 0.7 * core_t + 0.3 * rt["linear"]
        cands[f"leg|{name}"] = (ro[name], rt[name])
        cands[f"trees4|{name}"] = (trees_o, trees_t)
        cands[f"core|{name}"] = (core_o, core_t)
        cands[f"tm20|{name}"] = (tm20_o, tm20_t)
        cands[f"tm30|{name}"] = (tm30_o, tm30_t)
        for base, bo, bt in (("tm20", tm20_o, tm20_t), ("tm30", tm30_o, tm30_t)):
            for r in RATIOS:
                cands[f"ext{r:g}|{base}|{name}"] = (
                    rank01(bo + r * (bo - ro[name])),
                    rank01(bt + r * (bt - rt[name])))
        for base, bo, bt, tag in (("tm20", tm20_o, tm20_t, "tm20"),
                                  ("tm30", tm30_o, tm30_t, "tm30")):
            wide_o = wide_t = 0.0
            for r in (0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75):
                o, t = rank01(bo + r * (bo - ro[name])), rank01(bt + r * (bt - rt[name]))
                wide_o += o
                wide_t += t
            cands[f"plateau|{tag}|{name}"] = (rank01(wide_o), rank01(wide_t))

    feats = ["fac_full", "lgb_full", "lgbx_full", "cat_full", "xgb_full",
             "linear"] + leans + ["lean_avg"]
    Mo = np.column_stack([ro[k] for k in feats])
    Mt = np.column_stack([rt[k] for k in feats])
    meta_o = np.zeros(len(y))
    meta_t = np.zeros(len(test_ids))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Mo, y):
        lr = LogisticRegression(C=0.5, max_iter=3000)
        lr.fit(Mo[a], y[a])
        meta_o[b] = lr.predict_proba(Mo[b])[:, 1]
        meta_t += lr.predict_proba(Mt)[:, 1] / 5
    cands["stack_logit"] = (rank01(meta_o), rank01(meta_t))

    anchor_rank = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                         .loc[test_ids].default_probability.values)
    old_lean_o = rank01(align(z68["oof_lgb_lean"], train_ids))
    old_lean_t = rank01(align(z68["test_lgb_lean"], test_ids))
    trees_o = (ro["lgb_full"] + old_lean_o + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + old_lean_t + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
    core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
    tm30_o = 0.7 * core_o + 0.3 * ro["linear"]
    tm30_t = 0.7 * core_t + 0.3 * rt["linear"]
    anchor_oof = rank01(tm30_o + 0.5 * (tm30_o - old_lean_o))
    for w in (0.10, 0.25, 0.50):
        cands[f"stack_anchor_w{w:g}"] = (
            rank01((1 - w) * meta_o + w * anchor_oof),
            rank01((1 - w) * meta_t + w * anchor_rank))

    scored = []
    for name, (o, t) in cands.items():
        scored.append({"name": name, "oof": o, "test": t,
                       "auc": float(roc_auc_score(y, o)),
                       "ks": ks_score(y, o)})
    scored.sort(key=lambda r: -r["auc"])
    best = scored[0]["auc"]
    eligible = [r for r in scored if r["auc"] >= best - FLOOR]
    print(f"pool {len(scored)}, eligible {len(eligible)} at AUC {best:.6f}")

    must_names = [f"ext0.5|tm30|{best_lean}"]
    must = [r for r in scored if r["name"] == must_names[0]]
    if not must:
        raise SystemExit(f"forced direction missing: {must_names[0]}")
    kept = []
    for r in must:
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    for r in scored:
        if r["name"] == must_names[0] and r not in kept:
            kept.append(r)
    for r in eligible:
        if r in kept:
            continue
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -r["auc"])
    print(f"kept {len(kept)}")

    stale = sorted(OUT.glob("submission_v69_*.csv"))
    for f in stale:
        f.unlink()

    rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v69_{i:02d}_{k['name'].replace('|', '_')}.csv"
        d = pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                     "ks": round(k["ks"], 6),
                     "ks_minus_sigma": round(k["ks"] - SIGMA_OOF, 6),
                     "spearman_v66_anchor": round(float(
                         spearmanr(k["test"], anchor_rank).statistic), 6),
                     "file": p.name, "valid": bool(ok)})

    print("\n=== candidates by AUC ===")
    print(f"{'slot':>4} {'name':44s} {'auc':>9} {'ks':>9} {'ks-sd':>9} {'vs v66':>9}")
    for r in rows:
        print(f"{r['slot']:>4} {r['name']:44s} {r['auc']:9.6f} {r['ks']:9.6f} "
              f"{r['ks_minus_sigma']:9.6f} {r['spearman_v66_anchor']:9.6f}")

    rng = np.random.default_rng(5)
    for n in sorted({1, 3, len(rows)}):
        emax = 0.0 if n == 1 else float(np.mean(np.max(rng.standard_normal((40000, n)), axis=1)))
        print(f"  {n} diverse submits: expected max +{emax * SIGMA_PUB:.5f} "
              f"public ({emax:.2f} sd), independence upper bound")

    summary = {"sigma_oof": SIGMA_OOF, "sigma_public": SIGMA_PUB,
               "dedupe_threshold": TAU, "auc_floor_gap": FLOOR,
               "best_auc": best, "lean_variants": leans, "best_lean": best_lean,
               "kept": len(kept), "candidates": rows}
    (OUT / "summary_v69_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v69_candidates.json'}")


if __name__ == "__main__":
    main()
