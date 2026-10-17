"""v64 candidate builder: diversity first, argmax last.

The v63 run picked its submission by the largest out-of-fold KS among about 1300
blends and got 0.557 on the leaderboard. `diagnose_selection_noise.py` shows why
that was not a decision: KS is a maximum over thresholds, so its sampling sd is
0.0036 even at 70000 rows, and choosing the largest of 1300 noisy measurements
inflates the number by roughly 0.012 for free. The reported improvement from v61
to v63 was 0.0012, a third of one sampling sd.

So this builder does the opposite of the previous ones.

1. It pools every stored prediction vector from v61, v62 and v63, plus the v64
   run, and deduplicates them by rank correlation.
2. It keeps a set that is spread out in rank space rather than clustered at the
   top of any metric, because two files that rank the rows the same way give the
   leaderboard the same draw.
3. It scores each candidate by AUC, the smooth part of the ranking quality, and
   reports KS with its sampling sd attached rather than as a ranking key.
4. It reports the expected maximum over the candidate set under the measured
   public noise, so the value of the remaining submit slots is explicit.

Nothing is submitted from here, and no file outside `agas/` is touched.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "shared"))
import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE / "output"
ANCHOR = paths.version_output_of("v61") / "submission_v61_alt_ext0.5_tm30_lean.csv"
TAU = 0.998   # two candidates closer than this rank correlate are the same draw
FLOOR = 0.0025  # a candidate more than this far below the best AUC is not worth a slot


def rank01(v):
    return rankdata(v) / len(v)


def ks_cols(p, y):
    o = np.argsort(-p, axis=0)
    yy = y[o]
    npos = yy.sum()
    return float(np.max(np.cumsum(yy) / npos - np.cumsum(1 - yy) / (len(yy) - npos)))


def load_run(version, keys):
    p = paths.version_output_of(version) / f"preds_{version}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    oof = {k: rank01(z[f"oof_{k}"]) for k in keys if f"oof_{k}" in z.files}
    test = {k: rank01(z[f"test_{k}"]) for k in keys if f"test_{k}" in z.files}
    return oof, test, z["train_ids"], z["test_ids"]


RUNS = {
    "v61": ["lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear", "fac_full"],
    "v62": ["lgb_full", "lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean", "lgb_faclean",
            "lgbx_full", "cat_full", "xgb_full", "linear"],
    "v63": ["sel100", "sel200", "sel200b", "sel300", "sel400",
            "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"],
    "v64": ["sel200", "sel400", "lgb_full", "lgb_faclean", "lgbx_full",
            "cat_full", "xgb_full", "linear"],
}


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    diag = json.loads((OUT / "summary_noise_diagnosis.json").read_text())
    sigma_oof = float(diag["sigma_oof"])
    sigma_pub = float(diag["sigma_public"])

    tr = te = None
    y = None
    pool = {}
    for ver, keys in RUNS.items():
        got = load_run(ver, keys)
        if got is None:
            print(f"  {ver}: no prediction file, skipped")
            continue
        oof, test, tr_v, te_v = got
        if tr is None:
            tr, te = tr_v, te_v
            ymap = pd.read_csv(ROOT / "data/train_applications.csv",
                               usecols=["application_id", "target"])
            y = ymap.set_index("application_id").reindex(pd.Index(tr))["target"].values.astype(int)
        for k, v in oof.items():
            pool[f"{ver}|{k}"] = (v, test[k])
        print(f"  {ver}: {len(oof)} vectors")

    # blends of a rich leg against a reduced one, the two public-proven shapes
    pairs = [
        ("v63|sel200", "v63|sel400"), ("v63|sel200", "v63|lgb_full"),
        ("v63|sel200", "v63|lgb_faclean"), ("v62|lgb_top", "v62|lgb_fac"),
        ("v62|lgb_top", "v62|lgb_faclean"), ("v62|lgb_full", "v62|lgb_faclean"),
        ("v61|lgb_full", "v61|fac_full"), ("v61|lgb_full", "v61|lgb_lean"),
        ("v64|sel200", "v64|lgb_faclean"), ("v64|sel200", "v64|lgb_full"),
        ("v63|sel200", "v62|lgb_fac"), ("v64|sel200", "v63|sel400"),
    ]
    for (hi, lo) in pairs:
        if hi in pool and lo in pool:
            for r in (0.25, 0.5, 0.75, 1.0):
                a, b = pool[hi][0], pool[lo][0]
                pool[f"ext{r}|{hi}-{lo}"] = (rank01(a + r * (a - b)),
                                            rank01(pool[hi][1] + r * (pool[hi][1] - pool[lo][1])))
    # rank averages across runs, which is where partial decorrelation lives
    for tag, members in {
        "avg_v62v63_sel": [("v62|lgb_top", 1), ("v63|sel200", 1), ("v63|sel200b", 1)],
        "avg_v61v62v63": [("v61|lgb_full", 1), ("v62|lgb_top", 1), ("v63|sel200", 1)],
        "avg_sel_lean": [("v63|sel200", 2), ("v63|lgb_faclean", 1)],
        "avg_all_runs": [("v61|lgb_full", 1), ("v62|lgb_top", 1), ("v63|sel200", 1),
                         ("v64|sel200", 1)],
    }.items():
        members = [(k, w) for k, w in members if k in pool]
        if len(members) < 2:
            continue
        tot = sum(w for _, w in members)
        pool[tag] = (sum(w * pool[k][0] for k, w in members) / tot,
                     sum(w * pool[k][1] for k, w in members) / tot)
    print(f"  pool after blends: {len(pool)} vectors")

    # ---------------------------------------------------------------- dedupe
    names = sorted(pool)
    scored = []
    for n in names:
        o, t = pool[n]
        scored.append({"name": n, "auc": float(roc_auc_score(y, o)),
                       "ks": ks_cols(o, y), "oof": o, "test": t})
    scored.sort(key=lambda r: -r["auc"])
    best_auc = scored[0]["auc"]
    n_before = len(scored)
    eligible = [r for r in scored if r["auc"] >= best_auc - FLOOR]
    print(f"  AUC floor {FLOOR} below the best {best_auc:.6f}: {len(eligible)} of "
          f"{n_before} vectors qualify")

    kept = []
    for r in eligible:
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    print(f"  after deduplicating at rank correlation {TAU}: {len(kept)} vectors "
          f"(from {len(eligible)} eligible)")

    # coverage of rank space, reported so the spread is visible rather than asserted
    anchor_rank = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                         .reindex(pd.Index(te)).default_probability.values)
    kept = sorted(kept, key=lambda r: -r["auc"])
    corrs = [spearmanr(k["test"], anchor_rank).statistic for k in kept]
    print(f"  rank correlation to the 0.56312 file: min {min(corrs):.4f} "
          f"median {float(np.median(corrs)):.4f} max {max(corrs):.4f}")

    # Clear this builder's own prefix before writing. Three consecutive runs
    # left thirteen files behind and the slot numbers collided: slot 3 held
    # `v61|lgb_full` from the first run and `v64|xgb_full` from the third. Both
    # valid, both claiming slot 3, so which file you got depended on when it
    # was generated. The prefix is narrow on purpose, `submission_v64_`, so it
    # cannot touch the v61 to v63 outputs or the anchor.
    stale = sorted(OUT.glob("submission_v64_*.csv"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"  removed {len(stale)} stale file(s) from an earlier builder run")

    rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v64_{i:02d}_{k['name'].replace('|', '_').replace('.', '')}.csv"
        d = pd.DataFrame({"application_id": np.asarray(te),
                          "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and set(d.application_id) == set(np.asarray(te))
              and d.default_probability.between(0, 1).all()
              and not d.default_probability.isna().any())
        rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                     "ks": round(k["ks"], 6),
                     "ks_minus_sd": round(k["ks"] - sigma_oof, 6),
                     "spearman_anchor": round(spearmanr(k["test"], anchor_rank).statistic, 6),
                     "file": p.name, "valid": bool(ok)})

    for N in (3, 10, len(rows)):
        emax = 0.0 if N <= 1 else float(np.mean(np.max(
            np.random.default_rng(5).standard_normal((20000, min(N, len(rows)))), axis=1)))
        rows_note = emax
    print("\n=== candidates, ranked by AUC (KS shown with one sampling sd subtracted) ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    summary = {
        "objective": "AUC; KS reported with its sampling sd",
        "sigma_oof": sigma_oof, "sigma_public": sigma_pub,
        "dedupe_threshold": TAU, "auc_floor_gap": FLOOR,
        "best_auc": best_auc, "pool_size": len(scored),
        "eligible": len(eligible),
        "kept": len(kept), "candidates": rows,
    }
    # Upper bound only. This assumes independent public draws, but the kept
    # candidates rank rows at Spearman 0.984 to 0.994 against the public-best
    # file, so their draws are largely the same draw and the real gain is
    # smaller. A correlation-adjusted number is not computable here: that would
    # need the correlation of the public noise itself, which needs several
    # independent pairs of rankings, and the project has exactly one such pair.
    for n in sorted({min(N, len(rows)) for N in (3, 10, 20, 39)}):
        emax = 0.0 if n <= 1 else float(np.mean(np.max(
            np.random.default_rng(5).standard_normal((40000, n)), axis=1)))
        summary[f"expected_max_over_{n}_submits_sd_upper_bound"] = emax
        summary[f"expected_max_over_{n}_submits_public_upper_bound"] = emax * sigma_pub
        print(f"  {n:3d} diverse submits: expected maximum +{emax * sigma_pub:.5f} public "
              f"({emax:.2f} sd), independence upper bound")
    (OUT / "summary_v64_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v64_candidates.json'}")


if __name__ == "__main__":
    main()
