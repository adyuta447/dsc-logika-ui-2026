"""Candidate builder for v65.

Same selection rule as v64, for the same reason. The leaderboard is a single
draw of 30000 rows and the sampling spread of KS on that many rows is around
0.005, so taking the argmax of a few thousand OOF KS values measures noise. AUC
is the ranking key, KS is reported beside it with its sampling spread attached,
and candidates that rank rows nearly identically are dropped because they buy
another draw of the same distribution rather than a new one.

The pool spans v61 to v65 on purpose. v65 is the strongest single run, but the
older runs are not the same model: v61's `full - fac` axis and v63's selection
axis produce rankings that differ from v65's, and that difference is the point.

An AUC floor is applied before deduplication. Without it the tail of the pool
fills up with weak models that survive deduplication precisely because they
rank differently, and a candidate with a worse mean can only lower the expected
maximum when every candidate has the same spread.
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
sys.path.insert(0, str(HERE.parents[1] / "agas/shared"))
sys.path.insert(0, str(HERE.parents[1]))
os.chdir(HERE.parents[1])

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE / "output"
ANCHOR = paths.version_output_of("v61") / "submission_v61_alt_ext0.5_tm30_lean.csv"
TAU = 0.998
FLOOR = 0.0025


def rank01(v):
    return rankdata(v) / len(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diag = json.loads((OUT / "summary_noise_diagnosis.json").read_text())
    sigma_pub = float(diag["sigma_public"])
    train = pd.read_csv(ROOT / "data/train_applications.csv")
    lab = train.set_index("application_id").target.astype(int)
    te_ids = pd.read_csv(ROOT / "data/test_applications.csv").application_id.values
    # pool entries carry the ids as well as the vector, because the runs do not
    # all write their rows in the same order and a silent positional alignment
    # would compare a prediction against the wrong label
    pool = {}

    def load_run(tag, ver):
        p = paths.version_output_of(ver) / f"preds_{tag}.npz"
        if not p.exists():
            return
        z = np.load(p, allow_pickle=True)
        oid, tid = z["train_ids"], z["test_ids"]
        for k in z.files:
            if k.startswith("oof_"):
                pool[f"{ver}|{k[4:]}"] = (z[k], oid, z[f"test_{k[4:]}"], tid)
        print(f"  {ver}: {sum(1 for k in z.files if k.startswith('oof_'))} vectors")

    load_run("v61", "v61")
    load_run("v62", "v62")
    load_run("v63", "v63")
    load_run("v64", "v64")
    load_run("v65", "v65")

    # blends of the two axis directions, which is where the public-proven
    # scores in this project came from
    v65 = np.load(OUT / "preds_v65.npz", allow_pickle=True)
    rsel = rank01(v65["oof_sel"])
    rlean = rank01(v65["oof_lean"])
    tsel = rank01(v65["test_sel"])
    tlean = rank01(v65["test_lean"])
    oid, tid = v65["train_ids"], v65["test_ids"]
    for r in (0.25, 0.5, 0.75, 1.0):
        pool[f"v65|sel-lean_r{r}"] = (rsel + r * (rsel - rlean), oid,
                                      tsel + r * (tsel - tlean), tid)
        pool[f"v65|lean-sel_r{r}"] = (rlean - r * (rsel - rlean), oid,
                                      tlean - r * (tsel - tlean), tid)
    for w in (0.25, 0.5, 0.75):
        pool[f"v65|mix_sel_lean_w{w}"] = ((1 - w) * rsel + w * rlean, oid,
                                          (1 - w) * tsel + w * tlean, tid)
    rx, rc = rank01(v65["oof_xgb"]), rank01(v65["test_xgb"])
    rcb, tcb = rank01(v65["oof_cat"]), rank01(v65["test_cat"])
    for other, ro, to in (("xgb", rx, tx := rank01(v65["test_xgb"])),
                          ("cat", rcb, tcb)):
        for w in (0.25, 0.5):
            pool[f"v65|sel+{other}_w{w}"] = ((1 - w) * rsel + w * ro, oid,
                                             (1 - w) * tsel + w * to, tid)
    # the four-way mixes the main run reported; they were the strongest AUC in
    # the project and the two-way pairs above do not reach them
    tx = rank01(v65["test_xgb"])
    for name, (wo, wt) in {
        "mix_trees": (0.5 * rsel + 0.25 * rx + 0.25 * rcb,
                      0.5 * tsel + 0.25 * tx + 0.25 * tcb),
        "mix_all4": (0.4 * rsel + 0.2 * rlean + 0.2 * rx + 0.2 * rcb,
                     0.4 * tsel + 0.2 * tlean + 0.2 * tx + 0.2 * tcb),
        "mix_sel_xgb": (0.6 * rsel + 0.4 * rx, 0.6 * tsel + 0.4 * tx),
        "mix_sel_cat": (0.6 * rsel + 0.4 * rcb, 0.6 * tsel + 0.4 * tcb),
        "consensus_ranks": ((rsel + rlean + rx + rcb) / 4,
                           (tsel + tlean + tx + tcb) / 4),
        "mix_all4_axis025": (0.4 * rsel + 0.2 * rlean + 0.2 * rx + 0.2 * rcb
                             + 0.25 * (rsel - rlean),
                             0.4 * tsel + 0.2 * tlean + 0.2 * tx + 0.2 * tcb
                             + 0.25 * (tsel - tlean)),
    }.items():
        pool[f"v65|{name}"] = (wo, oid, wt, tid)

    rows = []
    for name, (oof, oid_, _, _) in pool.items():
        yy = lab.reindex(oid_).values
        assert not pd.isna(yy).any(), f"{name} has unlabeled rows"
        rows.append({"name": name, "oof": oof,
                     "auc": float(roc_auc_score(yy, oof)),
                     "ks": float(ks_2samp(oof[yy == 1], oof[yy == 0]).statistic)})
    scored = sorted(rows, key=lambda r: -r["auc"])
    best = scored[0]["auc"]
    eligible = [r for r in scored if r["auc"] >= best - FLOOR]
    print(f"  pool {len(scored)}, AUC floor {FLOOR} below {best:.6f}: {len(eligible)} qualify")

    anchor_rank = pd.read_csv(ANCHOR).set_index("application_id").default_probability
    anchor_rank = rank01(anchor_rank.reindex(te_ids).values)

    kept = []
    for r in eligible:
        t = pool[r["name"]][2]
        if all(spearmanr(t, k["test"]).statistic < TAU for k in kept):
            kept.append({**r, "test": t})
    print(f"  after deduplicating at rank correlation {TAU}: {len(kept)}")

    stale = sorted(OUT.glob("submission_v65_*.csv"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"  removed {len(stale)} stale file(s) from an earlier run")

    out_rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v65_{i:02d}_{k['name'].replace('|', '_').replace('.', '')}.csv"
        d = pd.DataFrame({"application_id": te_ids, "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        out_rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                         "ks": round(k["ks"], 6),
                         "spearman_anchor": round(
                             float(spearmanr(k["test"], anchor_rank).statistic), 6),
                         "file": p.name, "valid": bool(ok)})

    summary = {"sigma_public": sigma_pub, "dedupe_threshold": TAU,
               "auc_floor_gap": FLOOR, "best_auc": best,
               "pool_size": len(scored), "eligible": len(eligible),
               "kept": len(kept), "candidates": out_rows}
    print(f"\n=== candidates, ranked by AUC (KS with one sampling sd subtracted) ===")
    print(f"{'slot':>4} {'name':38s} {'auc':>9} {'ks':>9} {'ks-sd':>9} {'vs anchor':>10}")
    for r in out_rows:
        print(f"{r['slot']:>4} {r['name']:38s} {r['auc']:9.6f} {r['ks']:9.6f} "
              f"{r['ks'] - sigma_pub:9.6f} {r['spearman_anchor']:10.6f}")

    for n in sorted({3, len(out_rows)}):
        mu = np.array([r["auc"] for r in out_rows[:n]])
        rng = np.random.default_rng(5)
        emax = float(np.mean(np.max(rng.standard_normal((40000, n)), axis=1)))
        print(f"  {n} diverse submits: expected maximum "
              f"+{emax * sigma_pub:.5f} public ({emax:.2f} sd), independence upper bound")
        summary[f"expected_max_over_{n}_public_upper_bound"] = emax * sigma_pub

    (OUT / "summary_v65_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v65_candidates.json'}")


if __name__ == "__main__":
    main()