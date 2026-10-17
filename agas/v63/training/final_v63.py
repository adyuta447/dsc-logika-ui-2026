"""Final v63 candidate set, from the verified grid.

An earlier size sweep read its ranking from a CSV written by a pass whose grid
also included v61 legs, so its consensus of five landed at 0.528626 instead of the
0.529867 that `verify_v63_consensus.py` reproduces and confirms against the
shipped file. This script rebuilds the same grid the verification used, so the
numbers here, the numbers there, and the files on disk all come from one code
path.

It sweeps consensus sizes 2 to 12, reports bagged and per-fold-seed KS for each,
and writes the best size plus a few sizes either side of it, plus the strongest
single grid point and conservative mixes on the public anchor.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
def _add_shared():
    """Load agas/shared/bootstrap.py by path; agas is not an importable package."""
    import importlib.util as _il
    _here = Path(__file__).resolve().parent
    for _p in _here.parents:
        _c = _p / "shared" / "bootstrap.py"
        if _c.exists():
            _spec = _il.spec_from_file_location("_agas_bootstrap", _c)
            _m = _il.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            if str(_c.parent) not in sys.path:
                sys.path.insert(0, str(_c.parent))
            return _p
    raise RuntimeError(f"shared/bootstrap.py not found above {_here}")

_SHARED = _add_shared()
import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE.parent / "output"
V62 = paths.version_output_of("v62")
V61 = paths.version_output_of("v61")
ANCHOR = V61 / "submission_v61_alt_ext0.5_tm30_lean.csv"
K62 = ["lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean", "lgb_faclean",
       "lgb_full", "cat_full", "xgb_full", "lgbx_full", "linear"]
K63 = ["sel100", "sel200", "sel200b", "sel300", "sel400",
       "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
LEG_SETS = [("v63", "sel400"), ("v63", "sel300"), ("v63", "sel100"),
            ("v63", "lgb_faclean"), ("v63", "lgb_full"),
            ("v62", "lgb_top"), ("v62", "lgb_fac"), ("v62", "lgb_topwide")]
RATIOS = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def validate(d, test_ids):
    bad = []
    if len(d) != 30000:
        bad.append(f"rows={len(d)}")
    if not d.application_id.is_unique:
        bad.append("dup id")
    if set(d.application_id) != set(np.asarray(test_ids)):
        bad.append("id mismatch")
    if not d.default_probability.between(0, 1).all():
        bad.append("prob range")
    if d.default_probability.isna().any():
        bad.append("nan")
    return bad


def main():
    os.chdir(ROOT)
    z63 = np.load(OUT / "preds_v63.npz", allow_pickle=True)
    z62 = np.load(V62 / "preds_v62.npz", allow_pickle=True)
    seeds = [int(s) for s in z63["seeds"]]
    ns = len(seeds)
    tr, te = z63["train_ids"], z63["test_ids"]
    ymap = pd.read_csv(ROOT / "data/train_applications.csv",
                       usecols=["application_id", "target"])
    y = ymap.set_index("application_id").reindex(pd.Index(tr))["target"].values.astype(int)

    def z(run):
        return z63 if run == "v63" else z62

    def bag(run, k):
        return rank01(z(run)[f"oof_{k}"])

    def tst(run, k):
        return rank01(z(run)[f"test_{k}"])

    def sd(run, k, j):
        return rank01(z(run)[f"oofseed_{k}_{j}"])

    TREE_SETS = {
        "v63sel": [("v63", "sel200", 3), ("v63", "sel200b", 1), ("v63", "sel300", 1),
                   ("v63", "lgbx_full", 1), ("v63", "cat_full", 1), ("v63", "xgb_full", 1),
                   ("v63", "linear", 0.6)],
        "v63mix": [("v63", "sel200", 2), ("v63", "sel300", 1), ("v63", "lgbx_full", 1),
                   ("v63", "cat_full", 1), ("v63", "xgb_full", 1), ("v63", "linear", 0.6),
                   ("v62", "lgb_top", 2), ("v62", "lgb_faclean", 1), ("v62", "xgb_full", 1)],
        "allrun": [("v63", "sel200", 3), ("v63", "sel200b", 1), ("v63", "sel300", 1),
                   ("v63", "lgbx_full", 1), ("v63", "cat_full", 1), ("v63", "xgb_full", 1),
                   ("v63", "linear", 0.5), ("v62", "lgb_top", 2), ("v62", "lgb_faclean", 1),
                   ("v62", "xgb_full", 1)],
    }
    bases = {}
    for nm, specs in TREE_SETS.items():
        for extra_lin in ([0.0, 0.5] if nm == "v63sel" else [0.0]):
            s = list(specs)
            if extra_lin:
                s = [(r, k, w * (1 - extra_lin) / (1 - 0.6 * extra_lin)) for r, k, w in s]
                s.append(("v63", "linear", extra_lin))
            bname = nm if not extra_lin else f"{nm}_lin{extra_lin}"
            tot = sum(w for _, _, w in s)
            bo = sum((w / tot) * bag(r, k) for r, k, w in s)
            bt = sum((w / tot) * tst(r, k) for r, k, w in s)
            po = [sum((w / tot) * sd(r, k, j) for r, k, w in s) for j in range(ns)]
            bases[bname] = (bo, bt, po)

    rows, objs = [], {}

    def two_axis(bo, bt, po, r1, k1, a, r2, k2, b):
        l1b, l1t = bag(r1, k1), tst(r1, k1)
        l2b, l2t = bag(r2, k2), tst(r2, k2)
        l1p = [sd(r1, k1, j) for j in range(ns)]
        l2p = [sd(r2, k2, j) for j in range(ns)]
        s1o, s1t = rank01(bo + a * (bo - l1b)), rank01(bt + a * (bt - l1t))
        s1p = [rank01(po[j] + a * (po[j] - l1p[j])) for j in range(ns)]
        return (rank01(s1o + b * (s1o - l2b)), rank01(s1t + b * (s1t - l2t)),
                [rank01(s1p[j] + b * (s1p[j] - l2p[j])) for j in range(ns)])

    for bname, (bo, bt, po) in bases.items():
        b_ks = [ks(y, v) for v in po]
        for run, key in LEG_SETS:
            lb, lt = bag(run, key), tst(run, key)
            lp = [sd(run, key, j) for j in range(ns)]
            for r in RATIOS:
                ps = [rank01(po[j] + r * (po[j] - lp[j])) for j in range(ns)]
                ks_s = [ks(y, v) for v in ps]
                d = [float(x) - float(bb) for x, bb in zip(ks_s, b_ks)]
                nm = f"{bname}|{run}:{key}|r{r}"
                objs[nm] = (rank01(bo + r * (bo - lb)), rank01(bt + r * (bt - lt)), ps)
                rows.append({"name": nm, "base": bname, "kind": "one",
                             "legs": f"{run}:{key}", "ratio": r, "ks": float(ks(y, objs[nm][0])),
                             "auc": float(roc_auc_score(y, objs[nm][0])),
                             "ks_base": float(ks(y, bo)),
                             "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                             "n_pos": int(sum(x > 0 for x in d)),
                             "robust": int(sum(x > 0 for x in d)) >= ns - 1})
        for (r1, k1), (r2, k2) in itertools.combinations(LEG_SETS, 2):
            for a in (0.25, 0.5, 0.75):
                for b in (0.25, 0.5, 0.75):
                    ex, ext, ps = two_axis(bo, bt, po, r1, k1, a, r2, k2, b)
                    ks_s = [ks(y, v) for v in ps]
                    d = [float(x) - float(bb) for x, bb in zip(ks_s, b_ks)]
                    nm = f"{bname}|{r1}:{k1}a{a}|{r2}:{k2}b{b}"
                    objs[nm] = (ex, ext, ps)
                    rows.append({"name": nm, "base": bname, "kind": "two",
                                 "legs": f"{r1}:{k1} / {r2}:{k2}", "ratio": f"{a}/{b}",
                                 "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                                 "ks_base": float(ks(y, bo)),
                                 "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                                 "n_pos": int(sum(x > 0 for x in d)),
                                 "robust": int(sum(x > 0 for x in d)) >= ns - 1})

    df = pd.DataFrame(rows)
    print(f"grid {len(df)} rows | best single KS {df.ks.max():.6f} | "
          f"robust {int(df.robust.sum())} ({100 * df.robust.mean():.1f}%)")

    by_ks = df.sort_values("ks", ascending=False).name.tolist()
    by_rob = df[df.robust].sort_values(["worst_delta", "ks"], ascending=False).name.tolist()
    picks, seen = [], []
    for nm in by_ks[:14] + by_rob[:14] + by_ks[:30]:
        t = objs[nm][1]
        if all(abs(spearmanr(t, q).statistic - 1) > 1e-9 for q in seen):
            picks.append(nm)
            seen.append(t)
        if len(picks) >= 12:
            break

    sweep = []
    for size in range(2, 13):
        mem = picks[:size]
        o = np.mean([rank01(objs[m][0]) for m in mem], axis=0)
        ps = [np.mean([rank01(objs[m][2][j]) for m in mem], axis=0) for j in range(ns)]
        sweep.append({"size": size, "ks": float(ks(y, o)),
                      "auc": float(roc_auc_score(y, o)),
                      "per_seed": [round(float(ks(y, v)), 5) for v in ps]})
    rep = pd.DataFrame(sweep).sort_values("ks", ascending=False)
    print("\n=== consensus size sweep ===")
    print(rep.to_string(index=False))

    best_size = int(rep.iloc[0]["size"])
    best_per_seed = rep.iloc[0]["per_seed"]
    base_name = picks[0].split("|")[0]
    base_ks = [float(ks(y, v)) for v in bases[base_name][2]]
    delta = [a - b for a, b in zip(best_per_seed, base_ks)]
    print(f"\nbest size {best_size}, base {base_name}")
    print(f"  base per seed      {base_ks}")
    print(f"  consensus per seed {best_per_seed}")
    print(f"  delta per seed     {[round(v, 5) for v in delta]} "
          f"positive {sum(v > 0 for v in delta)}/{ns}")

    anchor = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                    .reindex(pd.Index(te)).default_probability.values)
    written = []
    sizes = sorted({best_size, 3, 5, 7, 9, 12} & set(range(2, 13)))
    for size in sizes:
        mem = picks[:size]
        o = np.mean([rank01(objs[m][0]) for m in mem], axis=0)
        t = np.mean([rank01(objs[m][1]) for m in mem], axis=0)
        name = f"consensus{size}" + ("_best" if size == best_size else "")
        p = OUT / f"submission_v63f_{name}.csv"
        d = pd.DataFrame({"application_id": np.asarray(te),
                          "default_probability": rank01(t)})
        d.to_csv(p, index=False)
        written.append({"name": name, "ks": float(ks(y, o)),
                        "auc": float(roc_auc_score(y, o)),
                        "spearman_anchor": float(spearmanr(t, anchor).statistic),
                        "file": p.name, "problems": validate(d, te)})

    top = by_ks[0]
    ob, tb, ps1 = objs[top]
    p = OUT / "submission_v63f_best_single.csv"
    d = pd.DataFrame({"application_id": np.asarray(te), "default_probability": rank01(tb)})
    d.to_csv(p, index=False)
    written.append({"name": "best_single", "ks": float(ks(y, ob)),
                    "auc": float(roc_auc_score(y, ob)),
                    "per_seed": [round(float(ks(y, v)), 5) for v in ps1],
                    "spearman_anchor": float(spearmanr(tb, anchor).statistic),
                    "file": p.name, "problems": validate(d, te)})

    mem = picks[:best_size]
    ct = np.mean([rank01(objs[m][1]) for m in mem], axis=0)
    for w in (0.35, 0.6):
        t = rank01((1 - w) * anchor + w * ct)
        p = OUT / f"submission_v63f_anchor{int(w * 100)}_consensus.csv"
        d = pd.DataFrame({"application_id": np.asarray(te),
                          "default_probability": rank01(t)})
        d.to_csv(p, index=False)
        written.append({"name": f"anchor{int(w * 100)}_consensus", "ks": None, "auc": None,
                        "spearman_anchor": float(spearmanr(t, anchor).statistic),
                        "file": p.name, "problems": validate(d, te)})

    written.sort(key=lambda r: -(r["ks"] or 0))
    with open(OUT / "summary_v63_final.json", "w") as fh:
        json.dump({"seeds": seeds, "previous_record": 0.528717,
                   "best_consensus_size": best_size,
                   "consensus_per_seed": best_per_seed, "base_per_seed": base_ks,
                   "delta_per_seed": delta, "size_sweep": sweep,
                   "picks": picks, "candidates": written}, fh, indent=2)
    print("\n=== final candidates ===")
    print(pd.DataFrame(written).to_string(index=False))


if __name__ == "__main__":
    main()
