"""Per-fold-seed verification of the v63 consensus, rebuilt from the same code path.

An earlier attempt reconstructed the candidates with slightly different weights
and reported 0.528804 for the consensus and 0.525100 for the v62 record, so its
comparison was not measuring what it claimed. This version rebuilds the bases and
the extrapolation exactly as `search_multirun_v63.py` does, repeats the same
deterministic pick sequence, and only then computes per-seed KS for the members
that actually form the consensus.

The point of the check is narrow and worth stating plainly: the bagged OOF KS
standard error at 70k rows is around 0.002 to 0.003, and the gain over the previous
record is 0.00115. Whether that gain is real can only be answered seed by seed.
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
V61 = paths.version_output_of("v61"
                              )
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


def load(path, keys):
    z = np.load(path, allow_pickle=True)
    seeds = [int(s) for s in z["seeds"]]
    return ({k: rank01(z[f"oof_{k}"]) for k in keys},
            {k: rank01(z[f"test_{k}"]) for k in keys},
            {k: [rank01(z[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in keys},
            seeds, z["train_ids"], z["test_ids"])


def main():
    os.chdir(ROOT)
    b63, t63, p63, seeds, tr, te = load(OUT / "preds_v63.npz", K63)
    b62, t62, p62, s62, _, _ = load(V62 / "preds_v62.npz", K62)
    assert seeds == s62
    ns = len(seeds)
    ymap = pd.read_csv(ROOT / "data/train_applications.csv",
                       usecols=["application_id", "target"])
    y = ymap.set_index("application_id").reindex(pd.Index(tr))["target"].values.astype(int)

    def legv(run, key):
        if run == "v63":
            return b63[key], t63[key], p63[key]
        return b62[key], t62[key], p62[key]

    def base(specs):
        tot = sum(w for _, _, w in specs)
        bo = bt = 0.0
        po = [0.0] * ns
        for run, key, w in specs:
            ww = w / tot
            b, t, p = legv(run, key)
            bo = bo + ww * b
            bt = bt + ww * t
            for j in range(ns):
                po[j] = po[j] + ww * p[j]
        return bo, bt, po

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
            bases[nm if not extra_lin else f"{nm}_lin{extra_lin}"] = base(s)

    rows, objs = [], {}

    def two_axis(bo, bt, po, r1, k1, a, r2, k2, b):
        l1b, l1t, l1p = legv(r1, k1)
        l2b, l2t, l2p = legv(r2, k2)
        s1 = rank01(bo + a * (bo - l1b))
        s1t = rank01(bt + a * (bt - l1t))
        s1p = [rank01(po[j] + a * (po[j] - l1p[j])) for j in range(ns)]
        return (rank01(s1 + b * (s1 - l2b)), rank01(s1t + b * (s1t - l2t)),
                [rank01(s1p[j] + b * (s1p[j] - l2p[j])) for j in range(ns)])


    def two_axis_ks(y, bo, bt, po, r1, k1, a, r2, k2, b):
        ex, ext, ps = two_axis(bo, bt, po, r1, k1, a, r2, k2, b)
        return ex, ext, [ks(y, v) for v in ps]

    def single_axis_ps(y, po, lp, r):
        return [rank01(po[j] + r * (po[j] - lp[j])) for j in range(ns)]

    for bname, (bo, bt, po) in bases.items():
        b_ks = [ks(y, v) for v in po]
        b_auc = float(roc_auc_score(y, bo))
        for run, key in LEG_SETS:
            lb, lt, lp = legv(run, key)
            for r in RATIOS:
                ex = rank01(bo + r * (bo - lb))
                ext = rank01(bt + r * (bt - lt))
                ps = single_axis_ps(y, po, lp, r)
                ks_s = [ks(y, v) for v in ps]
                d = [float(a) - float(bb) for a, bb in zip(ks_s, b_ks)]
                nm = f"{bname}|{run}:{key}|r{r}"
                objs[nm] = (ex, ext, ps)
                rows.append({"name": nm, "base": bname, "leg1": f"{run}:{key}", "leg2": "",
                             "ratio": r, "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                             "ks_base": float(ks(y, bo)), "auc_base": b_auc,
                             "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                             "n_pos": int(sum(x > 0 for x in d)),
                             "robust": int(sum(x > 0 for x in d)) >= ns - 1})
        for (r1, k1), (r2, k2) in itertools.combinations(LEG_SETS, 2):
            for a in (0.25, 0.5, 0.75):
                for b in (0.25, 0.5, 0.75):
                    ex, ext, ps = two_axis(bo, bt, po, r1, k1, a, r2, k2, b)
                    ks_s = [ks(y, v) for v in ps]
                    d = [float(x) - float(yb) for x, yb in zip(ks_s, b_ks)]
                    nm = f"{bname}|{r1}:{k1}a{a}|{r2}:{k2}b{b}"
                    objs[nm] = (ex, ext, ps)
                    rows.append({"name": nm, "base": bname, "leg1": f"{r1}:{k1}",
                                 "leg2": f"{r2}:{k2}", "ratio": f"{a}/{b}",
                                 "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                                 "ks_base": float(ks(y, bo)), "auc_base": b_auc,
                                 "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                                 "n_pos": int(sum(x > 0 for x in d)),
                                 "robust": int(sum(x > 0 for x in d)) >= ns - 1})

    df = pd.DataFrame(rows)
    print(f"grid rows {len(df)} | best KS {df.ks.max():.6f} | robust {int(df.robust.sum())} "
          f"({100 * df.robust.mean():.1f}%)")

    by_ks = df.sort_values("ks", ascending=False).name.tolist()
    by_rob = df[df.robust].sort_values(["worst_delta", "ks"], ascending=False).name.tolist()
    picks, seen = [], []
    for nm in by_ks[:14] + by_rob[:14] + by_ks[:30]:
        t = objs[nm][1]
        if all(abs(spearmanr(t, q).statistic - 1) > 1e-9 for q in seen):
            picks.append(nm)
            seen.append(t)
        if len(picks) >= 8:
            break

    print("\nconsensus members:")
    for i, nm in enumerate(picks[:5]):
        print(f"  {i + 1}. {nm}  ks={df.set_index('name').loc[nm, 'ks']:.6f}")

    members = picks[:5]
    con_o = np.mean([rank01(objs[n][0]) for n in members], axis=0)
    con_t = np.mean([rank01(objs[n][1]) for n in members], axis=0)
    con_ps = [np.mean([rank01(objs[n][2][j]) for n in members], axis=0) for j in range(ns)]

    base_name = members[0].split("|")[0]
    bo, bt, po = bases[base_name]
    base_ps = [ks(y, v) for v in po]

    print(f"\n=== per fold seed (seeds {seeds}) ===")
    print(f"  {'base ' + base_name:44s} ks={ks(y, bo):.6f} "
          f"per_seed={[round(v, 5) for v in base_ps]}")
    con_ks = [ks(y, v) for v in con_ps]
    print(f"  {'v63 consensus':44s} ks={ks(y, con_o):.6f} "
          f"per_seed={[round(v, 5) for v in con_ks]}")
    d = [a - b for a, b in zip(con_ks, base_ps)]
    print(f"\nextrapolation delta per seed: {[round(x, 5) for x in d]}")
    print(f"positive in {sum(x > 0 for x in d)}/{ns} seeds | mean {np.mean(d):+.5f} "
          f"| worst {min(d):+.5f}")

    anchor = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                    .reindex(pd.Index(te)).default_probability.values)
    # confirm the rebuilt consensus matches the shipped file
    shipped = pd.read_csv(OUT / "submission_v63f_consensus5_best.csv")
    shipped_r = rank01(shipped.set_index("application_id")
                       .reindex(pd.Index(te)).default_probability.values)
    print(f"\nrebuilt vs shipped submission_v63f_consensus5_best.csv spearman: "
          f"{spearmanr(con_t, shipped_r).statistic:.8f}")
    print(f"spearman vs public anchor (0.56312): {spearmanr(con_t, anchor).statistic:.6f}")
    print(f"previous record 0.528717 -> consensus {ks(y, con_o):.6f} "
          f"({ks(y, con_o) - 0.528717:+.6f})")

    with open(OUT / "summary_v63_verify.json", "w") as fh:
        json.dump({"seeds": seeds, "base": base_name,
                   "base_ks_per_seed": base_ps, "consensus_ks_per_seed": con_ks,
                   "delta_per_seed": d, "consensus_ks": float(ks(y, con_o)),
                   "consensus_auc": float(roc_auc_score(y, con_o)),
                   "members": members}, fh, indent=2)


if __name__ == "__main__":
    main()
