"""Final v62 candidate set.

What the search established: OOF KS has plateaued. v61's best extrapolation
reaches KS `0.528624` and v62's reaches `0.527846`, while v62's base model is
clearly the better ranker (bagged AUC `0.847419` for `lgb_top` against
`0.846647`). A 0.0008 gap is smaller than the OOF KS standard error, so neither
run wins on evidence alone, and the two runs are only partly correlated because
they used different feature blocks, three versus four fold seeds, and different
feature-selection policies.

Two things follow from that and drive this script:

1. Average the two runs before extrapolating. Averaging partly independent
   rankers reduces variance, and the mixed base is the only construction that
   uses the stronger v62 ranker without discarding the public-proven v61 one.
2. Stop chasing a single best number. With the standard error near 0.002-0.003,
   the defensible choice is a small diverse set built on directions that were
   either proven on the public board or hold up in every fold seed, plus a
   rank-average consensus and conservative mixes on the new anchor.

A two-axis pass is included because both proven directions use the same
mechanism, so pushing along two richness gradients at once is the natural
extension.
"""

import sys
import json
import os
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
ANCHOR = paths.version_output_of("v61") / "submission_v61_alt_ext0.5_tm30_lean.csv"
V61_TREES = ["lgb_full", "lgb_lean", "cat_full", "xgb_full"]
V62_TREES = ["lgb_full", "lgb_top", "cat_full", "xgb_full"]


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def validate(df, test_ids):
    bad = []
    if len(df) != 30000:
        bad.append(f"rows={len(df)}")
    if not df.application_id.is_unique:
        bad.append("dup id")
    if set(df.application_id) != set(test_ids):
        bad.append("id mismatch")
    if not df.default_probability.between(0, 1).all():
        bad.append("prob range")
    if df.default_probability.isna().any():
        bad.append("nan")
    return bad


def main():
    os.chdir(ROOT)
    z61 = np.load(paths.version_output_of("v61") / "preds_v61.npz", allow_pickle=True)
    z62 = np.load(OUT / "preds_v62.npz", allow_pickle=True)
    seeds = [int(s) for s in z62["seeds"]]
    tr, te = z61["train_ids"], z61["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    ymap = y.set_index("application_id")
    y = ymap.reindex(pd.Index(tr))
    y = y["target"].values.astype(int)

    K61 = ["lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear", "fac_full"]
    K62 = ["lgb_full", "lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean",
           "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
    B61 = {k: rank01(z61[f"oof_{k}"]) for k in K61}
    T61 = {k: rank01(z61[f"test_{k}"]) for k in K61}
    B62 = {k: rank01(z62[f"oof_{k}"]) for k in K62}
    T62 = {k: rank01(z62[f"test_{k}"]) for k in K62}
    P62 = {k: [rank01(z62[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in K62}

    def core(w_lin):
        def one(run, trees, B, T):
            spec = [(k, (1 - w_lin) / 3.0 / len(trees)) for k in trees]
            spec += [("lgbx_full", (1 - w_lin) / 3.0), ("linear", w_lin)]
            return (sum(x * B[k] for k, x in spec), sum(x * T[k] for k, x in spec))
        return one(62, V62_TREES, B62, T62), one(61, V61_TREES, B61, T61)

    cores = {w: core(w) for w in (0.1, 0.2, 0.3)}
    mix = {w: (0.5 * cores[w][0][0] + 0.5 * cores[w][1][0],
               0.5 * cores[w][0][1] + 0.5 * cores[w][1][1]) for w in (0.1, 0.2, 0.3)}

    rng = np.random.RandomState(11)
    halves = [rng.permutation(len(y))[: len(y) // 2] for _ in range(8)]

    def half(v):
        h = [ks(y[i], v[i]) for i in halves]
        return float(np.mean(h)), float(np.min(h))

    cands = {}

    def add(name, o, t):
        cands[name] = (np.asarray(o, float), np.asarray(t, float))

    def ext(base_o, base_t, leg, run, r):
        lb = (B62 if run == 62 else B61)[leg]
        lt = (T62 if run == 62 else T61)[leg]
        return rank01(base_o + r * (base_o - lb)), rank01(base_t + r * (base_t - lt))

    # single-axis candidates chosen from the combined search
    picks = [
        ("mixA_topwide_r025", "mix", 0.3, "lgb_topwide", 62, 0.25),
        ("mixB_facv61_r05", "mix", 0.2, "fac_full", 61, 0.5),
        ("mixC_facv62_r05", "mix", 0.2, "lgb_fac", 62, 0.5),
        ("mixD_faclean_r05", "mix", 0.2, "lgb_faclean", 62, 0.5),
        ("v61_fac_r025", "v61", 0.3, "fac_full", 61, 0.25),
        ("v61_facv62_r05", "v61", 0.1, "lgb_fac", 62, 0.5),
        ("v62_faclean_r025", "v62", 0.2, "lgb_faclean", 62, 0.25),
        ("v62_fac_r05", "v62", 0.2, "lgb_fac", 62, 0.5),
    ]
    for nm, which, w, leg, run, r in picks:
        if which == "mix":
            bo, bt = mix[w]
        elif which == "v61":
            bo, bt = cores[w][1]
        else:
            bo, bt = cores[w][0]
        add(nm, *ext(bo, bt, leg, run, r))

    # two axes at once on the mixed base, one leg from each run
    for w in (0.2, 0.3):
        bo, bt = mix[w]
        for a in (0.25, 0.5):
            for b in (0.25, 0.5):
                o1, t1 = ext(bo, bt, "lgb_fac", 62, a)
                o2, t2 = ext(bo, bt, "fac_full", 61, b)
                add(f"two_w{w}_fac{a}_fac61_{b}",
                    rank01(o1 + b * (o1 - o2)), rank01(t1 + b * (t1 - t2)))

    # plain mixed base, no extrapolation, as a reference point
    add("mix_base_w02", *mix[0.2])

    # consensus over the most defensible single-axis candidates
    core_names = [n for n, *_ in picks]
    con_o = np.mean([rank01(cands[n][0]) for n in core_names], axis=0)
    con_t = np.mean([rank01(cands[n][1]) for n in core_names], axis=0)
    add("consensus", con_o, con_t)

    anchor = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                    .reindex(pd.Index(te)).default_probability.values)
    for w in (0.25, 0.50, 0.75):
        add(f"anchor{int(w * 100)}_consensus", np.zeros(0), rank01((1 - w) * anchor + w * con_t))

    rows = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        hm, hn = (float(np.nan), float(np.nan))
        if len(o):
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
            hm, hn = half(o)
            rec["half_mean"] = hm
            rec["half_min"] = hn
        path = OUT / f"submission_v62f_{name}.csv"
        d = pd.DataFrame({"application_id": np.asarray(te), "default_probability": rank01(t)})
        d.to_csv(path, index=False)
        rec["file"] = path.name
        rec["spearman_anchor"] = float(spearmanr(t, anchor).statistic)
        rec["problems"] = validate(d, np.asarray(te))
        rows.append(rec)

    rows.sort(key=lambda r: r.get("ks", -1), reverse=True)
    with open(OUT / "summary_v62_final.json", "w") as fh:
        json.dump({"anchor": ANCHOR.name, "anchor_public": 0.56312, "seeds": seeds,
                   "candidates": rows}, fh, indent=2)
    pd.set_option("display.width", 240)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
