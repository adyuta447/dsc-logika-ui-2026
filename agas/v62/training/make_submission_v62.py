"""v62 candidate builder.

Reads the bagged v62 predictions, re-runs the gradient search to decide which
extrapolation axes are supported (positive KS delta in nearly every fold seed),
and writes a candidate set around those axes plus conservative mixes anchored on
the new public-best submission.

The anchor for the conservative mixes is
`agas/output/submission_v61_alt_ext0.5_tm30_lean.csv`, which the team scored at
`0.56312` on the public board, so it replaces the older `qc_00` file at `0.56171`
as the thing worth protecting.
"""

import sys
import argparse
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
TREE_KEYS = ["lgb_full", "lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean",
             "lgb_faclean", "cat_full", "xgb_full"]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=str, default="v62")
    args = ap.parse_args()
    os.chdir(ROOT)

    z = np.load(OUT / f"preds_{args.prefix}.npz", allow_pickle=True)
    seeds = list(z["seeds"])
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    y = y.set_index("application_id").loc[train_ids].target.values.astype(int)

    bag = {k: rank01(z[f"oof_{k}"]) for k in TREE_KEYS + ["lgbx_full", "linear"]}
    bagt = {k: rank01(z[f"test_{k}"]) for k in bag}

    def per(k, j):
        return rank01(z[f"oofseed_{k}_{j}"])

    def mix(spec):
        bo = sum(w * bag[k] for k, w in spec)
        bt = sum(w * bagt[k] for k, w in spec)
        po = [sum(w * per(k, j) for k, w in spec) for j in range(len(seeds))]
        return bo, bt, po

    bases = {}
    for w_lin, nm in [(0.1, "topmix10"), (0.2, "topmix20"), (0.3, "topmix30")]:
        spec = [(k, (1 - w_lin) / len(TREE_KEYS)) for k in TREE_KEYS]
        spec += [("lgbx_full", 0.15 * (1 - w_lin)), ("linear", w_lin)]
        bases[nm] = mix(spec)
    bases["trees"] = mix([(k, 1.0 / len(TREE_KEYS)) for k in TREE_KEYS])

    legs = ["lgb_fac", "lgb_lean", "lgb_faclean", "lgb_top", "lgb_topwide"]
    ratios = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]

    scored = []
    for bname, (bo, bt, po) in bases.items():
        b_ks = [ks(y, v) for v in po]
        for lname in legs:
            for r in ratios:
                ex = rank01(bo + r * (bo - bag[lname]))
                ks_s = [ks(y, rank01(po[j] + r * (po[j] - per(lname, j))))
                        for j in range(len(seeds))]
                delta = [a - b for a, b in zip(ks_s, b_ks)]
                scored.append({
                    "axis": f"{bname}-{lname}", "ratio": r,
                    "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                    "ks_base": float(ks(y, bo)),
                    "worst_delta": float(min(delta)), "mean_delta": float(np.mean(delta)),
                    "n_pos": int(sum(d > 0 for d in delta)),
                    "obj": (bo, bt, ex),
                })

    df = pd.DataFrame([{k: v for k, v in s.items() if k != "obj"} for s in scored])
    df["robust"] = df.n_pos >= len(seeds) - 1
    df = df.sort_values(["robust", "worst_delta", "ks"], ascending=[False, False, False])
    df.to_csv(OUT / f"gradient_{args.prefix}_final.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== axes ranked by robustness then worst per-seed delta, top 20 ===")
    print(df.head(20).to_string(index=False))

    # candidates: the three most robust axes, three ratios each
    picked = []
    for axis in df[df.robust].axis.unique()[:3]:
        sub = df[df.axis == axis].sort_values("worst_delta", ascending=False)
        rs = sorted({r for r in sub.ratio.head(3)})
        for r in rs:
            picked.append(next(s for s in scored if s["axis"] == axis and s["ratio"] == r))

    # deduplicate nearly identical blends
    uniq, seen = [], []
    for s in picked:
        t = s["obj"][1]
        if all(abs(spearmanr(t, q).statistic - 1) > 1e-9 for q in seen):
            uniq.append(s)
            seen.append(t)

    anchor_rank = rank01(pd.read_csv(ANCHOR)
                         .set_index("application_id").loc[test_ids].default_probability.values)

    cands = {}
    for i, s in enumerate(uniq):
        short = s["axis"].replace("topmix", "tm").replace("trees", "tr") + f"_r{s['ratio']}"
        cands[short] = (s["obj"][0], s["obj"][1])

    con_o = np.mean([rank01(u["obj"][0]) for u in uniq[:5]], axis=0)
    con_t = np.mean([rank01(u["obj"][1]) for u in uniq[:5]], axis=0)
    cands["consensus"] = (con_o, con_t)

    for w in (0.10, 0.25, 0.50):
        cands[f"anchor{int(w * 100)}_consensus"] = (np.zeros(0), rank01((1 - w) * anchor_rank + w * con_t))

    rows = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        if len(o):
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
        path = OUT / f"submission_{args.prefix}_{name}.csv"
        df_out = pd.DataFrame({"application_id": test_ids, "default_probability": rank01(t)})
        df_out.to_csv(path, index=False)
        rec["file"] = path.name
        rec["spearman_anchor"] = float(spearmanr(t, anchor_rank).statistic)
        rec["problems"] = validate(df_out, test_ids)
        rows.append(rec)

    rows.sort(key=lambda r: r.get("ks", -1), reverse=True)
    with open(OUT / f"summary_{args.prefix}_candidates.json", "w") as fh:
        json.dump({"anchor": ANCHOR.name, "seeds": seeds, "candidates": rows}, fh, indent=2)
    print("\n=== candidates ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
