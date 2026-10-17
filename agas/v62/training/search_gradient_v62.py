"""Gradient extrapolation search for v62, with the reduced leg excluded from the base.

The first pass of this search had a structural flaw: the base blend averaged all
eight tree legs, including the very leg being extrapolated against, which blurs
the axis. Here the base is rebuilt for every pair so the reduced leg is never
part of it.

Two extrapolation directions are already proven on the public board:
`v4 - fac` (`0.56171`) and `topmix30 - lgb_lean` (`0.56312`). Both take a model
trained on a richer feature set and push it away from the same family trained on
a reduced set. v62 keeps four levels of that richness ladder plus two
gain-selected levels, so the search covers:

    full 554 | topwide 350 | fac 509 | lean 446 | faclean 401 | top 200

For every base definition and every leg and ratio, the KS delta is also computed
inside each individual fold seed. The bagged OOF KS has a standard error near
0.002 to 0.003, larger than the gap between most candidates, so a direction that
only improves the bagged number is not trustworthy. Ranking is by
positive-in-every-seed first, then worst per-seed delta, then bagged KS.
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
ALL_KEYS = ["lgb_full", "lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean",
            "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
LEGS = ["lgb_fac", "lgb_lean", "lgb_faclean", "lgb_top", "lgb_topwide"]
BASE_TREES = {
    "b1_v61": ["lgb_full", "lgb_lean", "cat_full", "xgb_full"],
    "b2_top": ["lgb_full", "lgb_top", "cat_full", "xgb_full"],
    "b3_topwide": ["lgb_full", "lgb_top", "lgb_topwide", "cat_full", "xgb_full"],
    "b4_topsonly": ["lgb_top", "lgb_topwide", "cat_full", "xgb_full"],
    "b5_notop": ["lgb_full", "lgb_fac", "cat_full", "xgb_full"],
}
RATIOS = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]
W_LIN = [0.1, 0.2, 0.3]


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def main(prefix="v62", out_prefix="v62"):
    os.chdir(ROOT)
    z = np.load(OUT / f"preds_{prefix}.npz", allow_pickle=True)
    seeds = [int(s) for s in z["seeds"]]
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    y = y.set_index("application_id").loc[train_ids].target.values.astype(int)

    bag = {k: rank01(z[f"oof_{k}"]) for k in ALL_KEYS}
    bagt = {k: rank01(z[f"test_{k}"]) for k in ALL_KEYS}
    pers = {k: [rank01(z[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in ALL_KEYS}

    def build(tree_keys, w_lin, exclude=None):
        keys = [k for k in tree_keys if k != exclude]
        if not keys:
            return None
        wt = (1 - w_lin) / (len(keys) + 0.15 * 3)  # trees share, scaled below
        # core = (2/3 * mean(trees) + 1/3 * lgbx); then mix with linear
        tree_share = 1 - w_lin
        spec = [(k, tree_share / 3.0 / len(keys)) for k in keys]
        spec.append(("lgbx_full", tree_share / 3.0))
        spec.append(("linear", w_lin))
        bo = sum(w * bag[k] for k, w in spec)
        bt = sum(w * bagt[k] for k, w in spec)
        po = [sum(w * pers[k][j] for k, w in spec) for j in range(len(seeds))]
        return bo, bt, po, float(sum(w for _, w in spec))

    rows = []
    objs = {}
    for bname, tkeys in BASE_TREES.items():
        for w in W_LIN:
            for leg in LEGS:
                built = build(tkeys, w, exclude=leg)
                if built is None:
                    continue
                bo, bt, po, wsum = built
                b_ks = [ks(y, v) for v in po]
                for r in RATIOS:
                    ex = rank01(bo + r * (bo - bag[leg]))
                    ext = rank01(bt + r * (bt - bagt[leg]))
                    ks_s = [ks(y, rank01(po[j] + r * (po[j] - pers[leg][j])))
                            for j in range(len(seeds))]
                    delta = [a - b for a, b in zip(ks_s, b_ks)]
                    nm = f"{bname}_w{w}-{leg}_r{r}"
                    objs[nm] = (ex, ext)
                    rows.append({
                        "name": nm, "base": bname, "w_lin": w, "leg": leg, "ratio": r,
                        "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                        "ks_base": float(ks(y, bo)),
                        "worst_delta": float(min(delta)), "mean_delta": float(np.mean(delta)),
                        "n_pos": int(sum(d > 0 for d in delta)),
                    })

    df = pd.DataFrame(rows)
    df["robust"] = df.n_pos >= len(seeds) - 1
    df = df.sort_values(["robust", "worst_delta", "ks"], ascending=[False, False, False])
    df.to_csv(OUT / f"gradient_{out_prefix}.csv", index=False)

    pd.set_option("display.width", 240)
    print(f"=== top 25 by robustness then worst per-seed delta (seeds={seeds}) ===")
    print(df.head(25).to_string(index=False))
    print("\n=== top 15 by bagged KS only ===")
    print(df.sort_values("ks", ascending=False).head(15).to_string(index=False))
    print(f"\nrobust rows: {int(df.robust.sum())} / {len(df)}")

    with open(OUT / f"summary_gradient_{out_prefix}.json", "w") as fh:
        json.dump({"seeds": seeds, "rows": rows}, fh, indent=2)
    return df, objs


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "v62",
         sys.argv[2] if len(sys.argv) > 2 else "v62")
