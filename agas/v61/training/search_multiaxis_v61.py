"""Multi-axis extrapolation search on the v61 prediction set.

The team just proved a second public direction: `topmix30 - lgb_lean` (ratio 0.5)
scored `0.56312`, better than the old anchor `qc_00` at `0.56171`. That is the
same mechanism as the original `v4 - fac` recipe: extrapolate a richer feature
set away from a reduced one. So the natural next step is to extrapolate along
two such axes at once instead of one.

Everything here reuses the saved OOF and test vectors in `preds_v61.npz`, so
nothing is retrained. Selection uses six random half-samples of the OOF rows
as well as the full-sample KS, because the KS standard error at 70k rows is
around 0.002-0.003 and single-split ranking is noise fitting.
"""

import sys
import itertools
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


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def load():
    z = np.load(OUT / "preds_v61.npz", allow_pickle=True)
    oof = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    test = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv",
                    usecols=["application_id", "target"])
    y = y.set_index("application_id").loc[train_ids].target.values.astype(int)
    return oof, test, y, train_ids, test_ids


def build_legs(oof, test):
    trees_o = (oof["lgb_full"] + oof["lgb_lean"] + oof["cat_full"] + oof["xgb_full"]) / 4
    trees_t = (test["lgb_full"] + test["lgb_lean"] + test["cat_full"] + test["xgb_full"]) / 4
    core_o = (trees_o * 2 + oof["lgbx_full"]) / 3
    core_t = (trees_t * 2 + test["lgbx_full"]) / 3
    legs = {
        "topmix20": (0.8 * core_o + 0.2 * oof["linear"], 0.8 * core_t + 0.2 * test["linear"]),
        "topmix30": (0.7 * core_o + 0.3 * oof["linear"], 0.7 * core_t + 0.3 * test["linear"]),
        "topmix10": (0.9 * core_o + 0.1 * oof["linear"], 0.9 * core_t + 0.1 * test["linear"]),
        "trees4": (trees_o, trees_t),
        "core": (core_o, core_t),
    }
    return legs


def main():
    os.chdir(ROOT)
    oof, test, y, train_ids, test_ids = load()
    legs = build_legs(oof, test)

    rng = np.random.RandomState(7)
    halves = [rng.permutation(len(y))[: len(y) // 2] for _ in range(6)]

    rows = []

    def score(name, bo, bt):
        hks = [ks(y[h], bo[h]) for h in halves]
        rows.append({
            "name": name,
            "auc": float(roc_auc_score(y, bo)),
            "ks": float(ks(y, bo)),
            "half_mean": float(np.mean(hks)),
            "half_min": float(np.min(hks)),
        })

    # stage 1: single axis, fine grid around and beyond the proven direction
    grid = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]
    for bname in ("topmix10", "topmix20", "topmix30", "trees4"):
        for lname in ("lgb_lean", "fac_full"):
            for r in grid:
                bo = rank01(legs[bname][0] + r * (legs[bname][0] - oof[lname]))
                bt = rank01(legs[bname][1] + r * (legs[bname][1] - test[lname]))
                score(f"{bname}_-{lname}_r{r}", bo, bt)

    stage1 = pd.DataFrame(rows)
    print("=== stage 1: single axis, top 15 by full-sample KS ===")
    print(stage1.sort_values("ks", ascending=False).head(15).to_string(index=False))

    # stage 2: two axes at once, centred on the proven directions
    rows2 = []

    def score2(name, bo, bt):
        hks = [ks(y[h], bo[h]) for h in halves]
        rows2.append({
            "name": name,
            "auc": float(roc_auc_score(y, bo)),
            "ks": float(ks(y, bo)),
            "half_mean": float(np.mean(hks)),
            "half_min": float(np.min(hks)),
        })

    for bname in ("topmix30", "topmix20"):
        bo0, bt0 = legs[bname]
        for a in (0.25, 0.5, 0.75, 1.0):
            for b in (0.0, 0.25, 0.5, 0.75, 1.0):
                for lname in ("fac_full",):
                    bo = rank01(bo0 + a * (bo0 - oof["lgb_lean"]) + b * (bo0 - oof[lname]))
                    bt = rank01(bt0 + a * (bt0 - test["lgb_lean"]) + b * (bt0 - test[lname]))
                    score2(f"{bname}_a{a}_lean_b{b}_{lname}", bo, bt)

    stage2 = pd.DataFrame(rows2)
    print("\n=== stage 2: two axes, top 15 by full-sample KS ===")
    print(stage2.sort_values("ks", ascending=False).head(15).to_string(index=False))

    out = OUT / "multiaxis_search_v61.csv"
    pd.concat([stage1.assign(stage=1), stage2.assign(stage=2)]).to_csv(out, index=False)
    with open(OUT / "summary_multiaxis_v61.json", "w") as fh:
        json.dump({"stage1": rows, "stage2": rows2}, fh, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
