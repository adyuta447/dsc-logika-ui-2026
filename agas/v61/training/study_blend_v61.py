"""Robustness study for the v61 blend family.

The OOF KS standard error on 70k rows is around 0.002-0.003, so picking the
single best ratio on one OOF vector is noise fitting. This script re-scores
every blend on several random halves of the OOF rows and reports the mean and
worst half, which is what the candidate ranking should be based on.
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


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def main():
    os.chdir(ROOT)
    z = np.load(OUT / "preds_v61.npz", allow_pickle=True)
    oof = {k[4:]: z[k] for k in z.files if k.startswith("oof_")}
    test = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
    train_ids = z["train_ids"]
    test_ids = z["test_ids"]
    ymap = pd.read_csv("data/train_applications.csv", usecols=["application_id", "target"])
    y = ymap.set_index("application_id").loc[train_ids].target.values.astype(int)

    ro = {k: rank01(v) for k, v in oof.items()}
    rt = {k: rank01(v) for k, v in test.items()}

    trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
    core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
    legs_o = {
        "topmix20": (0.8 * core_o + 0.2 * ro["linear"], 0.8 * core_t + 0.2 * rt["linear"]),
        "topmix30": (0.7 * core_o + 0.3 * ro["linear"], 0.7 * core_t + 0.3 * rt["linear"]),
        "trees4": (trees_o, trees_t),
        "lgb_full": (ro["lgb_full"], rt["lgb_full"]),
    }
    min_o = {
        "fac_full": (ro["fac_full"], rt["fac_full"]),
        "lgb_lean": (ro["lgb_lean"], rt["lgb_lean"]),
    }

    rng = np.random.RandomState(7)
    halves = []
    for _ in range(6):
        perm = rng.permutation(len(y))
        halves.append(perm[: len(perm) // 2])

    rows = []
    for lname, (lo, lt) in legs_o.items():
        for mname, (mo, mt) in min_o.items():
            for r in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
                bo = rank01(lo + r * (lo - mo))
                bt = rank01(lt + r * (lt - mt))
                hks = [ks(y[h], bo[h]) for h in halves]
                rows.append({
                    "blend": f"{lname}_minus_{mname}",
                    "ratio": r,
                    "auc": float(roc_auc_score(y, bo)),
                    "ks": float(ks(y, bo)),
                    "ks_half_mean": float(np.mean(hks)),
                    "ks_half_min": float(np.min(hks)),
                    "spearman_anchor": float(spearmanr(bt, rank01(
                        pd.read_csv(paths.ANCHOR)
                        .set_index("application_id").loc[test_ids].default_probability.values)).statistic),
                })

    df = pd.DataFrame(rows)
    df["score"] = df.ks_half_mean + df.ks_half_min
    df = df.sort_values("ks_half_mean", ascending=False)
    df.to_csv(OUT / "blend_robustness_v61.csv", index=False)
    with open(OUT / "summary_v61_robustness.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
