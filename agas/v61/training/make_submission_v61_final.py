"""Final v61 candidate builder.

Selection rule (fixed before looking at the leaderboard):

1. The extrapolation axis must be `full - fac`, the only direction that has
   previously produced a proven public score for this team
   (`qc_00_ext1.5_v4_fac_full_only.csv`, public 0.56171).
2. The ratio must sit on the broad plateau found by `study_blend_v61.py`
   (0.75 to 1.5), not on a single lucky spike.
3. A consensus rank average is offered because averaging rank vectors of
   neighbouring ratios removes most of the ratio choice risk.
4. Two conservative candidates perturb the proven anchor by a small weight.

Every file is validated: 30000 rows, unique ids matching the test set, and
probabilities inside [0, 1].
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
PREFIX = "v61"


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def validate(df, test_ids):
    problems = []
    if len(df) != 30000:
        problems.append(f"rows={len(df)}")
    if not df.application_id.is_unique:
        problems.append("duplicate application_id")
    if set(df.application_id) != set(test_ids):
        problems.append("id set mismatch")
    p = df.default_probability
    if p.min() < 0 or p.max() > 1:
        problems.append("probability out of range")
    if p.isna().any():
        problems.append("NaN probability")
    return problems


def main():
    os.chdir(ROOT)
    z = np.load(OUT / "preds_v61.npz", allow_pickle=True)
    oof = {k[4:]: z[k] for k in z.files if k.startswith("oof_")}
    test = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    ymap = pd.read_csv("data/train_applications.csv", usecols=["application_id", "target"])
    y = ymap.set_index("application_id").loc[train_ids].target.values.astype(int)

    ro = {k: rank01(v) for k, v in oof.items()}
    rt = {k: rank01(v) for k, v in test.items()}

    trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
    core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
    tm20_o, tm20_t = 0.8 * core_o + 0.2 * ro["linear"], 0.8 * core_t + 0.2 * rt["linear"]
    tm30_o, tm30_t = 0.7 * core_o + 0.3 * ro["linear"], 0.7 * core_t + 0.3 * rt["linear"]

    def ext(lo, lt, mo, mt, r):
        return rank01(lo + r * (lo - mo)), rank01(lt + r * (lt - mt))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    Z = np.column_stack([ro["lgb_full"], ro["lgb_lean"], ro["lgbx_full"],
                         ro["cat_full"], ro["xgb_full"], ro["linear"]])
    Zt = np.column_stack([rt["lgb_full"], rt["lgb_lean"], rt["lgbx_full"],
                          rt["cat_full"], rt["xgb_full"], rt["linear"]])
    mo = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=101).split(Z, y):
        mo[b] = LogisticRegression(C=0.05, max_iter=4000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
    mt = LogisticRegression(C=0.05, max_iter=4000).fit(Z, y).predict_proba(Zt)[:, 1]

    e075 = ext(tm20_o, tm20_t, ro["fac_full"], rt["fac_full"], 0.75)
    e100 = ext(tm20_o, tm20_t, ro["fac_full"], rt["fac_full"], 1.0)
    e150 = ext(tm20_o, tm20_t, ro["fac_full"], rt["fac_full"], 1.5)
    alt = ext(tm30_o, tm30_t, ro["lgb_lean"], rt["lgb_lean"], 0.5)

    anchor = rank01(pd.read_csv(paths.ANCHOR)
                    .set_index("application_id").loc[test_ids].default_probability.values)

    cands = {
        "ext0.75_full_fac": e075,
        "ext1.0_full_fac": e100,
        "ext1.5_full_fac": e150,
        "alt_ext0.5_tm30_lean": alt,
        "topmix20": (tm20_o, tm20_t),
        "meta_logit": (rank01(mo), rank01(mt)),
    }
    consensus_o = np.mean([rank01(e075[0]), rank01(e100[0]), rank01(e150[0]),
                           rank01(alt[0]), rank01(mo)], axis=0)
    consensus_t = np.mean([rank01(e075[1]), rank01(e100[1]), rank01(e150[1]),
                           rank01(alt[1]), rank01(mt)], axis=0)
    cands["consensus"] = (consensus_o, consensus_t)

    for w in (0.05, 0.15, 0.35):
        cands[f"anchor{int(w * 100)}_consensus"] = (
            np.zeros(0), rank01((1 - w) * anchor + w * consensus_t))

    rows = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        if len(o):
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
        df = pd.DataFrame({"application_id": test_ids, "default_probability": rank01(t)})
        path = OUT / f"submission_{PREFIX}_{name}.csv"
        df.to_csv(path, index=False)
        rec["file"] = str(path)
        rec["spearman_anchor"] = float(spearmanr(t, anchor).statistic)
        rec["problems"] = validate(df, test_ids)
        rows.append(rec)

    rows.sort(key=lambda r: r.get("ks", -1), reverse=True)
    with open(OUT / f"summary_{PREFIX}_final.json", "w") as fh:
        json.dump({"source": "preds_v61.npz", "candidates": rows}, fh, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
