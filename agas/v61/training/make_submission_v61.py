"""Blend study and submission writer for the v61 prediction set.

Reads `agas/output/preds_v61.npz`, scores every base model and a family of
rank blends on out-of-fold data, then writes the candidate submissions plus a
summary. Blends are built only from directions that are supported by the OOF
evidence in this script; the report lists the metrics so the choice is
auditable. Nothing outside `agas/` is modified.
"""

import sys
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
ANCHOR = paths.ANCHOR


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default="preds_v61.npz")
    ap.add_argument("--prefix", type=str, default="v61")
    ap.add_argument("--targets", type=str, default="data/test_applications.csv")
    args = ap.parse_args()
    os.chdir(ROOT)

    z = np.load(OUT / args.preds, allow_pickle=True)
    apps = pd.read_csv(args.targets)
    ymap = pd.concat([
        pd.read_csv("data/train_applications.csv", usecols=["application_id", "target"]),
    ])
    oof = {k[4:]: z[k] for k in z.files if k.startswith("oof_")}
    test = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
    train_ids, test_ids = z["train_ids"], z["test_ids"]

    y = ymap.set_index("application_id").loc[train_ids].target.values.astype(int)

    ro = {k: rank01(v) for k, v in oof.items()}
    rt = {k: rank01(v) for k, v in test.items()}

    cands = {}

    def add(name, o, t):
        cands[name] = (np.asarray(o, dtype=float), np.asarray(t, dtype=float))

    for k in oof:
        add(k, ro[k], rt[k])

    trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    add("trees4", trees_o, trees_t)

    core_o = (trees_o * 2 + ro["lgbx_full"]) / 3
    core_t = (trees_t * 2 + rt["lgbx_full"]) / 3
    add("core", core_o, core_t)

    for w in (0.1, 0.2, 0.3):
        add(f"topmix{int(w * 100):02d}", (1 - w) * core_o + w * ro["linear"],
            (1 - w) * core_t + w * rt["linear"])

    meta_cols = [ro["lgb_full"], ro["lgb_lean"], ro["lgbx_full"], ro["cat_full"],
                 ro["xgb_full"], ro["linear"]]
    meta_cols_t = [rt["lgb_full"], rt["lgb_lean"], rt["lgbx_full"], rt["cat_full"],
                   rt["xgb_full"], rt["linear"]]
    Z = np.column_stack(meta_cols)
    Zt = np.column_stack(meta_cols_t)
    for c_seed in (101, 202):
        mo = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=c_seed).split(Z, y):
            m = LogisticRegression(C=0.05, max_iter=4000).fit(Z[a], y[a])
            mo[b] = m.predict_proba(Z[b])[:, 1]
        m = LogisticRegression(C=0.05, max_iter=4000).fit(Z, y)
        add(f"meta{c_seed}", mo, m.predict_proba(Zt)[:, 1])

    base_o, base_t = cands["topmix20"][0], cands["topmix20"][1]
    fac_o, fac_t = ro["fac_full"], rt["fac_full"]
    for r in (0.5, 1.0, 1.5, 2.0, 3.0):
        add(f"ext{r}_full_fac",
            rank01(base_o + r * (base_o - fac_o)), rank01(base_t + r * (base_t - fac_t)))

    lean_o, lean_t = ro["lgb_lean"], rt["lgb_lean"]
    for r in (0.5, 1.0):
        add(f"ext{r}_full_lean",
            rank01(base_o + r * (base_o - lean_o)), rank01(base_t + r * (base_t - lean_t)))

    meta_best = max((cands[k] for k in ("meta101", "meta202")), key=lambda p: ks(y, p[0]))
    for r in (0.5, 1.0, 1.5):
        add(f"ext{r}_meta_fac",
            rank01(meta_best[0] + r * (meta_best[0] - fac_o)),
            rank01(meta_best[1] + r * (meta_best[1] - fac_t)))

    anchor = pd.read_csv(ANCHOR)
    anchor_rank = anchor.set_index("application_id").loc[test_ids].default_probability.values
    anchor_rank = rank01(anchor_rank)
    for w in (0.15, 0.3, 0.5):
        add(f"anchor{int(w * 100)}_v61",
            np.zeros(0), rank01((1 - w) * anchor_rank + w * base_t))

    rows = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        if len(o):
            from sklearn.metrics import roc_auc_score
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
        rec["spearman_anchor"] = float(spearmanr(t, anchor_rank).statistic)
        rows.append(rec)
        if len(t) == len(test_ids):
            pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(t)}).to_csv(
                OUT / f"submission_{args.prefix}_{name}.csv", index=False)

    df = pd.DataFrame(rows)
    df = df.sort_values("ks", ascending=False, na_position="last")
    df.to_csv(OUT / f"blend_report_{args.prefix}.csv", index=False)
    with open(OUT / f"summary_{args.prefix}_blends.json", "w") as fh:
        json.dump({"base_score": float(ks(y, base_o)), "blends": rows}, fh, indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
