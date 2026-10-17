"""v63 direction search and candidate builder.

Three things are established going in. Feature selection is the real lever, so
the strongest leg is `sel200` (200 gain-selected columns) rather than the full
554-column set. Both proved public directions extrapolate a richer-feature model
away from a reduced one, so selection sharpness counts as another axis of that
same quantity: extrapolating away from `sel400` sharpens the gradient the same
way extrapolating away from `faclean` does. And OOF KS has a standard error near
0.002 to 0.003, so a direction is only recommended when it holds up inside
individual fold seeds, not just on the bagged vector.

Every v63 leg stores per-fold-seed OOF, so per-seed deltas are available for all
of them. Bases that mix v61, v62 and v63 are judged by row resampling instead,
since v61 kept no per-seed OOF.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
ANCHOR = paths.version_output_of("v61") / "submission_v61_alt_ext0.5_tm30_lean.csv"
V63_KEYS = ["sel100", "sel200", "sel200b", "sel300", "sel400",
            "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
LEGS63 = ["sel400", "sel300", "sel100", "lgb_full", "lgb_faclean"]
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


def main(prefix="v63", write=True):
    os.chdir(ROOT)
    z = np.load(OUT / f"preds_{prefix}.npz", allow_pickle=True)
    seeds = [int(s) for s in z["seeds"]]
    tr, te = z["train_ids"], z["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    ymap = y.set_index("application_id").reindex(pd.Index(tr))
    y = ymap["target"].values.astype(int)

    B = {k: rank01(z[f"oof_{k}"]) for k in V63_KEYS}
    T = {k: rank01(z[f"test_{k}"]) for k in V63_KEYS}
    P = {k: [rank01(z[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in V63_KEYS}

    # older runs, for a base that spans runs
    B62 = {}
    T62 = {}
    p62 = OUT / "preds_v62.npz"
    if p62.exists():
        z62 = np.load(p62, allow_pickle=True)
        for k in ["lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean", "lgb_faclean",
                  "lgb_full", "cat_full", "xgb_full", "lgbx_full", "linear"]:
            B62[k] = rank01(z62[f"oof_{k}"])
            T62[k] = rank01(z62[f"test_{k}"])

    rng = np.random.RandomState(11)
    halves = [rng.permutation(len(y))[: len(y) // 2] for _ in range(8)]

    def half(v):
        h = [ks(y[i], v[i]) for i in halves]
        return float(np.mean(h)), float(np.min(h))

    def blend(Bd, Td, Pd, spec):
        bo = sum(w * Bd[k] for k, w in spec)
        bt = sum(w * Td[k] for k, w in spec)
        po = None
        if Pd is not None:
            po = [sum(w * Pd[k][j] for k, w in spec) for j in range(len(seeds))]
        return bo, bt, po

    # base definitions. w_lin is the spline-linear share, the rest is a mean of
    # tree legs including the LightGBM extra-config leg.
    TREE_SETS = {
        "sel": ["sel200", "sel200b", "sel300", "cat_full", "xgb_full"],
        "sel_min": ["sel200", "sel200b", "cat_full", "xgb_full"],
        "sel_wide": ["sel200", "sel200b", "sel300", "sel400", "cat_full", "xgb_full"],
        "v62_like": ["sel200", "sel300", "lgb_full", "cat_full", "xgb_full"],
    }
    bases = {}
    for nm, trees in TREE_SETS.items():
        for w in (0.0, 0.1, 0.2):
            spec = [(k, (1 - w) / (len(trees) + 0.15 * 3)) for k in trees]
            spec += [("lgbx_full", 0.15 * (1 - w)), ("linear", w)]
            bases[f"{nm}_w{w}"] = blend(B, T, P, spec)

    if B62:
        for w in (0.0, 0.1, 0.2):
            a = [(k, (1 - w) / 6) for k in ["sel200", "sel300", "cat_full", "xgb_full",
                                            "lgbx_full", "linear"]]
            b = [(k, 0.5 * (1 - w) / 4) for k in ["lgb_top", "lgb_topwide", "lgb_full", "xgb_full"]]
            spec = a + b
            bo = sum(wt * B[k] for k, wt in a) + sum(wt * B62[k] for k, wt in b)
            bt = sum(wt * T[k] for k, wt in a) + sum(wt * T62[k] for k, wt in b)
            bases[f"mixrun_w{w}"] = (bo, bt, None)

    rows, objs = [], {}

    # full grid
    for bname, (bo, bt, po) in bases.items():
        b_ks = [ks(y, v) for v in po] if po is not None else None
        hm0, hn0 = half(bo)
        for leg in LEGS63 + (["fac_full", "lgb_top"] if B62 else []):
            if leg in B62:
                lb, lt, lp = B62[leg], T62[leg], None
            else:
                lb, lt, lp = B[leg], T[leg], P[leg]
            for r in RATIOS:
                ex = rank01(bo + r * (bo - lb))
                ext = rank01(bt + r * (bt - lt))
                ehm, ehn = half(ex)
                nm = f"{bname}-{leg}_r{r}"
                rec = {"name": nm, "base": bname, "leg": leg, "ratio": r,
                       "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                       "ks_base": float(ks(y, bo)), "half_mean": ehm, "half_min": ehn,
                       "half_min_delta": ehn - hn0, "half_mean_delta": ehm - hm0}
                if b_ks is not None and lp is not None:
                    ks_s = [ks(y, rank01(po[j] + r * (po[j] - lp[j]))) for j in range(len(seeds))]
                    d = [a - b for a, b in zip(ks_s, b_ks)]
                    rec["worst_delta"] = float(min(d))
                    rec["mean_delta"] = float(np.mean(d))
                    rec["n_pos"] = int(sum(x > 0 for x in d))
                    rec["robust"] = rec["n_pos"] >= len(seeds) - 1
                else:
                    rec["robust"] = rec["half_min_delta"] > 0
                objs[nm] = (ex, ext)
                rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"gradient_{prefix}.csv", index=False)
    pd.set_option("display.width", 250)
    print("=== top 25 by bagged KS ===")
    print(df.sort_values("ks", ascending=False).head(25).to_string(index=False))
    rob = df[df.n_pos.notna()]
    print(f"\n=== v63-only bases, per-seed robust (top 15 by worst_delta) ===")
    print(rob.sort_values(["robust", "worst_delta", "ks"], ascending=[False, False, False])
          .head(15).to_string(index=False))
    print(f"\nrobust rows: {int(rob.robust.sum())} / {len(rob)}  (all rows: {int(df.robust.sum())} / {len(df)})")

    if not write:
        return df, objs, bo, bt, seeds, y, tr, te

    # candidates: most robust axes, then a consensus, then anchor mixes
    order = df.copy()
    order["_key"] = order.n_pos.fillna(-1)
    picked = []
    seen_axes = []
    for _, r in order.sort_values(["ks"], ascending=False).iterrows():
        if r.leg in seen_axes and len(seen_axes) < 3:
            continue
        picked.append(r["name"])
        seen_axes.append(r.leg)
        if len(picked) >= 9:
            break

    uniq, seen = [], []
    for nm in picked:
        t = objs[nm][1]
        if all(abs(spearmanr(t, q).statistic - 1) > 1e-9 for q in seen):
            uniq.append(nm)
            seen.append(t)

    cands = {}
    for nm in uniq:
        short = nm.replace("sel_min_w0.2-", "sm").replace("sel_wide_w0.1-", "sw") \
                  .replace("sel_w0.1-", "s").replace("mixrun_w0.1-", "mx") \
                  .replace("_full", "F").replace("_faclean", "FL").replace("_sel400", "400") \
                  .replace("_sel300", "300").replace("_fac_full", "v62f") \
                  .replace("_r", "r")
        cands[short] = objs[nm]

    con_o = np.mean([rank01(objs[nm][0]) for nm in uniq[:5]], axis=0)
    con_t = np.mean([rank01(objs[nm][1]) for nm in uniq[:5]], axis=0)
    cands["consensus"] = (con_o, con_t)

    anchor = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                    .reindex(pd.Index(te)).default_probability.values)
    for w in (0.25, 0.5, 0.75):
        cands[f"anchor{int(w * 100)}_consensus"] = (np.zeros(0), rank01((1 - w) * anchor + w * con_t))

    out_rows = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        if len(o):
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
            rec["half_mean"], rec["half_min"] = half(o)
        path = OUT / f"submission_{prefix}_{name}.csv"
        d = pd.DataFrame({"application_id": np.asarray(te), "default_probability": rank01(t)})
        d.to_csv(path, index=False)
        rec["file"] = path.name
        rec["spearman_anchor"] = float(spearmanr(t, anchor).statistic)
        rec["problems"] = validate(d, te)
        out_rows.append(rec)

    out_rows.sort(key=lambda r: r.get("ks", -1), reverse=True)
    with open(OUT / f"summary_{prefix}_final.json", "w") as fh:
        json.dump({"anchor": ANCHOR.name, "anchor_public": 0.56312, "seeds": seeds,
                   "best_ks_previous_run": 0.528717, "candidates": out_rows}, fh, indent=2)
    print("\n=== candidates ===")
    print(pd.DataFrame(out_rows).to_string(index=False))
    return df, objs, None, None, seeds, y, tr, te


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v63",
         "--nowrite" not in sys.argv)
