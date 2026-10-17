"""Cross-run and two-axis extrapolation search for v63.

Two fixes and one extension over the first v63 pass.

The first pass looked for the v62 predictions inside `v63/output/`, so the
cross-run bases were silently never built and only v63-only bases appeared in the
report. Both files are addressed explicitly here.

The extension follows the one result that actually broke the plateau. The v62
record of KS 0.528717 did not come from a single axis, it came from extrapolating
two feature-richness gradients at once, so the same construction is applied here
to the v63 legs and to v62 plus v63 combined bases. Both runs keep per-fold-seed
OOF over the same four seeds, so per-seed deltas are available for every
cross-run candidate and nothing has to fall back on row resampling.
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


def load(path, keys):
    z = np.load(path, allow_pickle=True)
    b = {k: rank01(z[f"oof_{k}"]) for k in keys}
    t = {k: rank01(z[f"test_{k}"]) for k in keys}
    seeds = [int(s) for s in z["seeds"]]
    p = {k: [rank01(z[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in keys}
    return b, t, p, seeds, z["train_ids"], z["test_ids"]


def main():
    os.chdir(ROOT)
    b63, t63, p63, seeds, tr, te = load(OUT / "preds_v63.npz", K63)
    b62, t62, p62, s62, _, _ = load(V62 / "preds_v62.npz", K62)
    assert seeds == s62, (seeds, s62)

    ymap = pd.read_csv(ROOT / "data/train_applications.csv",
                       usecols=["application_id", "target"])
    y = ymap.set_index("application_id").reindex(pd.Index(tr))["target"].values.astype(int)
    ns = len(seeds)

    def leg(run, key):
        if run == "v63":
            return b63[key], t63[key], p63[key]
        return b62[key], t62[key], p62[key]

    def base(specs):
        """specs: list of (run, key, weight); weights normalised to one."""
        tot = sum(w for _, _, w in specs)
        bo = bt = 0.0
        po = [0.0] * ns
        for run, key, w in specs:
            ww = w / tot
            b, t, p = leg(run, key)
            bo = bo + ww * b
            bt = bt + ww * t
            for j in range(ns):
                po[j] = po[j] + ww * p[j]
        return bo, bt, po

    BASE_SETS = {
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
    for nm, specs in BASE_SETS.items():
        for extra_lin in ([0.0, 0.5] if nm == "v63sel" else [0.0]):
            s = list(specs)
            if extra_lin:
                s = [(r, k, w * (1 - extra_lin) / (1 - 0.6 * extra_lin)) for r, k, w in s]
                s.append(("v63", "linear", extra_lin))
            bases[nm if not extra_lin else f"{nm}_lin{extra_lin}"] = base(s)

    rows, objs = [], {}
    for bname, (bo, bt, po) in bases.items():
        b_ks = [ks(y, v) for v in po]
        b_auc = float(roc_auc_score(y, bo))
        cache = {}
        for run, key in LEG_SETS:
            lb, lt, lp = leg(run, key)
            cache[(run, key)] = []
            for r in RATIOS:
                ex = rank01(bo + r * (bo - lb))
                ext = rank01(bt + r * (bt - lt))
                ks_s = [ks(y, rank01(po[j] + r * (po[j] - lp[j]))) for j in range(ns)]
                d = [a - b for a, b in zip(ks_s, b_ks)]
                nm = f"{bname}|{run}:{key}|r{r}"
                objs[nm] = (ex, ext)
                rows.append({"name": nm, "base": bname, "leg1": f"{run}:{key}", "leg2": "",
                             "ratio": r, "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                             "ks_base": float(ks(y, bo)), "auc_base": b_auc,
                             "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                             "n_pos": int(sum(x > 0 for x in d)),
                             "robust": int(sum(x > 0 for x in d)) >= ns - 1})
                cache[(run, key)].append(nm)

        # two axes at once
        for (r1, k1), (r2, k2) in itertools.combinations(LEG_SETS, 2):
            for a in (0.25, 0.5, 0.75):
                for b in (0.25, 0.5, 0.75):
                    l1b, l1t, l1p = leg(r1, k1)
                    l2b, l2t, l2p = leg(r2, k2)
                    ex = rank01(rank01(bo + a * (bo - l1b)) + b * (rank01(bo + a * (bo - l1b)) - l2b))
                    ext = rank01(rank01(bt + a * (bt - l1t)) + b * (rank01(bt + a * (bt - l1t)) - l2t))
                    step1 = [rank01(po[j] + a * (po[j] - l1p[j])) for j in range(ns)]
                    ex_p = [rank01(step1[j] + b * (step1[j] - l2p[j])) for j in range(ns)]
                    ks_s = [ks(y, v) for v in ex_p]
                    d = [x - yb for x, yb in zip(ks_s, b_ks)]
                    nm = f"{bname}|{r1}:{k1}a{a}|{r2}:{k2}b{b}"
                    objs[nm] = (ex, ext)
                    rows.append({"name": nm, "base": bname, "leg1": f"{r1}:{k1}", "leg2": f"{r2}:{k2}",
                                 "ratio": f"{a}/{b}", "ks": float(ks(y, ex)),
                                 "auc": float(roc_auc_score(y, ex)),
                                 "ks_base": float(ks(y, bo)), "auc_base": b_auc,
                                 "worst_delta": float(min(d)), "mean_delta": float(np.mean(d)),
                                 "n_pos": int(sum(x > 0 for x in d)),
                                 "robust": int(sum(x > 0 for x in d)) >= ns - 1})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gradient_v63_multirun.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"rows {len(df)} | robust {int(df.robust.sum())} "
          f"({100 * df.robust.mean():.1f}%) | best KS {df.ks.max():.6f}")
    print("\n=== top 25 by bagged KS ===")
    print(df.sort_values("ks", ascending=False).head(25).to_string(index=False))
    print("\n=== top 15 robust by worst per-seed delta ===")
    print(df[df.robust].sort_values(["worst_delta", "ks"], ascending=False)
          .head(15).to_string(index=False))

    # candidates: top by KS, then most robust, then a consensus, then anchor mixes
    by_ks = list(df.sort_values("ks", ascending=False).name)
    by_rob = list(df[df.robust].sort_values(["worst_delta", "ks"], ascending=False).name)
    picks, seen = [], []
    for nm in by_ks[:14] + by_rob[:14] + by_ks[:30]:
        t = objs[nm][1]
        if all(abs(spearmanr(t, q).statistic - 1) > 1e-9 for q in seen):
            picks.append(nm)
            seen.append(t)
        if len(picks) >= 8:
            break

    cands = {}
    for i, nm in enumerate(picks, 1):
        cands[f"cand{i:02d}"] = objs[nm]
    con_o = np.mean([rank01(objs[nm][0]) for nm in picks[:5]], axis=0)
    con_t = np.mean([rank01(objs[nm][1]) for nm in picks[:5]], axis=0)
    cands["consensus"] = (con_o, con_t)
    con3_o = np.mean([rank01(objs[nm][0]) for nm in picks[:3]], axis=0)
    con3_t = np.mean([rank01(objs[nm][1]) for nm in picks[:3]], axis=0)
    cands["consensus3"] = (con3_o, con3_t)

    anchor = rank01(pd.read_csv(ANCHOR).set_index("application_id")
                    .reindex(pd.Index(te)).default_probability.values)
    for w in (0.35, 0.6):
        cands[f"anchor{int(w * 100)}_consensus"] = (np.zeros(0), rank01((1 - w) * anchor + w * con_t))

    out = []
    for name, (o, t) in cands.items():
        rec = {"name": name}
        if len(o):
            rec["auc"] = float(roc_auc_score(y, o))
            rec["ks"] = float(ks(y, o))
        p = OUT / f"submission_v63m_{name}.csv"
        d = pd.DataFrame({"application_id": np.asarray(te), "default_probability": rank01(t)})
        d.to_csv(p, index=False)
        rec["file"] = p.name
        rec["spearman_anchor"] = float(spearmanr(t, anchor).statistic)
        rec["problems"] = validate(d, te)
        out.append(rec)
    out.sort(key=lambda r: r.get("ks", -1), reverse=True)
    with open(OUT / "summary_v63_multirun.json", "w") as fh:
        json.dump({"anchor_public": 0.56312, "record_before": 0.528717,
                   "seeds": seeds, "candidates": out}, fh, indent=2)
    print("\n=== candidates ===")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
