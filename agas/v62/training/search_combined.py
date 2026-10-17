"""Combined v61 + v62 extrapolation search.

v62 has the better ranking model (`lgb_top` at 200 gain-selected columns,
bagged AUC `0.847419` against `0.846647` for v61) but its best extrapolation
reaches KS `0.5279` against v61's `0.528624`. That gap is smaller than the OOF
KS standard error, so neither number is evidence on its own.

The two ensembles were trained from different feature blocks, three versus four
fold seeds, and different feature-selection policies, so their errors are only
partly shared. This script puts both on one rank scale and searches:

- bases built only from v62, where the per-fold-seed KS delta is available;
- bases that mix the v61 and v62 cores, where per-seed evaluation is not
  available because v61 did not store per-seed OOF, so those are judged by
  resampling rows instead: KS is recomputed on eight random halves of the OOF
  rows and the mean and worst half are reported.

Legs come from both runs, including `v61 fac_full`, which is the leg behind the
first public-proven direction. The already-proven v61 blends are carried as
reference rows so the combined base can be compared against them directly.
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

V61_KEYS = ["lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear", "fac_full"]
V62_KEYS = ["lgb_full", "lgb_top", "lgb_topwide", "lgb_fac", "lgb_lean",
            "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"]
RATIOS = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5]
V61_TREES = ["lgb_full", "lgb_lean", "cat_full", "xgb_full"]
V62_TREES = ["lgb_full", "lgb_top", "cat_full", "xgb_full"]


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def main():
    os.chdir(ROOT)
    z61 = np.load(paths.version_output_of("v61") / "preds_v61.npz", allow_pickle=True)
    z62 = np.load(OUT / "preds_v62.npz", allow_pickle=True)
    seeds = [int(s) for s in z62["seeds"]]
    tr, te = z61["train_ids"], z61["test_ids"]
    y = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    y = y.set_index("application_id").loc[tr].target.values.astype(int)

    B61 = {k: rank01(z61[f"oof_{k}"]) for k in V61_KEYS}
    T61 = {k: rank01(z61[f"test_{k}"]) for k in V61_KEYS}
    B62 = {k: rank01(z62[f"oof_{k}"]) for k in V62_KEYS}
    T62 = {k: rank01(z62[f"test_{k}"]) for k in V62_KEYS}
    P62 = {k: [rank01(z62[f"oofseed_{k}_{j}"]) for j in range(len(seeds))] for k in V62_KEYS}

    rng = np.random.RandomState(11)
    halves = [rng.permutation(len(y))[: len(y) // 2] for _ in range(8)]

    def half_stats(v):
        h = [ks(y[i], v[i]) for i in halves]
        return float(np.mean(h)), float(np.min(h))

    def core(run, trees, w_lin):
        B, T = (B62, T62) if run == 62 else (B61, T61)
        spec = [(k, (1 - w_lin) / 3.0 / len(trees)) for k in trees]
        spec += [("lgbx_full", (1 - w_lin) / 3.0), ("linear", w_lin)]
        bo = sum(x * B[k] for k, x in spec)
        bt = sum(x * T[k] for k, x in spec)
        if run == 62:
            po = [sum(x * P62[k][j] for k, x in spec) for j in range(len(seeds))]
        else:
            po = None
        return bo, bt, po

    bases = {}
    for w in (0.1, 0.2, 0.3):
        bases[f"v62_w{w}"] = core(62, V62_TREES, w)
        bases[f"v61_w{w}"] = core(61, V61_TREES, w)
        a, at, ap = core(62, V62_TREES, w)
        b, bt, _ = core(61, V61_TREES, w)
        bases[f"mix_w{w}"] = (0.5 * a + 0.5 * b, 0.5 * at + 0.5 * bt, None)

    legs = [("62", k) for k in ["lgb_faclean", "lgb_fac", "lgb_lean", "lgb_topwide", "lgb_top"]] + \
           [("61", k) for k in ["fac_full", "lgb_lean"]]

    def leg_vec(run, k):
        if run == "62":
            return B62[k], T62[k], P62[k]
        return B61[k], T61[k], None

    rows, objs = [], {}
    for bname, (bo, btt, po) in bases.items():
        hm, hn = half_stats(bo)
        b_ks_seed = [ks(y, v) for v in po] if po else None
        for run, leg in legs:
            lb, lt, lp = leg_vec(run, leg)
            for r in RATIOS:
                ex = rank01(bo + r * (bo - lb))
                ext = rank01(btt + r * (btt - lt))
                ehm, ehn = half_stats(ex)
                rec = {
                    "name": f"{bname}-{run}:{leg}_r{r}", "base": bname,
                    "leg": f"{run}:{leg}", "ratio": r,
                    "ks": float(ks(y, ex)), "auc": float(roc_auc_score(y, ex)),
                    "ks_base": float(ks(y, bo)), "auc_base": float(roc_auc_score(y, bo)),
                    "half_mean": ehm, "half_min": ehn,
                    "half_mean_delta": ehm - hm, "half_min_delta": ehn - hn,
                }
                if po is not None and lp is not None:
                    ks_s = [ks(y, rank01(po[j] + r * (po[j] - lp[j]))) for j in range(len(seeds))]
                    d = [a - b for a, b in zip(ks_s, b_ks_seed)]
                    rec["worst_delta"] = float(min(d))
                    rec["mean_delta"] = float(np.mean(d))
                    rec["n_pos"] = int(sum(x > 0 for x in d))
                    rec["robust"] = rec["n_pos"] >= len(seeds) - 1
                else:
                    rec["worst_delta"] = np.nan
                    rec["mean_delta"] = np.nan
                    rec["n_pos"] = -1
                    rec["robust"] = rec["half_min_delta"] > 0
                objs[rec["name"]] = (ex, ext)
                rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gradient_combined.csv", index=False)
    pd.set_option("display.width", 250)

    seed_ok = df[df.n_pos >= 0]
    print("=== v62-only bases, per-fold-seed robustness, top 15 ===")
    print(seed_ok.sort_values(["robust", "worst_delta", "ks"], ascending=[False, False, False])
          .head(15).to_string(index=False))

    print("\n=== all bases sorted by bagged KS, top 15 ===")
    print(df.sort_values("ks", ascending=False).head(15).to_string(index=False))

    print("\n=== mixed bases sorted by half_min_delta then KS, top 15 ===")
    print(df[df.base.str.startswith("mix")].sort_values(["half_min_delta", "ks"], ascending=False)
          .head(15).to_string(index=False))

    np.savez_compressed(OUT / "combined_objs.npz", train_ids=tr, test_ids=te,
                        **{k: v[0] for k, v in objs.items()},
                        **{f"test_{k}": v[1] for k, v in objs.items()})
    with open(OUT / "summary_gradient_combined.json", "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
