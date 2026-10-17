"""How much of the OOF KS signal is real, and how much is the argmax of noise.

This decides the whole v64 strategy, so it is measured rather than argued.

Three questions, all answered from stored predictions without retraining:

1. Sampling noise. KS is a maximum over thresholds of a cumulative sum, so its
   estimate on a finite sample is itself noisy. Resampling rows of a fixed
   prediction vector measures that noise directly, separately for 70000 rows
   (the OOF size) and 30000 rows (the public size).
2. Selection inflation. If every candidate has the same true quality and the
   measurements differ only by noise, then picking the argmax over N candidates
   buys `sigma * E[max of N standard normals]` in measured KS and nothing at
   all in true quality. That number is the real cost of choosing candidates by
   OOF KS.
3. Whether the gains reported for v61, v62 and v63 survive. If the gain is
   smaller than the inflation it is indistinguishable from picking noise.

Run: python3 agas/v64/diagnose_selection_noise.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm

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
OUT = paths.version_output_of("v64")
PUBLIC_N = 30000
TRIALS = 400

def rank01(v):
    from scipy.stats import rankdata
    return rankdata(v) / len(v)


def ks_cols(P, y):
    """KS from a single sorted probability vector, matching the competition definition."""
    o = np.argsort(-P, axis=0)
    yy = y[o]
    npos = yy.sum()
    return np.max(np.cumsum(yy) / npos - np.cumsum(1 - yy) / (len(yy) - npos))


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    z63 = np.load(paths.version_output_of("v63") / "preds_v63.npz", allow_pickle=True)
    z61 = np.load(paths.version_output_of("v61") / "preds_v61.npz", allow_pickle=True)
    z62 = np.load(paths.version_output_of("v62") / "preds_v62.npz", allow_pickle=True)
    tr = z63["train_ids"]
    ymap = pd.read_csv(ROOT / "data/train_applications.csv",
                       usecols=["application_id", "target"])
    y = ymap.set_index("application_id").reindex(pd.Index(tr))["target"].values.astype(int)
    n_all = len(y)
    n_pos = int(y.sum())

    report = {}

    # ---------------------------------------------------------------- 1
    print("=" * 74)
    print("1. SAMPLING NOISE OF KS: one fixed prediction vector, rows resampled")
    print("=" * 74)
    print(f"  {n_all} OOF rows, {n_pos} positives, base rate {n_pos / n_all:.4f}")
    print(f"  Public split is {PUBLIC_N} rows with the same base rate, so about "
          f"{round(PUBLIC_N * n_pos / n_all)} positives.\n")
    vecs = {
        "v63 sel200": rank01(z63["oof_sel200"]),
        "v62 lgb_top": rank01(z62["oof_lgb_top"]),
        "v61 lgb_full": rank01(z61["oof_lgb_full"]),
    }
    noise = {}
    sizes = [(n_all, "n70000"), (PUBLIC_N, "n30000"), (9000, "n9000")]
    for name, v in vecs.items():
        full = ks_cols(v, y)
        row = {"full_oof_ks": float(full)}
        print(f"  {name}")
        for n, tag in sizes:
            if n == n_all:
                # sampling every row of the same size is the full vector, so use a
                # with-replacement bootstrap, which has the same limiting spread
                vals = np.array([ks_cols(v[rng.integers(0, n_all, n_all)], y[rng.integers(0, n_all, n_all)])
                                 for _ in range(TRIALS)])
                kind = "bootstrap"
            else:
                vals = np.array([ks_cols(v[idx], y[idx])
                                 for idx in (rng.choice(n_all, n, replace=False) for _ in range(TRIALS))])
                kind = "subsample"
            row[f"mean_{tag}"] = float(vals.mean())
            row[f"std_{tag}"] = float(vals.std())
            row[f"range_{tag}"] = [float(vals.min()), float(vals.max())]
            print(f"    {kind:10s} n={n:6d}  mean {vals.mean():.6f}  sd {vals.std():.6f}  "
                  f"range {vals.min():.6f} to {vals.max():.6f}")
        noise[name] = row
        print()

    sigma_oof = float(np.mean([noise[k]["std_n70000"] for k in noise]))
    sigma_pub = float(np.mean([noise[k]["std_n30000"] for k in noise]))
    sigma_9k = float(np.mean([noise[k]["std_n9000"] for k in noise]))
    print(f"  mean sampling sd at OOF size    : {sigma_oof:.6f}")
    print(f"  mean sampling sd at public size : {sigma_pub:.6f}")
    print(f"  mean sampling sd at 9000 rows   : {sigma_9k:.6f}")
    print(f"  public noise is {sigma_pub / sigma_oof:.1f}x the OOF sampling noise, so a")
    print("  KS computed on 30000 rows cannot resolve the quality differences that")
    print("  OOF on 70000 rows is trying to measure.")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 74)
    print("2. COST OF CHOOSING CANDIDATES BY OOF KS")
    print("=" * 74)
    print("  Each simulated candidate has identical true quality. Only its measured")
    print("  OOF KS differs, by the sampling noise measured above. The gap between")
    print("  the winner's measured KS and the common true value is pure selection bias.\n")
    print(f"  {'N candidates':>12} {'E[argmax] gain':>15} {'sd of winner':>14}")
    infl = {}
    for N in (10, 30, 64, 1296):
        emax = float(np.mean(np.max(rng.standard_normal((20000, N)), axis=1)))
        infl[N] = float(emax * sigma_oof)
        print(f"  {N:>12} {infl[N]:>15.6f} {sigma_oof:>14.6f}")
    print(f"\n  OOF noise sd is {sigma_oof:.6f}. Taking the argmax over the ~1300 "
          f"grid points searched in v63")
    print(f"  inflates the measured number by roughly {infl[1296]:.6f} for free.")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 74)
    print("3. DO THE REPORTED v61 TO v63 GAINS SURVIVE?")
    print("=" * 74)
    steps = [
        ("v61 best", 0.528624),
        ("v62 best", 0.528717),
        ("v63 best", 0.529867),
    ]
    print(f"\n  {'step':>12} {'KS':>10} {'gain':>10} {'gain / sd':>11}")
    prev = None
    gains = []
    for name, k in steps:
        g = "" if prev is None else f"{k - prev:+.6f}"
        ratio = "" if prev is None else f"{(k - prev) / sigma_oof:.2f}"
        if prev is not None:
            gains.append(k - prev)
        print(f"  {name:>12} {k:>10.6f} {g:>10} {ratio:>11}")
        prev = k
    print(f"\n  typical step gain {np.mean(gains):+.6f}, which is "
          f"{np.mean(gains) / sigma_oof:.2f} sampling sd.")
    print(f"  The full v61 to v63 improvement is {steps[-1][1] - steps[0][1]:+.6f}, "
          f"{(steps[-1][1] - steps[0][1]) / sigma_oof:.2f} sd.")
    print(f"  The public score gap that must be closed is "
          f"{0.57 - 0.56312:.5f}, which is {(0.57 - 0.56312) / sigma_pub:.2f} public sd.")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 74)
    print("4. WHAT DIVERSITY BUYS: EXPECTED MAXIMUM OVER N INDEPENDENT SUBMITS")
    print("=" * 74)
    print("  A resubmission of the same file scores the same. Only a genuinely")
    print("  different rank vector gives an independent draw from the noise.\n")
    print(f"  {'slots':>7} {'E[max] over noise sd':>21}")
    for N in (1, 3, 10, 20, 39):
        emax = 0.0 if N == 1 else float(np.mean(np.max(rng.standard_normal((20000, N)), axis=1)))
        print(f"  {N:>7} {emax:>21.3f} sd  (+{emax * sigma_pub:.5f} public)")
    print(f"\n  With {sigma_pub:.6f} public sd, the choice between one submit and")
    print(f"  thirty-nine diverse submits is worth about "
          f"{(float(np.mean(np.max(rng.standard_normal((20000, 39)), axis=1))) * sigma_pub):.5f} public,")
    print("  which is larger than every model improvement achieved so far.")

    payload = {"sigma_oof": sigma_oof, "sigma_public": sigma_pub,
               "noise": noise, "inflation": infl,
               "reported_steps": steps, "n_pos": n_pos, "n_rows": n_all}
    (OUT / "summary_noise_diagnosis.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {OUT / 'summary_noise_diagnosis.json'}")


if __name__ == "__main__":
    main()
