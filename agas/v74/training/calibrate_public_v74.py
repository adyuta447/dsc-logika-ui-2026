"""Calibrate the public leaderboard sample size, then size the noise.

Every version from v67 on has treated the public spread as sd 0.0053 and the
gap to the leader as real signal. Both numbers are assumptions. One fact
contradicts them: anchor OOF KS is 0.528713 at n=70000 but its public score is
0.56424. KS is a maximum statistic, so it is biased upward at small n. The
+0.0355 offset is therefore a measurement of how small the public split is.

This script inverts that: subsample the anchor's own OOF scores at many sizes,
measure E[KS] and sd[KS] at each size, and read off the size whose expected KS
matches the observed public score. The sd at that size is the real noise floor,
which decides whether chasing 0.578 is a modelling problem or a draw problem.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
OUT = HERE.parent / "output"

PUBLIC_OBSERVED = 0.56424          # submission_v66_04, the anchor
N_TEST = 30000
DRAWS = 600
SIZES = (750, 1000, 1500, 2000, 2500, 3000, 4500, 6000, 9000, 15000, 21000, 30000)


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    """Two-sample KS == max |TPR - FPR| over thresholds, on ranks."""
    o = np.argsort(score)
    yy = y[o]
    pos = np.cumsum(yy) / max(yy.sum(), 1)
    neg = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return float(np.abs(pos - neg).max())


def anchor_oof(z, y):
    """Rebuild ext0.5|tm30|lgb_lean, the v66_04 recipe, in OOF space."""
    r = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
    trees = (r["lgb_full"] + r["lgb_lean"] + r["cat_full"] + r["xgb_full"]) / 4
    core = (trees * 2 + r["lgbx_full"]) / 3
    tm30 = 0.7 * core + 0.3 * r["linear"]
    return rank01(tm30 + 0.5 * (tm30 - r["lgb_lean"]))


def main():
    z = np.load(ROOT / "agas/v66/output/preds_v66.npz", allow_pickle=True)
    ids = z["train_ids"]
    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[ids].target.values.astype(int))

    s = anchor_oof(z, y)
    full = ks(s, y)
    print(f"anchor OOF KS at n={len(y)}: {full:.6f}")
    print(f"observed public KS: {PUBLIC_OBSERVED:.6f}  (offset {PUBLIC_OBSERVED - full:+.6f})\n")

    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    prev = y.mean()
    rows = []
    print(f"{'n':>7} {'n_pos':>6} {'n_eff':>8} {'E[KS]':>9} {'sd[KS]':>9} {'bias':>9} {'gap/sd':>8}")
    for n in SIZES:
        npos = int(round(n * prev))
        vals = np.empty(DRAWS)
        for d in range(DRAWS):
            idx = np.concatenate([rng.choice(pos, npos, replace=False),
                                  rng.choice(neg, n - npos, replace=False)])
            vals[d] = ks(s[idx], y[idx])
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        neff = npos * (n - npos) / n
        rows.append({"n": n, "n_pos": npos, "n_eff": round(neff, 1),
                     "mean_ks": round(mu, 6), "sd_ks": round(sd, 6),
                     "bias": round(mu - full, 6),
                     "public_gap_in_sd": round((0.578 - PUBLIC_OBSERVED) / sd, 2)})
        print(f"{n:>7} {npos:>6} {neff:>8.0f} {mu:9.6f} {sd:9.6f} {mu - full:+9.6f} "
              f"{(0.578 - PUBLIC_OBSERVED) / sd:8.2f}")

    # size whose expected KS best matches the single public observation
    best = min(rows, key=lambda r: abs(r["mean_ks"] - PUBLIC_OBSERVED))
    frac = best["n"] / N_TEST
    print(f"\nimplied public size ~{best['n']} rows ({frac:.0%} of test), "
          f"noise sd ~{best['sd_ks']:.5f}")
    print(f"leader 0.578 sits {best['public_gap_in_sd']:.2f} sd above our 0.56424")

    # expected maximum over k independent draws at that sd: the value of
    # submitting decorrelated near-ties instead of one anchor
    rng2 = np.random.default_rng(5)
    print("\nexpected public gain from k *independent* equal-quality submits:")
    emax = {}
    for k in (1, 2, 3, 4, 5, 6, 8):
        e = float(np.mean(np.max(rng2.standard_normal((40000, k)), axis=1)))
        emax[k] = round(e * best["sd_ks"], 5)
        print(f"  k={k}: +{e * best['sd_ks']:.5f}  ({e:.2f} sd)")

    summary = {"anchor_oof_ks": round(full, 6), "public_observed": PUBLIC_OBSERVED,
               "implied_public_n": best["n"], "implied_public_frac": round(frac, 4),
               "noise_sd": best["sd_ks"], "curve": rows, "expected_max_gain": emax}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "public_calibration_v74.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'public_calibration_v74.json'}")


if __name__ == "__main__":
    main()
