"""Aggregate a partial v69 run without waiting for every fold seed.

The main trainer writes every completed fold seed to ``preds_v69_partial.npz``.
This script averages whatever is already there, so a run can be stopped early
without losing work. It is also the fallback if the machine has to be freed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

OUT = Path(__file__).resolve().parents[1] / "output"
NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default=str(OUT / "preds_v69_partial.npz"))
    ap.add_argument("--features", default=str(OUT / "features_v69.pkl"))
    ap.add_argument("--out-prefix", default="v69")
    return ap.parse_args()


def ks_score(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def main():
    args = parse()
    p = np.load(args.partial, allow_pickle=True)
    done = [int(s) for s in p["done"]]
    if not done:
        raise SystemExit("partial file has no completed fold seeds")
    variants = [str(v) for v in p["variants"]]

    f = pd.read_pickle(args.features)
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    y = tr.target.astype(int).values

    oof = {v: np.mean([p[f"seed_{s}_oof_{v}"] for s in done], axis=0)
           for v in variants}
    test = {v: np.mean([p[f"seed_{s}_test_{v}"] for s in done], axis=0)
            for v in variants}

    ids = {"application_id", "customer_id", "target", "decision_datetime"}
    drops = ("cf_", "pv_")
    n_lean = len([c for c in tr.columns if c not in ids
                  and c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)
                  and not c.startswith(drops)])
    summary = {
        "features": args.features,
        "variants": variants,
        "folds": int(p["folds"]),
        "fold_seeds": done,
        "lgb_seeds": [int(s) for s in p["lgb_seeds"]],
        "use_factor": bool(p["use_factor"]),
        "n_full": int(len([c for c in tr.columns if c not in ids])),
        "n_lean": int(n_lean),
        "drops": list(drops),
        "minutes": None,
        "partial": True,
        "per_seed": {
            str(s): {v: {"auc": float(roc_auc_score(y, p[f"seed_{s}_oof_{v}"])),
                        "ks": ks_score(y, p[f"seed_{s}_oof_{v}"])}
                     for v in variants}
            for s in done
        },
    }
    for v in variants:
        summary[v] = {"auc": float(roc_auc_score(y, oof[v])),
                      "ks": ks_score(y, oof[v])}

    np.savez_compressed(
        OUT / f"preds_{args.out_prefix}.npz",
        train_ids=tr.application_id.values, test_ids=te.application_id.values,
        y=y, **{f"oof_{v}": a for v, a in oof.items()},
        **{f"test_{v}": a for v, a in test.items()},
        **{f"seed_{s}_oof_{v}": p[f"seed_{s}_oof_{v}"] for s in done for v in variants},
        **{f"seed_{s}_test_{v}": p[f"seed_{s}_test_{v}"] for s in done for v in variants})
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))

    print(f"aggregated {len(done)} seeds: {done}")
    for v in variants:
        print(f"BAG {v:12s} auc={summary[v]['auc']:.6f} ks={summary[v]['ks']:.6f}")


if __name__ == "__main__":
    main()
