"""Sweep feature-selection levels and the hyperparameters that suit them.

Motivation: in v62 a 200-column gain-selected set became the strongest single
model (AUC 0.847419 against 0.846629 for all 554 columns), but it kept the
hyperparameters tuned for 554 columns. `feature_fraction` 0.3 means each tree
sees only 60 of the 200 selected columns, and `min_child_samples` 200 is sized
for a much wider feature space. If that mismatch is real, re-tuning for the
selected set should beat it.

Every configuration is averaged over fold seeds 42, 7 and 13 so the comparison
is not decided by one split. Selection is per fold on that fold's training rows.
"""

import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "agas" / "shared" / "harness_output"
OUT.mkdir(parents=True, exist_ok=True)

BASE = ["--lr", "0.01", "--l2", "5", "--seeds", "0", "--folds", "5", "--drop", "gn_"]
SEEDS = ["42", "7", "13"]

CONFIGS = [
    ("ctl_notop_ff03", ["--ff", "0.3", "--leaves", "15", "--mcs", "200"]),
    ("t200_ff03", ["--top-n", "200", "--ff", "0.3", "--leaves", "15", "--mcs", "200"]),
    ("t200_ff06", ["--top-n", "200", "--ff", "0.6", "--leaves", "15", "--mcs", "200"]),
    ("t200_ff09", ["--top-n", "200", "--ff", "0.9", "--leaves", "15", "--mcs", "200"]),
    ("t200_ff06_mcs60", ["--top-n", "200", "--ff", "0.6", "--leaves", "15", "--mcs", "60"]),
    ("t200_ff06_lv31", ["--top-n", "200", "--ff", "0.6", "--leaves", "31", "--mcs", "200"]),
    ("t200_ff06_lv31_mcs60", ["--top-n", "200", "--ff", "0.6", "--leaves", "31", "--mcs", "60"]),
    ("t150_ff07", ["--top-n", "150", "--ff", "0.7", "--leaves", "15", "--mcs", "120"]),
    ("t300_ff05", ["--top-n", "300", "--ff", "0.5", "--leaves", "15", "--mcs", "200"]),
    ("t400_ff04", ["--top-n", "400", "--ff", "0.4", "--leaves", "15", "--mcs", "200"]),
    ("t100_ff09", ["--top-n", "100", "--ff", "0.9", "--leaves", "15", "--mcs", "120"]),
    ("t250_ff07_lv23", ["--top-n", "250", "--ff", "0.7", "--leaves", "23", "--mcs", "120"]),
    ("t200_ff06_l2_2", ["--top-n", "200", "--ff", "0.6", "--leaves", "15", "--mcs", "200",
                        "--l2", "2"]),
    ("t200_ff06_lr02", ["--top-n", "200", "--ff", "0.6", "--leaves", "15", "--mcs", "200",
                        "--lr", "0.02"]),
]


def main():
    only = sys.argv[1:] or None
    rows = []
    for name, extra in CONFIGS:
        if only and name not in only:
            continue
        agg = {"auc": [], "ks": []}
        t0 = time.time()
        for fs in SEEDS:
            tag = f"sweep_{name}_fs{fs}"
            cmd = ["python3", "agas/shared/cv_harness.py", "--tag", tag,
                   "--fold-seed", fs, *BASE, *extra]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print("FAIL", name, fs, r.stderr.strip()[-200:], flush=True)
                break
            d = json.loads((OUT / f"harness_{tag}.json").read_text())
            agg["auc"].append(d["auc"])
            agg["ks"].append(d["ks"])
        if len(agg["ks"]) == len(SEEDS):
            import statistics as st
            rec = {
                "name": name,
                "auc": sum(agg["auc"]) / len(agg["auc"]),
                "ks": sum(agg["ks"]) / len(agg["ks"]),
                "ks_sd": st.pstdev(agg["ks"]),
                "ks_per_seed": [round(k, 5) for k in agg["ks"]],
                "minutes": round((time.time() - t0) / 60, 1),
            }
            rows.append(rec)
            print(f"{name:24s} auc={rec['auc']:.6f} ks={rec['ks']:.6f} "
                  f"sd={rec['ks_sd']:.5f} {rec['minutes']:.1f}m {rec['ks_per_seed']}", flush=True)
            (ROOT / "agas" / "shared" / "harness_output" / "sweep_v63.json").write_text(
                json.dumps(rows, indent=2))

    rows.sort(key=lambda r: -r["ks"])
    print("\n=== ranked by KS ===")
    for r in rows:
        print(f"{r['name']:24s} auc={r['auc']:.6f} ks={r['ks']:.6f} sd={r['ks_sd']:.5f}")
    rows.sort(key=lambda r: -r["auc"])
    print("\n=== ranked by AUC ===")
    for r in rows:
        print(f"{r['name']:24s} auc={r['auc']:.6f} ks={r['ks']:.6f} sd={r['ks_sd']:.5f}")


if __name__ == "__main__":
    main()
