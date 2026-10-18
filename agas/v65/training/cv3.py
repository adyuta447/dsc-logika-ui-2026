"""Average a CV configuration over several fold seeds.

The single-seed harness number moves by about 0.0006 AUC from fold seed alone,
which is the same size as the effect any of these feature blocks is supposed to
have. Every comparison from here on therefore runs three fold seeds and reports
the mean and the spread, so a block is only accepted when it wins on the mean and
does not lose on spread.
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARNESS = ROOT / "agas/shared/cv_harness.py"
SEEDS = [42, 7, 13]


def run(tag, fold_seeds=SEEDS, **kw):
    # the name must not be `seeds`: that is a LightGBM argument name, and a
    # clash silently retargets the fold-seed loop onto the model seeds
    parts = []
    for s in fold_seeds:
        cmd = [sys.executable, str(HARNESS), "--fold-seed", str(s), "--tag", f"{tag}_s{s}"]
        for k, v in kw.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"{tag} seed {s} failed")
        j = json.loads(r.stdout[r.stdout.index("{"):])
        parts.append(j)
    out = {
        "tag": tag,
        "auc_mean": statistics.mean(p["auc"] for p in parts),
        "auc_sd": statistics.pstdev([p["auc"] for p in parts]),
        "ks_mean": statistics.mean(p["ks"] for p in parts),
        "ks_sd": statistics.pstdev([p["ks"] for p in parts]),
        "n_features": parts[0]["n_features"],
        "auc_per_seed": [round(p["auc"], 6) for p in parts],
        "minutes": sum(p["minutes"] for p in parts),
    }
    print(f"{out['tag']:28s} n={out['n_features']:4d} "
          f"auc={out['auc_mean']:.6f} +-{out['auc_sd']:.6f}  "
          f"ks={out['ks_mean']:.6f} +-{out['ks_sd']:.6f}  "
          f"({out['minutes']:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    tag = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    for k, v in list(kw.items()):
        if v in ("True", "False"):
            kw[k] = v == "True"
    run(tag, **kw)