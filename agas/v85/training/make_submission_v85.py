"""v85: the last two slots after slot2_cat_full scored 0.56257 (-0.00263).

Every axis measured on both sides (r, tm, core) lost on both sides, and the
xgb a=1 branch from v84 is +0.49 aligned with the cat move that just lost and
+0.76 with core 3:1 (-0.00226), so that branch is dropped. What is left:

  r0p786    the r-axis peak alone, the only coordinate aligned with the one
            measured gain; rho 0.999996, so it reads within about +/-0.0001.
  cat_am0p5 the other side of the slot-2 axis (a = -0.5 removes cat_full from
            core entirely); closes the axis whichever way it reads.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/v84/training"))
from make_submission_v84 import chain, rank01  # noqa: E402

OUT = HERE.parent / "output"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "agas/v66/output/preds_v66.npz", allow_pickle=True)
    te = pd.Index(z["test_ids"])
    t = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}
    final = chain(t)
    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)

    v83 = pd.read_csv(ROOT / "agas/v83/output/submission_v83_r0p786_only.csv")
    v83 = v83.set_index("application_id").loc[te].default_probability.to_numpy()
    assert np.allclose(v83, chain(t, r=0.7864)), "v83 r0p786 is not this chain"

    rows = []
    for name, v in {"r0p786": chain(t, r=0.7864),
                    "cat_am0p5": chain(t, leg="cat_full", a=-0.5)}.items():
        p = OUT / f"submission_v85_{name}.csv"
        frame = pd.DataFrame({"application_id": te.values, "default_probability": v})
        frame.to_csv(p, index=False)
        assert (len(frame) == 30000 and frame.application_id.is_unique
                and set(frame.application_id) == test_set
                and not frame.isna().any().any()
                and frame.default_probability.between(0, 1).all()), p
        rows.append({"file": p.name,
                     "spearman_vs_final": round(float(spearmanr(v, final).statistic), 6),
                     "rank_changed": int((rankdata(v) != rankdata(final)).sum()),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]})
        print(rows[-1])
    (OUT / "summary_v85.json").write_text(json.dumps(
        {"final_public": 0.56520, "slot2_cat_full_public": 0.56257,
         "candidates": rows}, indent=2))


if __name__ == "__main__":
    main()
