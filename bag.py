import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score

# Averages the top_mix and linear30 blends of several fold seed runs, scoring the bag on the averaged out of fold ranks.
runs = sys.argv[1:]
f = pd.read_parquet("features.parquet", columns=["target"])
y = f[f.target.notna()].target.astype(int).values
te = pd.read_csv(f"{runs[0]}/variants/top_mix.csv").application_id
rank = lambda v: rankdata(v) / len(v)
ks = lambda p: ks_2samp(p[y == 1], p[y == 0]).statistic


def blends(res, i):
    one = lambda m, t: res[f"{m}_{t}"] if f"{m}_{t}" in res else res[f"{m}_{t}_0"]
    trees = lambda t: (np.mean([rank(res[f"lgb_{t}_{s}"][i]) for s in range(3)], 0) + rank(one("cat", t)[i]) + rank(one("xgb", t)[i])) / 3
    core = (trees("full") + trees("lean") + rank(one("cat", "full")[i]) + rank(one("cat", "lean")[i])) / 4
    lin = rank(res["linear"][i])
    return {"top_mix": 0.8 * (core + trees("top")) / 2 + 0.2 * lin, "linear30": 0.7 * (core + trees("top")) / 2 + 0.3 * lin,
            "trees_only": (core + trees("top")) / 2}


B = {r: [blends(pd.read_pickle(f"{r}/models/predictions.pkl"), i) for i in (0, 1)] for r in runs}
for v in ["top_mix", "linear30", "trees_only"]:
    for r in runs:
        print(f"{v:9s} {r:10s} auc {roc_auc_score(y, B[r][0][v]):.5f} ks {ks(B[r][0][v]):.5f}")
    oof = np.mean([B[r][0][v] for r in runs], 0)
    print(f"{v:9s} {'bag':10s} auc {roc_auc_score(y, oof):.5f} ks {ks(oof):.5f}")
    sub = pd.DataFrame({"application_id": te, "default_probability": rank(np.mean([B[r][1][v] for r in runs], 0))})
    assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
    sub.to_csv(f"variants/bag_{runs[0]}_{v}_{len(runs)}.csv", index=False)
