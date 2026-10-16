import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
rank = lambda v: rankdata(v) / len(v)
from common import variants
groups = {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"]}
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
v61 = pd.read_csv("agas/output/dl/submission_v61_alt_ext0.5_tm30_lean.csv").set_index("application_id").default_probability.reindex(ids).values
T = {}
for g, rs in groups.items():
    vt = [variants(r, 1) for r in rs]
    for k in vt[0]: T[f"{g}|{k}"] = rank(np.mean([v[k] for v in vt], 0))
sp = lambda a: spearmanr(v61, a).correlation
print({k: round(sp(v), 5) for k, v in sorted(T.items(), key=lambda kv: -sp(kv[1]))[:8]})
for hi in ["v4|linear30", "v4|top_mix"]:
    for lo in ["v4|lean_only", "fac|lean_only", "fac|linear30", "fac|top_mix", "v4|full_only"]:
        for a in (0.5, 1.0):
            print(hi, lo, a, round(sp(rank(T[hi] + a * (T[hi] - T[lo]))), 5))
