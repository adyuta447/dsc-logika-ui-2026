import sys, numpy as np, pandas as pd
sys.path.insert(0, "work"); from common import variants, rank
from sklearn.metrics import roc_auc_score
# Screens candidate features for information beyond bag v4 top_mix: AUC of the feature inside 40 bins of the OOF prediction, averaged.
# A strong feature the model already uses scores about 0.5. Usage: python experiments/cond_screen.py work/oth_cand.parquet
f = pd.read_parquet("features.parquet", columns=["application_id", "target"]); tr = f[f.target.notna()].reset_index(drop=True); y = tr.target.astype(int).values
oof = rank(np.mean([variants(r, 0)["top_mix"] for r in ["run_v4_s42", "run_v4_s7", "run_v4_s13"]], 0))
C = tr[["application_id"]].merge(pd.read_parquet(sys.argv[1]), on="application_id", how="left")
bins = pd.qcut(oof, 40, labels=False)
def cauc(x):
    num = den = 0
    for b in range(40):
        m = (bins == b) & ~np.isnan(x)
        if m.sum() < 50 or y[m].min() == y[m].max(): continue
        num += m.sum() * (roc_auc_score(y[m], x[m]) - 0.5); den += m.sum()
    return 0.5 + num / den
rows = []
for c in C.columns.drop("application_id"):
    x = C[c].astype(float).values; ok = ~np.isnan(x); a = cauc(x)
    se = 0.5 / np.sqrt(ok.sum() * y.mean() * (1 - y.mean()) * 3)
    rows.append((c, round(roc_auc_score(y[ok], x[ok]), 4), round(a, 4), round((a - 0.5) / se, 1), round(ok.mean(), 2)))
r = pd.DataFrame(rows, columns=["fitur", "AUC_sendiri", "AUC_kondisional", "z", "cakupan"]).sort_values("z", key=abs, ascending=False)
print(r.to_string(index=False))
