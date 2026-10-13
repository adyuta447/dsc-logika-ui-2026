import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
# Where is the lucky file better than the reference on all 70000 train rows? AUC change split into within and between segment pairs.
os.environ["N"] = "4000"
src = open(sys.argv[1] + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
L, R = O["ext1.5|v4-fac|full_only"], O["v4|top_mix"]
f = pd.read_parquet("features.parquet"); tr = f[f.target.notna()].reset_index(drop=True); assert (tr.target.values == y).all()
q = lambda c, k=5: pd.qcut(tr[c].rank(method="first"), k, labels=False)
seg = {"grade": tr.interest_rate.round(3), "purpose": tr.loan_purpose, "region": tr.region, "employment": tr.employment_type, "housing": tr.housing_status,
       "age": q("age"), "income": q("declared_monthly_income"), "ec_richness": q("ec_n_base"), "repeat": tr.rh_last_loan_late.notna()}
def parts(s, idx):
    yy, out = y[idx], {}
    for name, p in (("L", L), ("R", R)):
        pp, ss = p[idx], s[idx]; P, N = yy.sum(), len(yy) - yy.sum()
        tot = roc_auc_score(yy, pp) * P * N; win = wp = 0
        for g in np.unique(ss):
            m = ss == g; Pg, Ng = yy[m].sum(), m.sum() - yy[m].sum()
            if Pg and Ng: win += roc_auc_score(yy[m], pp[m]) * Pg * Ng; wp += Pg * Ng
        out[name] = (win / wp, (tot - win) / (P * N - wp), tot / (P * N))
    return [out["L"][i] - out["R"][i] for i in range(3)]
rng = np.random.default_rng(0); chunks = np.array_split(rng.permutation(len(y)), 10)
print(f"all rows: AUC lucky {roc_auc_score(y, L):.5f} reference {roc_auc_score(y, R):.5f}")
for k, s in seg.items():
    s = pd.Series(s).astype(str).values; full = parts(s, np.arange(len(y))); se = np.std([parts(s, c) for c in chunks], 0) / np.sqrt(10)
    print(f"{k:12s} within {full[0]:+.5f} (se {se[0]:.5f})  between {full[1]:+.5f} (se {se[1]:.5f})  total {full[2]:+.5f}")
# per segment level, within AUC change and the mean rank shift the lucky file gives that level, next to its default rate
for k in ("grade", "ec_richness", "repeat", "employment"):
    s = pd.Series(seg[k]).astype(str).values
    rows = [(g, (s == g).sum(), y[s == g].mean(), roc_auc_score(y[s == g], L[s == g]) - roc_auc_score(y[s == g], R[s == g]), (L - R)[s == g].mean()) for g in np.unique(s)]
    print(k, pd.DataFrame(rows, columns=["level", "n", "default", "dAUC within", "rank shift"]).round(5).to_string(index=False))
