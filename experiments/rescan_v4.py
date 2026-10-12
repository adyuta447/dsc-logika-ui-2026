import numpy as np, pandas as pd, sys, glob, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
S = sys.argv[1]
sys.argv = ["bag.py", "run_v4_s42"]
exec(open("bag.py").read().split("for v in")[0])
p = B["run_v4_s42"][0]["top_mix"]; lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet", columns=["application_id", "target"]); ids = pd.Index(f[f.target.notna()].application_id.astype(str).values)
tr = pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]).set_index("application_id").loc[ids]
M = pd.get_dummies(tr.decision_datetime.dt.month).values[:, 1:].astype(float)
def ll(X):
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y); q = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9); return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
X0 = np.column_stack([lo, M]); base = ll(X0); rows = []
seen = set()
for fp in sorted(glob.glob(S + "cand*.parquet")):
    c = pd.read_parquet(fp)
    if "application_id" not in c: continue
    c = c.drop_duplicates("application_id").assign(application_id=lambda d: d.application_id.astype(str)).set_index("application_id").reindex(ids)
    for col in c.columns:
        if col in seen or col == "customer_id" or not np.issubdtype(c[col].dtype, np.number): continue
        seen.add(col); v = c[col].astype(float).replace([np.inf, -np.inf], np.nan)
        if v.notna().mean() < 0.05 or v.nunique() < 2: continue
        r = v.rank(pct=True).fillna(0.5).values; na = v.isna().astype(float).values
        rows.append((col, fp.split("/")[-1], round(v.notna().mean(), 2), round(2 * (ll(np.column_stack([X0, r, r ** 2, na])) - base), 1)))
d = pd.DataFrame(rows, columns=["feature", "file", "cov", "LR"]).sort_values("LR", ascending=False)
print(len(d), "candidates; top 25 over model + month:"); print(d.head(25).to_string(index=False))
d.to_csv(S + "rescan_v4.csv", index=False)
