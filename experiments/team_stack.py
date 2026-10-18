import sys, numpy as np, pandas as pd
sys.path.insert(0, "work"); from common import variants, rank
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# Final team model: logistic regression on the OOF ranks of our bag (v4, fac) and Agas's notebook components (v61..v64), same as team_stack in v61_plan.py.
# Writes variants/team_stack.csv. Its CV score uses weights fit inside folds.
f = pd.read_parquet("features.parquet", columns=["target"]); y = f[f.target.notna()].target.astype(int).values
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
O, T = {}, {}
for g, rs in {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"]}.items():
    vo = [variants(r, 0) for r in rs]; vt = [variants(r, 1) for r in rs]
    for k in vo[0]: O[f"{g}|{k}"] = rank(np.mean([v[k] for v in vo], 0)); T[f"{g}|{k}"] = rank(np.mean([v[k] for v in vt], 0))
for v in ("v61", "v62", "v63", "v64"):
    z = np.load(f"agas/{v}/output/preds_{v}.npz", allow_pickle=False)
    for k in [k[4:] for k in z.keys() if k.startswith("oof_")]: O[f"a{v}|{k}"] = rank(z["oof_" + k]); T[f"a{v}|{k}"] = rank(z["test_" + k])
X, XT = np.column_stack(list(O.values())), np.column_stack(list(T.values())); p = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y): p[b] = LogisticRegression(max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
print(f"{len(O)} komponen, AUC CV {roc_auc_score(y, p):.5f} (bag v4 top_mix 0.84774)")
t = rank(LogisticRegression(max_iter=2000).fit(X, y).decision_function(XT))
test = pd.read_csv("data/test_applications.csv").application_id
d = pd.DataFrame({"application_id": ids, "default_probability": t})
assert len(d) == 30000 and d.application_id.is_unique and set(d.application_id) == set(test) and d.default_probability.between(0, 1).all()
d.to_csv("variants/team_stack.csv", index=False)
print("spearman vs bag v4 top_mix", round(np.corrcoef(t, T["v4|top_mix"])[0, 1], 5), " vs v61", round(np.corrcoef(t, rank(pd.read_csv("agas/output/dl/submission_v61_alt_ext0.5_tm30_lean.csv").set_index("application_id").default_probability.reindex(ids).values))[0, 1], 5))
