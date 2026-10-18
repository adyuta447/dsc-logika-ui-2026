import os, sys
import numpy as np, pandas as pd
from scipy.special import ndtr
sys.path.insert(0, "work"); from common import variants, rank

# Scores files around Agas's v61 (public 0.56312) on simulated public splits conditioned on every public score seen,
# like harvest_kf.py. Files with only test predictions (Agas's, and ours whose OOF was lost in the 16 Oct reboot) get a pseudo OOF:
# their test ranks projected on our notebook components, plus independent noise as large as what the projection misses.
f = pd.read_parquet("features.parquet", columns=["target"]); y = f[f.target.notna()].target.astype(int).values
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
def comps(r, i):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    c = {f"{m}_{t}": np.mean([rank(res[k][i]) for k in res if k.startswith(f"{m}_{t}")], 0) for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    return c | {"linear": rank(res["linear"][i])}
groups = {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"], "fac3": ["run_fac", "run_s7", "run_s13"]}
O, T, CO, CT = {}, {}, [], []
for g, rs in groups.items():
    vo = [variants(r, 0) for r in rs]; vt = [variants(r, 1) for r in rs]
    for k in vo[0]: O[f"{g}|{k}"] = rank(np.mean([v[k] for v in vo], 0)); T[f"{g}|{k}"] = rank(np.mean([v[k] for v in vt], 0))
    if g != "fac3":
        co = [comps(r, 0) for r in rs]; ct = [comps(r, 1) for r in rs]
        CO += [np.mean([c[k] for c in co], 0) for k in co[0]]; CT += [np.mean([c[k] for c in ct], 0) for k in ct[0]]
# Agas's notebook components with real OOF (pushed 17 Oct 23:44, same row order as data/*.csv): candidates, and part of the projection basis
for v in ("v61", "v62", "v63", "v64"):
    z = np.load(f"agas/{v}/output/preds_{v}.npz", allow_pickle=False)
    for k in [k[4:] for k in z.keys() if k.startswith("oof_")]:
        n_ = f"a{v}|{k}"; O[n_] = rank(z["oof_" + k]); T[n_] = rank(z["test_" + k]); CO.append(O[n_]); CT.append(T[n_])
# team stack: logistic regression on the OOF ranks of both pipelines, fit inside folds for the OOF and on all rows for the test (OOF AUC 0.84806 vs 0.84774 ours alone)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ns = [n for n in O if n.startswith(("v4|", "fac|", "av6"))]; X, XS = np.column_stack([O[n] for n in ns]), np.column_stack([T[n] for n in ns]); p = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y): p[b] = LogisticRegression(max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
O["team_stack"] = rank(p); T["team_stack"] = rank(LogisticRegression(max_iter=2000).fit(X, y).decision_function(XS))
XO = np.column_stack(CO + [np.ones(len(y))]); XT = np.column_stack(CT + [np.ones(len(ids))])
def ext(n_, hi, lo, a_): O[n_] = rank(O[hi] + a_ * (O[hi] - O[lo])); T[n_] = rank(T[hi] + a_ * (T[hi] - T[lo]))
def mix(n_, a, b, w): O[n_] = rank(w * O[a] + (1 - w) * O[b]); T[n_] = rank(w * T[a] + (1 - w) * T[b])
for k in ("trees_only", "full_only", "xgb_full"): ext(f"ext1.5|v4-fac|{k}", f"v4|{k}", f"fac|{k}", 1.5)
ext("ext1.0|v4-fac|xgb_full", "v4|xgb_full", "fac|xgb_full", 1.0); ext("ext1.5|v4full-faclean", "v4|full_only", "fac|lean_only", 1.5)
mix("p17_00", "ext1.0|v4-fac|xgb_full", "ext1.5|v4full-faclean", 0.5)
rng = np.random.default_rng(int(os.environ.get("SEED", "5")))
def pseudo(n_, path):
    t = rank(pd.read_csv(path).set_index("application_id").default_probability.reindex(ids).values)
    b = np.linalg.lstsq(XT, t, rcond=None)[0]; e = t - XT @ b
    O[n_] = rank(XO @ b + rng.normal(0, e.std(), len(y))); T[n_] = t
    print(f"pseudo {n_:12s} projection corr {np.corrcoef(t, XT @ b)[0, 1]:.5f}")
dl = "agas/output/dl/"
for n_, p in [("v61", dl + "submission_v61_alt_ext0.5_tm30_lean.csv"), ("v58ext", dl + "submission_v58ext_ext1.5_v4_fac_full_only.csv"),
              ("v57", dl + "submission_v57_best.csv"), ("v63f", dl + "submission_v63f_consensus5_best.csv"), ("qb_00", "variants/qb_00_new_lgb_fullset.csv"),
              ("p07_00", "variants/p07_00_ext1.0_full_lgb_extra31_facfull_lgb_extra31.csv"), ("p10_00", "variants/p10_00_ext2.0_plus_ec_lgb_extra_lgb_extra.csv"),
              ("p12_00", "variants/p12_00_ext2.0_plus_prof_lgb_mc800_lgb_mc800.csv"), ("p19_00", "variants/p19_00_ext1.0_full_lgb_extra_lgb_extra.csv"),
              ("p21_00", "variants/p21_00_ext1.5_full_lgb_shallow_facfull_lgb_shallow.csv")]:
    pseudo(n_, p)
# public scores minus bag v4 top_mix (0.55860)
obs = {"fac|top_mix": 0.55865, "fac|linear30": 0.55826, "ext1.5|v4-fac|trees_only": 0.55810, "qb_00": 0.55931, "ext1.5|v4-fac|full_only": 0.56171,
       "p07_00": 0.55994, "p10_00": 0.55763, "p12_00": 0.55789, "p17_00": 0.56083, "p19_00": 0.56014, "p21_00": 0.55779,
       "v57": 0.55765, "v63f": 0.55712, "v58ext": 0.55963, "v61": 0.56312, "mix0.7|v61+p19_00": 0.56299, "mix0.7|v61+p17_00": 0.56253, "mix0.7|v61+av64|linear": 0.56086}
obs = {k: v - 0.55860 for k, v in obs.items()}
# the current queue head, for comparison
queue = [l.strip() for l in open("scheduled_unified.sh") if l.strip().startswith("variants/p22_")][:int(os.environ.get("NQ", "12"))]
for p in queue: pseudo(p.split("/")[1][:6], p)
# candidates around v61: steps past it away from a weaker file, and mixes with the other good files
for lo in ("v4|top_mix", "v4|linear30", "v4|lean_only", "fac|top_mix", "ext1.5|v4-fac|full_only", "v58ext", "v57", "p21_00", "v63f"):
    for a_ in (0.25, 0.5, 1.0): ext(f"ext{a_}|v61-{lo}", "v61", lo, a_)
for b in ("ext1.5|v4-fac|full_only", "p17_00", "p19_00", "qb_00", "v58ext", "v4|full_only", "v4|top_mix", "v4|linear30", "v63f"):
    for w in (0.3, 0.5, 0.7, 0.85): mix(f"mix{w}|v61+{b}", "v61", b, w)
mix("team_avg", "v4|top_mix", "av62|lgb_top", 0.5)
for b in ("team_stack", "team_avg"):
    for w in (0.3, 0.5, 0.7, 0.85): mix(f"mix{w}|v61+{b}", "v61", b, w)
for b in [n for n in O if n.startswith(("av64|", "av63|sel", "av62|lgb_top"))]:
    for w in (0.5, 0.7, 0.85): mix(f"mix{w}|v61+{b}", "v61", b, w)

def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)
N = int(os.environ.get("N", "4000")); names = list(O)
P = np.column_stack([O[n] for n in names])
D = np.array([ks_cols(P[idx], y[idx]) for idx in (np.random.default_rng([1, b]).choice(len(y), 9000, replace=False) for b in range(N))])
ir = names.index("v4|top_mix")
shrink = np.array([np.std(T[n] - T["v4|top_mix"]) / max(np.std(O[n] - O["v4|top_mix"]), 1e-12) for n in names])
diff = (D - D[:, [ir]]) * shrink
sig = 0.0006
def condition(diff, obs):
    oi = [names.index(n) for n in obs]; A = diff - diff.mean(0); Ao = A[:, oi]
    K = np.linalg.solve(Ao.T @ Ao + (len(A) - 1) * sig ** 2 * np.eye(len(oi)), Ao.T @ A).T
    e = np.random.default_rng(2).normal(0, sig, (len(A), len(oi)))
    return diff + (np.array(list(obs.values())) - diff[:, oi] - e) @ K.T
# leave one out: how well do the other scores predict each observed one
for k in obs:
    rest = {a: b for a, b in obs.items() if a != k}; j = names.index(k); c = condition(diff, rest)[:, j]
    print(f"loo {k:28s} observed {0.5586 + obs[k]:.5f} predicted {0.5586 + c.mean():.5f} sd {c.std():.5f}")
post = condition(diff, obs); goal = 0.56501 - 0.55860
r = pd.DataFrame({"prior": 0.5586 + diff.mean(0), "mean": 0.5586 + post.mean(0), "sd": post.std(0), "p565": (post >= goal).mean(0),
                  "rho_v61": [np.corrcoef(T[n], T["v61"])[0, 1] for n in names]}, index=names)
r = r.drop(list(obs) + ["v4|top_mix"], errors="ignore").sort_values("p565", ascending=False)
pd.set_option("display.width", 200); print(r.head(40).round(5).to_string())
os.makedirs("work", exist_ok=True); r.to_csv(f"work/v61_plan_{os.environ.get('SEED', '5')}.csv")
pd.to_pickle({n: T[n] for n in r.index[:40]}, "work/v61_top_T.pkl")
# greedy picks for the next slots: each one adds the most chance that the best of all picks so far clears 0.565
top = list(r.index[:60]); M = post[:, [names.index(n) for n in top]]; cur = post[:, [names.index(n) for n in obs]].max(1); picks = []
for _ in range(4):
    j = max(range(len(top)), key=lambda j: (np.maximum(cur, M[:, j]) >= goal).mean() + 1e-3 * np.maximum(cur, M[:, j]).mean())
    cur = np.maximum(cur, M[:, j]); picks.append(top[j]); print(f"pick {top[j]:40s} P(best above 0.565) {(cur >= goal).mean():.4f}")
pd.to_pickle({n: T[n] for n in top}, "work/v61_top_T.pkl")
