import glob, hashlib, os, re, subprocess, sys, tempfile
import numpy as np, pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata

# Picks the next public submissions: every candidate is scored on simulated public splits, the splits are moved to agree with
# every public score seen so far, and files are chosen greedily for the best chance of a public score above 0.565 (ties: best expected maximum).
S = sys.argv[1]
f = pd.read_parquet("features.parquet", columns=["target"]); y = f[f.target.notna()].target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
def variants(r, i):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    pick = lambda pre: np.mean([rank(res[k][i]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    lin = rank(res["linear"][i]); tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    return {"top_mix": 0.8 * (core + tr["top"]) / 2 + 0.2 * lin, "trees_only": (core + tr["top"]) / 2, "linear30": 0.7 * (core + tr["top"]) / 2 + 0.3 * lin,
            "main": 0.8 * core + 0.2 * lin, "top_only": 0.8 * tr["top"] + 0.2 * lin, "lgb_only": (c["lgb_full"] + c["lgb_lean"] + c["lgb_top"]) / 3,
            "cat_only": (c["cat_full"] + c["cat_lean"] + c["cat_top"]) / 3, "xgb_only": (c["xgb_full"] + c["xgb_lean"] + c["xgb_top"]) / 3,
            "full_only": tr["full"], "lean_only": tr["lean"], "linear_only": lin, "lgb_full": c["lgb_full"], "cat_full": c["cat_full"], "xgb_full": c["xgb_full"]}
# fac3 uses only the fold seeds v4 has, so v4 minus fac3 carries the feature step without fold noise
groups = {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"], "fac3": ["run_fac", "run_s7", "run_s13"]}
if os.path.exists("run_v4_s21/models/predictions.pkl"): groups["v4_4"] = ["run_v4_s42", "run_v4_s7", "run_v4_s13", "run_v4_s21"]
O, T = {}, {}
for g, rs in groups.items():
    vo = [variants(r, 0) for r in rs]; vt = [variants(r, 1) for r in rs]
    for k in vo[0]:
        O[f"{g}|{k}"] = rank(np.mean([v[k] for v in vo], 0)); T[f"{g}|{k}"] = rank(np.mean([v[k] for v in vt], 0))
for name in sorted({p.split("/")[-1].rsplit("_s", 1)[0] for p in glob.glob(S + "div/*.npz")}):
    z = [np.load(p) for p in sorted(glob.glob(S + f"div/{name}_s*.npz"))]
    if len(z) < 3: continue
    O[f"new|{name}"] = rank(np.mean([rank(q["oof"]) for q in z], 0)); T[f"new|{name}"] = rank(np.mean([rank(q["pred"]) for q in z], 0))
    O[f"mix|{name}"] = rank(0.5 * O["v4|top_mix"] + 0.5 * O[f"new|{name}"]); T[f"mix|{name}"] = rank(0.5 * T["v4|top_mix"] + 0.5 * T[f"new|{name}"])
def ext(n_, hi, lo, a_):
    O[n_] = rank(O[hi] + a_ * (O[hi] - O[lo])); T[n_] = rank(T[hi] + a_ * (T[hi] - T[lo]))
# extrapolations: step past the better model along the direction it improved on
for k in ("top_mix", "trees_only", "linear30", "main", "top_only", "lgb_only", "cat_only", "xgb_only", "full_only", "lean_only", "lgb_full", "cat_full", "xgb_full"):
    for hi, lo in [("v4", "fac"), ("v4", "fac3")] + [("v4_4", "fac")] * ("v4_4" in groups):
        for a_ in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0): ext(f"ext{a_}|{hi}-{lo}|{k}", f"{hi}|{k}", f"{lo}|{k}", a_)
for new_, old_ in (("lgb_extra", "lgb_only"), ("xgb_deep", "xgb_only"), ("lgb_shallow", "lgb_only"), ("cat_deep", "cat_only")):
    if f"new|{new_}" in O:
        for a_ in (0.5, 1.0): ext(f"ext{a_}|{new_}-v4|{old_}", f"new|{new_}", f"v4|{old_}", a_)
# per model type: full set minus lean set, and full set v4 minus the same without the season and rhythm features
for k in ("lgb_extra", "xgb_deep", "lgb_shallow", "lgb_ff06", "lgb_extra31", "lgb_mc800", "xgb_shallow", "cat_deep", "lgb_deep", "lgb_goss", "lgb_ff015", "cat_lossguide",
          "hgb", "xgb_lossguide", "cat_d6", "lgb_extra_s1"):
    for hi, lo, steps in ((f"full_{k}", k, (0.5, 1.0, 1.5)), (f"full_{k}", f"facfull_{k}", (1.0, 1.5, 2.0))):
        if f"new|{hi}" in O and f"new|{lo}" in O:
            for a_ in steps: ext(f"ext{a_}|{hi}-{lo}", f"new|{hi}", f"new|{lo}", a_)
# full set minus lean set on the notebook blends, the direction the public split has rewarded so far
for g in [g for g in ("v4", "v4_4") if g in groups]:
    for a_ in (0.5, 1.0, 1.5, 2.0, 3.0): ext(f"ext{a_}|{g}|full-lean", f"{g}|full_only", f"{g}|lean_only", a_)
# bolder steps for the 0.565 goal: further along v4 minus fac, and both rewarded directions at once (v4 full set minus fac lean set)
for k in ("full_only", "lgb_full", "cat_full", "xgb_full"):
    for lo in ("fac", "fac3"):
        for a_ in (4.0, 5.0): ext(f"ext{a_}|v4-{lo}|{k}", f"v4|{k}", f"{lo}|{k}", a_)
for a_ in (0.5, 1.0, 1.5, 2.0): ext(f"ext{a_}|v4full-faclean", "v4|full_only", "fac|lean_only", a_)
# lean set plus one group of the dropped fields (sixth, eighth and ninth pass)
for k in ("lgb_extra", "xgb_deep", "lgb_extra31", "lgb_shallow", "xgb_shallow", "lgb_mc800", "lgb_deep", "lgb_goss", "lgb_ff015", "lgb_ff06"):
    for g in ("prof", "time", "ec", "cs"):
        if f"new|plus_{g}_{k}" in O:
            for a_ in (1.0, 2.0): ext(f"ext{a_}|plus_{g}_{k}-{k}", f"new|plus_{g}_{k}", f"new|{k}", a_)
# full set minus one group of the dropped fields (tenth and eleventh passes)
for k in ("lgb_extra", "lgb_shallow", "lgb_extra31", "xgb_deep", "xgb_shallow", "lgb_mc800", "lgb_deep", "lgb_goss", "lgb_ff015", "lgb_ff06"):
    for g in ("prof", "time", "ec", "cs"):
        if f"new|minus_{g}_{k}" in O and f"new|full_{k}" in O:
            for a_ in (1.0, 2.0): ext(f"ext{a_}|full_{k}-minus_{g}_{k}", f"new|full_{k}", f"new|minus_{g}_{k}", a_)

# public scores relative to bag v4 top_mix (0.55860): two early manual submits, then every scored file in the submit log
san = lambda n: re.sub(r"[^A-Za-z0-9.]+", "_", n)
by = {san(n): n for n in O}; assert len(by) == len(O)
def resolve(s):
    # file names lose the separators, so a candidate, or a 50/50 mix of two, is found again by its cleaned name
    if s in by: return by[s]
    hit = [(a, by[s[len(san(a)) + 6:]]) for a in by.values() if s.startswith(f"mix2_{san(a)}_") and s[len(san(a)) + 6:] in by]
    if hit:
        (a, b), n_ = hit[0], f"mix2|{hit[0][0]}+{hit[0][1]}"
        O[n_] = rank((O[a] + O[b]) / 2); T[n_] = rank((T[a] + T[b]) / 2); return n_
obs = {"fac|top_mix": 0.00005, "fac|linear30": -0.00034}
for stem, score in re.findall(r"^- \S+ \S+ (\S+): public (0\.\d+)", open("PROGRESS.md").read(), re.M):
    n_ = resolve(stem.split("_", 2)[2])
    if n_: obs[n_] = float(score) - 0.55860
    else: print("not a candidate:", stem)
print("observed:", {k: round(v, 5) for k, v in obs.items()})

def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)
N = int(os.environ.get("N", "10000"))
split = lambda b: np.random.default_rng([1, b]).choice(len(y), 9000, replace=False)
cf = S + "ks_cache.pkl"; cache = pd.read_pickle(cf) if os.path.exists(cf) else {}
def ks(ns):
    # KS of each candidate on the same simulated splits, cached by the prediction itself so a changed candidate is scored again
    keys = [hashlib.md5(O[n].tobytes()).hexdigest() + f"_{N}" for n in ns]
    todo = [j for j, k in enumerate(keys) if k not in cache]
    if todo:
        P = np.column_stack([O[ns[j]] for j in todo])
        D = np.array([ks_cols(P[idx], y[idx]) for idx in map(split, range(N))])
        cache.update({keys[j]: D[:, i] for i, j in enumerate(todo)}); pd.to_pickle(cache, cf)
    return np.column_stack([cache[k] for k in keys])
sig = 0.0006
def condition(diff):
    # ensemble Kalman update: each split moves toward the observed public differences as far as they predict each candidate,
    # which keeps every split in play however many scores are observed
    oi = [names.index(n) for n in obs]; A = diff - diff.mean(0); Ao = A[:, oi]
    K = np.linalg.solve(Ao.T @ Ao + (len(A) - 1) * sig ** 2 * np.eye(len(oi)), Ao.T @ A).T
    e = np.random.default_rng(2).normal(0, sig, (len(A), len(oi)))
    return diff + (np.array(list(obs.values())) - diff[:, oi] - e) @ K.T
names = list(O); ir = names.index("v4|top_mix"); print("candidates:", len(names))
# test predictions average 30 models but OOF ones a single model per fold seed, so on the test rows a candidate differs less from
# bag v4 top_mix than on the OOF rows; its simulated difference is shrunk by that ratio (observed scores: log likelihood 37.5 -> 38.4)
shrink = lambda ns: np.array([np.std(T[n] - T["v4|top_mix"]) / max(np.std(O[n] - O["v4|top_mix"]), 1e-12) for n in ns])
base = ks(["v4|top_mix"]); diff = (ks(names) - base) * shrink(names)
m = pd.Series(condition(diff).mean(0), index=names).sort_values(ascending=False)
print("best conditioned means:", m.head(12).round(5).to_dict())
# 50/50 mixes of the ten candidates that look best on the conditioned splits
top = list(m.index[:10]); new = []
for i, a in enumerate(top):
    for b in top[i + 1:]:
        n_ = f"mix2|{a}+{b}"
        if n_ not in O: O[n_] = rank((O[a] + O[b]) / 2); T[n_] = rank((T[a] + T[b]) / 2); new.append(n_)
names += new; post = condition(np.hstack([diff, (ks(new) - base) * shrink(new)]))
print("mixes added:", len(new), "best mix means:", pd.Series(post[:, -len(new):].mean(0), index=new).sort_values(ascending=False).head(5).round(5).to_dict())
nslots = int(os.environ.get("NSLOTS", "37"))
fam = [names.index(n) for n in ["v4|top_mix"] + list(obs)]; cur = post[:, fam].max(1)
# the current rank lines from the full leaderboard, for the report (the 15 Oct lines if the download fails)
try:
    d = tempfile.mkdtemp(); subprocess.run([".venv/bin/kaggle", "competitions", "leaderboard", "-d", "-p", d], capture_output=True, timeout=120)
    sc = np.sort(pd.read_csv(glob.glob(d + "/*.zip")[0]).Score.values)[::-1]; lb = {"rank 13": sc[12], "rank 10": sc[9], "rank 8": sc[7]}
except Exception:
    lb = {"rank 13": 0.56447, "rank 10": 0.56742, "rank 8": 0.56832}
lb |= {"0.565": 0.565, "0.568": 0.568}
print(f"now: E[best] {0.55860 + cur.mean():.5f}")
# user target since 15 Oct: a public score above 0.565
goal = 0.56501 - 0.55860; picked = []; print("goal: above 0.565")
for step in range(nslots):
    # chance of reaching the goal, smoothed over 0.0003 so the pick does not chase single splits, plus the expected maximum as tie break
    g, j = max((ndtr((np.maximum(cur, post[:, j]) - goal) / 0.0003).mean() + np.maximum(cur, post[:, j]).mean(), j) for j in range(len(names)) if j not in fam)
    fam.append(j); cur = np.maximum(cur, post[:, j]); picked.append(names[j])
    if step < 12 or step % 10 == 0 or step == nslots - 1:
        print(f"+ {names[j]:30s} E[best] {0.55860 + cur.mean():.5f}", "  ".join(f"{k} {(cur >= v - 0.55860).mean():.2f}" for k, v in lb.items()))
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
for i, n in enumerate(picked):
    sub = pd.DataFrame({"application_id": ids, "default_probability": T[n]})
    assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
    sub.to_csv(f"variants/{os.environ.get('PREFIX', 'qe')}_{i:02d}_{san(n)}.csv", index=False)
print("files:", len(picked))
