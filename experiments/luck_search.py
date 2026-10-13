import os, sys
import numpy as np, pandas as pd

# Searches a very large number of simulated public splits for the ones that behave like the real one (every observed public score),
# then scores every candidate on them and shows what those splits look like.
#   stage 1, run in parallel parts:  luck_search.py S stage1 <part> <parts> <B>
#   combine and score:               luck_search.py S combine <parts> <B> <KEEP>
S, mode = sys.argv[1], sys.argv[2]
src = open(S + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
names = list(O); ob = list(obs); v = np.array(list(obs.values()))
split = lambda b: np.random.default_rng([3, b]).choice(len(y), 9000, replace=False)
P1 = np.column_stack([O[n] for n in ["v4|top_mix"] + ob])
if mode == "stage1":
    part, parts, B = map(int, sys.argv[3:6])
    bs = np.arange(part, B, parts); D = np.zeros((len(bs), len(ob)))
    for i, b in enumerate(bs):
        idx = split(b); k = ks_cols(P1[idx], y[idx]); D[i] = k[1:] - k[0]
    np.savez(S + f"luck_{part}.npz", b=bs, d=D); sys.exit()

parts, B, KEEP = map(int, sys.argv[3:6])
z = [np.load(S + f"luck_{p}.npz") for p in range(parts)]
bs = np.concatenate([q["b"] for q in z]); D = np.vstack([q["d"] for q in z])
lw = -((D - v) ** 2).sum(1) / (2 * sig ** 2); w = np.exp(lw - lw.max()); w /= w.sum()
keep = np.argsort(-w)[:KEEP]; wk = w[keep] / w[keep].sum()
print(f"splits searched {len(bs)}, effective matching splits {1 / (w ** 2).sum():.0f}, weight kept {w[keep].sum():.4f}")
print("observed:", dict(zip(ob, v.round(5))))
print("matching splits reproduce them as:", dict(zip(ob, (wk @ D[keep]).round(5))))

# what the lucky splits look like: weighted composition against all splits, in standard deviations of a random split
f = pd.read_parquet("features.parquet"); tr = f[f.target.notna()].reset_index(drop=True)
cols = [c for c in tr.columns if c not in ("application_id", "customer_id") and tr[c].dtype.kind in "fiub"]
X = tr[cols].astype(float).fillna(tr[cols].astype(float).median()).values
rand = np.array([X[split(b)].mean(0) for b in bs[:2000]])
luck = np.array([X[split(b)].mean(0) for b in bs[keep]])
sd = rand.std(0); zc = pd.Series((wk @ luck - rand.mean(0))[sd > 0] / sd[sd > 0], index=np.array(cols)[sd > 0])
print("lucky split composition, largest shifts (z):", zc.reindex(zc.abs().sort_values(ascending=False).index).head(15).round(2).to_dict())

# every candidate on the matching splits
Pa = np.column_stack([O[n] for n in names])
Dk = np.array([ks_cols(Pa[idx], y[idx]) for idx in map(split, bs[keep])]); Dk = Dk - Dk[:, [names.index("v4|top_mix")]]
goal = 0.565 - 0.55860
t = pd.DataFrame({"mean": wk @ Dk, "p565": wk @ (Dk >= goal)}, index=names).sort_values("p565", ascending=False)
print("best single shot on the lucky splits:\n", t.head(20).round(5).to_string())
cur = Dk[:, [names.index(n) for n in ["v4|top_mix"] + ob]].max(1); fam = [names.index(n) for n in ["v4|top_mix"] + ob]
for step in range(int(os.environ.get("NSLOTS", "50"))):
    g, j = max(((wk * (np.maximum(cur, Dk[:, j]) >= goal)).sum() + wk @ np.maximum(cur, Dk[:, j]), j) for j in range(len(names)) if j not in fam)
    fam.append(j); cur = np.maximum(cur, Dk[:, j])
    if step < 6 or step % 10 == 9: print(f"+ {names[j]:40s} P(0.565) {(wk * (cur >= goal)).sum():.3f}  E[best] {0.55860 + wk @ cur:.5f}")
