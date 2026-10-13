import os, re, sys
import numpy as np, pandas as pd
# Single shot chance of reaching public 0.565 for every candidate, after conditioning on the observed scores, grouped by family.
os.environ["N"] = "4000"
src = open(sys.argv[1] + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
names = list(O); post = condition(ks(names) - ks(["v4|top_mix"]))
goal = 0.565 - 0.55860
t = pd.DataFrame({"mean": post.mean(0), "sd": post.std(0), "p565": (post >= goal).mean(0)}, index=names)
fam = lambda n: re.sub(r"\d\.\d", "a", n.split("|")[0]) + "|" + ("full" if "full" in n else "lean") + ("|" + n.split("|")[-1] if n.startswith(("v4|", "fac", "ext")) and "-" in n else "")
t["family"] = [n.split("|")[0] if n.startswith(("new", "mix")) else re.sub(r"ext\d\.\d", "ext", n.rsplit("|", 1)[0]) for n in names]
print(t.sort_values("p565", ascending=False).head(25).round(5).to_string())
print(t.groupby("family")[["mean", "sd", "p565"]].agg(["mean", "max"]).sort_values(("p565", "max"), ascending=False).head(20).round(4).to_string())
print("corr of p565 with sd:", round(np.corrcoef(t["sd"], t["p565"])[0, 1], 2), " with mean:", round(np.corrcoef(t["mean"], t["p565"])[0, 1], 2))
