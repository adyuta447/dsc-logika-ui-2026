import os, sys
import numpy as np
from scipy.stats import multivariate_normal
# Do the observed public scores fit the simulation better when each candidate's simulated difference from bag v4 top_mix
# is shrunk by r = sd(test difference) / sd(OOF difference)?  (Test predictions average 30 models, OOF only one per seed.)
S = sys.argv[1]
src = open(S + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
ob = list(obs); v = np.array(list(obs.values()))
d = ks(ob) - ks(["v4|top_mix"])
r = np.array([np.std(T[n] - T["v4|top_mix"]) / np.std(O[n] - O["v4|top_mix"]) for n in ob])
print("r:", dict(zip(ob, r.round(3))))
for name, dd in (("as now", d), ("shrunk by r", d * r)):
    mu, C = dd.mean(0), np.cov(dd.T)
    print(f"{name:12s} joint log likelihood of the observed scores {multivariate_normal(mu, C + sig ** 2 * np.eye(len(ob))).logpdf(v):.2f}",
          " z:", ((v - mu) / np.sqrt(np.diag(C) + sig ** 2)).round(2))
# the whole candidate set: how much would the shrink change the spread
names = list(O); rr = np.array([np.std(T[n] - T["v4|top_mix"]) / max(np.std(O[n] - O["v4|top_mix"]), 1e-12) for n in names if n != "v4|top_mix"])
print("r over all candidates: quantiles", np.quantile(rr, [0, .1, .5, .9, 1]).round(3))
