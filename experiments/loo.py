import os, sys
import numpy as np
from scipy.stats import norm
# Does conditioning on some public scores predict the others? Leave one out, and sequential in submit order.
os.environ["N"] = "4000"
src = open(sys.argv[1] + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
ob = list(obs); D = ks(["v4|top_mix"] + ob); d = D[:, 1:] - D[:, [0]]; v = np.array(list(obs.values()))
mu, C = d.mean(0), np.cov(d.T)
def pred(i, rest):
    if not rest: return mu[i], np.sqrt(C[i, i])
    Cr = C[np.ix_(rest, rest)] + sig ** 2 * np.eye(len(rest)); c = C[i, rest]
    return mu[i] + c @ np.linalg.solve(Cr, v[rest] - mu[rest]), np.sqrt(C[i, i] - c @ np.linalg.solve(Cr, c) + sig ** 2)
for title, sets in (("leave one out", [[j for j in range(len(ob)) if j != i] for i in range(len(ob))]), ("sequential", [list(range(i)) for i in range(len(ob))])):
    print(title); lp0 = lp1 = 0
    for i, rest in enumerate(sets):
        m0, s0 = pred(i, []); m1, s1 = pred(i, rest)
        lp0 += norm.logpdf(v[i], m0, np.sqrt(s0 ** 2 + sig ** 2)); lp1 += norm.logpdf(v[i], m1, s1)
        print(f"  {ob[i]:28s} actual {v[i]:+.5f}  prior {m0:+.5f} sd {s0:.5f}  conditioned {m1:+.5f} sd {s1:.5f}  error {abs(v[i] - m0):.5f} -> {abs(v[i] - m1):.5f}")
    print(f"  log predictive density: prior {lp0:.2f}, conditioned {lp1:.2f}")
print("correlations:\n", np.corrcoef(d.T).round(2))
