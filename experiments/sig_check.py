import sys
import numpy as np
# Marginal likelihood of the observed public differences under the simulated splits, for several mismatch levels sig.
src = open(sys.argv[1] + "harvest_kf.py").read()
exec(src[:src.index("sig = 0.0006")])
names = list(O); diff = ks(names) - ks(["v4|top_mix"])
oi = [names.index(n) for n in obs]; v = np.array(list(obs.values())); X = diff[:, oi]
mu, C = X.mean(0), np.cov(X.T)
print("prior mean of observed:", dict(zip(obs, mu.round(5))), "sd:", np.sqrt(np.diag(C)).round(5))
for s in (0, 0.0003, 0.0006, 0.001, 0.0015, 0.002, 0.003):
    M = C + s ** 2 * np.eye(len(v)); r = v - mu
    print(f"sig {s:.4f} loglik {-0.5 * (r @ np.linalg.solve(M, r) + np.linalg.slogdet(M)[1]):.3f}")
