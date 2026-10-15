import sys, glob, os
import numpy as np, pandas as pd
# Is a candidate's difference from bag v4 top_mix as large on the test rows as on the OOF rows the simulation uses?
S = sys.argv[1]
src = open(S + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
q = [resolve(os.path.basename(f)[8:-4]) for f in sorted(glob.glob("variants/p15_*.csv"))[:12]] + list(obs)
rows = []
for n in dict.fromkeys(q):
    if n and n != "v4|top_mix":
        so, st = np.std(O[n] - O["v4|top_mix"]), np.std(T[n] - T["v4|top_mix"])
        rows.append((n, so, st, st / so))
t = pd.DataFrame(rows, columns=["candidate", "sd_oof", "sd_test", "ratio"]).set_index("candidate")
print(t.round(4).to_string()); print("median ratio", t.ratio.median().round(3))
