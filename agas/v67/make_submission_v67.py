import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "agas/shared"))
os.chdir(ROOT)

OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(OUT / "preds_v67.npz", allow_pickle=True)
y = z["y"].astype(int)
ref_tr = list(z["train_ids"]); ref_te = list(z["test_ids"])

def r01(v):
    return rankdata(v) / len(v)

meta_oof = z["oof_meta_logit"]; meta_test = z["test_meta_logit"]
meta_lgb_oof = z["oof_meta_lgb"]; meta_lgb_test = z["test_meta_lgb"]
anchor_oof = z["oof_anchor"]; anchor_test = z["test_anchor"]

test_ids = pd.read_csv(ROOT / "data/test_applications.csv", usecols=["application_id"]).application_id.values
assert list(test_ids) == ref_te

candidates = {}

def add(name, oof, test):
    candidates[name] = (np.asarray(oof, float).ravel(), np.asarray(test, float).ravel())

add("meta_logit", meta_oof, meta_test)
add("meta_lgb", meta_lgb_oof, meta_lgb_test)
add("anchor", anchor_oof, anchor_test)

# blends with the proven public vector (v66_04)
for w in (0.10, 0.25, 0.50):
    o = r01((1-w)*anchor_oof + w*meta_oof)
    t = r01((1-w)*anchor_test + w*meta_test)
    add(f"blend_anchor_meta_w{w:.2f}", o, t)

# simple consensus of meta + anchor ranks
o = (r01(meta_oof) + r01(anchor_oof)) / 2
t = (r01(meta_test) + r01(anchor_test)) / 2
add("consensus_meta_anchor", o, t)

# write files
names = sorted(candidates)
for i, name in enumerate(names, 1):
    o, t = candidates[name]
    fn = OUT / f"submission_v67_{i:02d}_{name}.csv"
    pd.DataFrame({"application_id": test_ids, "default_probability": t}).to_csv(fn, index=False)

# summary
rows = []
for name in names:
    o, t = candidates[name]
    rows.append({"name": name, "auc": float(roc_auc_score(y, o)),
                 "ks": float(ks_2samp(o[y==1], o[y==0]).statistic)})
rows.sort(key=lambda r: -r["auc"])
json.dump({"candidates": rows}, open(OUT / "summary_v67_candidates.json", "w"), indent=2)
print(f"wrote {len(names)} candidates")
for r in rows:
    print(f"{r['name']:32s} AUC={r['auc']:.6f} KS={r['ks']:.6f}")
