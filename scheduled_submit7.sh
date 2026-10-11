#!/bin/zsh
# Submits one top_mix file at 07:01: the 2 seed run if it finished with a better CV, otherwise the 1 seed factor run.
cd /Users/yutaaa/Developments/dsc-logikaui
target=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-12 07:01:00" +%s)
sleep $(( target - $(date +%s) ))
date
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }
f=$(.venv/bin/python - <<'PY'
import json, os
pick = "run_fac/variants/top_mix.csv"
try:
    nb = json.load(open("run_seeds/notebook_seeds.ipynb"))
    lines = [l for c in nb["cells"] for o in c.get("outputs", []) for l in "".join(o.get("text", "")).splitlines()]
    row = [l.split() for l in lines if l.startswith("top_mix ")]
    # notebook rewrites itself only when nbconvert finishes, so outputs present means the run is complete
    if row and os.path.exists("run_seeds/variants/top_mix.csv"):
        auc, ks = float(row[0][2]), float(row[0][4])
        print(f"2 seed top_mix auc {auc} ks {ks}", file=__import__("sys").stderr)
        if auc > 0.84649 and ks >= 0.5237:
            pick = "run_seeds/variants/top_mix.csv"
except Exception as e:
    print("2 seed check failed:", e, file=__import__("sys").stderr)
print(pick)
PY
)
echo "picked $f"
valid $f && .venv/bin/kaggle competitions submit -f $f -m "${f:h:h} ${f:t:r}"
sleep 90
.venv/bin/kaggle competitions submissions | head -5
.venv/bin/kaggle competitions leaderboard -s | head -16
