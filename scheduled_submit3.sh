#!/bin/zsh
# Submits three 10 fold notebook variants once the Kaggle quota resets (07:00 WIB).
cd /Users/yutaaa/Developments/dsc-logikaui
target=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-12 07:01:00" +%s)
sleep $(( target - $(date +%s) ))
date
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }

for f in run10/variants/main.csv run10/variants/top_mix.csv run10/variants/trees_only.csv; do
  valid $f && .venv/bin/kaggle competitions submit -f $f -m "notebook 10fold variant ${f:t:r}" && sleep 30
done
sleep 60
.venv/bin/kaggle competitions submissions | head -6
.venv/bin/kaggle competitions leaderboard -s | head -14
