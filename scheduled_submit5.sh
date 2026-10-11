#!/bin/zsh
# Submits top_mix from the factor run, the factor full data refit, and the plain 10 fold run as a control once the Kaggle quota resets (07:00 WIB).
cd /Users/yutaaa/Developments/dsc-logikaui
target=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-12 07:01:00" +%s)
sleep $(( target - $(date +%s) ))
date
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }

for f in run_fac/variants/top_mix.csv run_full_fac/variants/top_mix.csv run10/variants/top_mix.csv; do
  valid $f && .venv/bin/kaggle competitions submit -f $f -m "${f:h:h} ${f:t:r}" && sleep 30
done
sleep 60
.venv/bin/kaggle competitions submissions | head -6
.venv/bin/kaggle competitions leaderboard -s | head -14
