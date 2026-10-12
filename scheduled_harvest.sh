#!/bin/zsh
# Submits the diverse variant queue one file at a time (three a day, hours apart), checking the leaderboard before and after each.
cd /Users/yutaaa/Developments/dsc-logikaui
K=.venv/bin/kaggle
queue=(
  "2026-10-13 07:01|variants/harvest_01_v4_lgb_only.csv"
  "2026-10-13 12:01|variants/harvest_02_v4_top_only.csv"
  "2026-10-13 18:01|variants/harvest_03_fac_top_only.csv"
  "2026-10-14 07:01|variants/harvest_04_v4_cat_only.csv"
  "2026-10-14 12:01|variants/harvest_05_v4_xgb_only.csv"
  "2026-10-14 18:01|variants/harvest_06_v4_main.csv"
  "2026-10-15 07:01|variants/harvest_08_v4_trees_only.csv"
  "2026-10-15 12:01|variants/harvest_09_v4_linear30.csv"
  "2026-10-15 18:01|variants/harvest_10_fac_full_only.csv"
  "2026-10-16 07:01|variants/harvest_11_v4_lean_only.csv"
  "2026-10-16 12:01|variants/harvest_12_fac_linear_only.csv"
)
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }
our_rank() { $K competitions leaderboard -s 2>/dev/null | grep -n "kali ini beneran deh" | head -1 | awk -F: '{print $1-4}' }
for item in $queue; do
  when=${item%%|*}; f=${item#*|}
  target=$(date -j -f "%Y-%m-%d %H:%M" "$when" +%s)
  # compares wall clock time, so a slot missed while the laptop slept runs as soon as it wakes
  while (( $(date +%s) < target )); do sleep 30; done
  echo "$(date '+%F %T') rank before: $(our_rank)"
  if ! valid $f; then echo "invalid file $f, skipped"; continue; fi
  if ! $K competitions submit -f $f -m "harvest ${f:t:r}"; then echo "submit failed for $f"; continue; fi
  line=""
  for i in {1..60}; do
    line=$($K competitions submissions 2>/dev/null | grep "${f:t}" | head -1)
    [[ $line == *COMPLETE* || $line == *ERROR* ]] && break
    sleep 30
  done
  score=$(echo $line | awk '{print $NF}'); r=$(our_rank)
  echo "$(date '+%F %T') ${f:t} public $score rank after: $r"
  grep -q "^## Log submit otomatis" PROGRESS.md || printf "\n## Log submit otomatis\n\n" >> PROGRESS.md
  echo "- $(date '+%F %H:%M') ${f:t:r}: public $score, rank $r" >> PROGRESS.md
  git add PROGRESS.md && git commit -q -m "Auto submit ${f:t:r}: public $score, rank $r" && git push -q
done
