#!/bin/zsh
# Unified queue (extrapolations, new model types, variants): submits one file at a time (three a day, hours apart), checking the leaderboard before and after each.
cd /Users/yutaaa/Developments/dsc-logikaui
K=.venv/bin/kaggle
queue=(
  "2026-10-13 12:01|variants/qb_00_new_lgb_fullset.csv"
  "2026-10-13 18:01|variants/qb_01_ext1.0_v4_fac_top_mix.csv"
  "2026-10-14 07:01|variants/qb_02_ext0.5_v4_fac_trees_only.csv"
  "2026-10-14 12:01|variants/qb_03_new_lgb_shallow.csv"
  "2026-10-14 18:01|variants/qb_04_ext0.5_v4_fac_top_only.csv"
  "2026-10-15 07:01|variants/qb_05_ext1.5_v4_fac_full_only.csv"
  "2026-10-15 12:01|variants/qb_06_v4_lgb_only.csv"
  "2026-10-15 18:01|variants/qb_07_mix_lgb_extra.csv"
  "2026-10-16 07:01|variants/qb_08_new_lgb_extra31.csv"
  "2026-10-16 12:01|variants/qb_09_fac_top_only.csv"
  "2026-10-16 18:01|variants/qb_10_new_xgb_shallow.csv"
  "2026-10-17 07:01|variants/qb_11_ext1.0_v4_fac_main.csv"
  "2026-10-17 12:01|variants/qb_12_ext1.0_xgb_deep_v4_xgb_only.csv"
  "2026-10-17 18:01|variants/qb_13_mix_lgb_goss.csv"
  "2026-10-18 07:01|variants/qb_14_fac_lean_only.csv"
  "2026-10-18 12:01|variants/qb_15_mix_lgb_ff015.csv"
  "2026-10-18 18:01|variants/qb_16_ext0.5_v4_fac_cat_only.csv"
  "2026-10-19 07:01|variants/qb_17_new_lgb_ff015.csv"
  "2026-10-19 12:01|variants/qb_18_ext1.0_v4_fac_lgb_only.csv"
  "2026-10-19 18:01|variants/qb_19_v4_linear30.csv"
  "2026-10-20 07:01|variants/qb_20_ext2.0_v4_fac_full_only.csv"
  "2026-10-20 12:01|variants/qb_21_new_cat_lossguide.csv"
  "2026-10-20 18:01|variants/qb_22_ext2.0_v4_fac_cat_only.csv"
  "2026-10-21 07:01|variants/qb_23_ext1.5_v4_fac_linear30.csv"
  "2026-10-21 12:01|variants/qb_24_new_lgb_ff06.csv"
  "2026-10-21 18:01|variants/qb_25_fac_xgb_only.csv"
  "2026-10-22 07:01|variants/qb_26_mix_cat_deep.csv"
  "2026-10-22 12:01|variants/qb_27_ext2.0_v4_fac_lgb_only.csv"
  "2026-10-22 18:01|variants/qb_28_fac_full_only.csv"
  "2026-10-23 07:01|variants/qb_29_ext1.0_v4_fac_full_only.csv"
  "2026-10-23 12:01|variants/qb_30_mix_lgb_shallow.csv"
  "2026-10-23 18:01|variants/qb_31_mix_lgb_lintree.csv"
  "2026-10-24 07:01|variants/qb_32_ext1.5_v4_fac_lgb_only.csv"
  "2026-10-24 12:01|variants/qb_33_v4_top_only.csv"
  "2026-10-24 18:01|variants/qb_34_ext1.0_v4_fac_linear30.csv"
  "2026-10-25 07:01|variants/qb_35_mix_lgb_ff06.csv"
  "2026-10-25 12:01|variants/qb_36_ext0.5_v4_fac_lgb_only.csv"
  "2026-10-25 18:01|variants/qb_37_ext1.0_v4_fac_cat_only.csv"
  "2026-10-26 07:01|variants/qb_38_ext2.0_v4_fac_main.csv"
  "2026-10-26 12:01|variants/qb_39_ext1.0_v4_fac_trees_only.csv"
  "2026-10-26 18:01|variants/qb_40_mix_cat_lossguide.csv"
  "2026-10-27 07:01|variants/qb_41_fac_linear_only.csv"
  "2026-10-27 12:01|variants/qb_42_ext1.0_lgb_extra_v4_lgb_only.csv"
  "2026-10-27 18:01|variants/qb_43_v4_main.csv"
  "2026-10-28 07:01|variants/qb_44_ext1.5_v4_fac_lean_only.csv"
  "2026-10-28 12:01|variants/qb_45_v4_trees_only.csv"
  "2026-10-28 18:01|variants/qb_46_ext0.5_v4_fac_linear30.csv"
)
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }
our_rank() { d=$(mktemp -d); $K competitions leaderboard -d -p $d >/dev/null 2>&1; unzip -o -q $d/*.zip -d $d 2>/dev/null
  .venv/bin/python -c "import pandas as pd,glob,sys; t=pd.read_csv(glob.glob(sys.argv[1]+'/*.csv')[0]); r=t.Rank[t.TeamName=='kali ini beneran deh']; print(int(r.iloc[0]) if len(r) else '')" $d; rm -rf $d }
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
