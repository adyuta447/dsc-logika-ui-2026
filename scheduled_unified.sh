#!/bin/zsh
# Unified queue (extrapolations, new model types, variants): submits one file at a time (three a day, hours apart), checking the leaderboard before and after each.
cd /Users/yutaaa/Developments/dsc-logikaui
K=.venv/bin/kaggle
queue=(
  "2026-10-14 10:01|variants/p03_00_ext1.5_v4full_faclean.csv"
  "2026-10-14 15:01|variants/p03_01_ext0.5_full_lgb_extra31_lgb_extra31.csv"
  "2026-10-14 20:01|variants/p03_02_ext1.0_v4_fac_xgb_full.csv"
  "2026-10-15 07:01|variants/p03_03_ext1.0_full_lgb_shallow_lgb_shallow.csv"
  "2026-10-15 12:01|variants/p03_04_ext1.0_full_lgb_extra_lgb_extra.csv"
  "2026-10-15 18:01|variants/p03_05_ext3.0_v4_full_lean.csv"
  "2026-10-16 07:01|variants/p03_06_mix2_ext1.0_v4full_faclean_ext1.0_v4_fac_xgb_full.csv"
  "2026-10-16 12:01|variants/p03_07_ext1.0_full_lgb_mc800_lgb_mc800.csv"
  "2026-10-16 18:01|variants/p03_08_new_full_lgb_extra31.csv"
  "2026-10-17 07:01|variants/p03_09_ext0.5_v4_fac3_cat_full.csv"
  "2026-10-17 12:01|variants/p03_10_ext1.0_v4_fac_lgb_full.csv"
  "2026-10-17 18:01|variants/p03_11_ext2.0_v4_fac_full_only.csv"
  "2026-10-18 07:01|variants/p03_12_ext1.0_full_xgb_deep_xgb_deep.csv"
  "2026-10-18 12:01|variants/p03_13_ext1.0_lgb_extra_v4_lgb_only.csv"
  "2026-10-18 18:01|variants/p03_14_ext1.0_v4_fac_main.csv"
  "2026-10-19 07:01|variants/p03_15_ext1.5_v4_fac_xgb_full.csv"
  "2026-10-19 12:01|variants/p03_16_ext1.0_v4full_faclean.csv"
  "2026-10-19 18:01|variants/p03_17_ext1.0_full_lgb_extra31_lgb_extra31.csv"
  "2026-10-20 07:01|variants/p03_18_ext1.0_full_xgb_shallow_xgb_shallow.csv"
  "2026-10-20 12:01|variants/p03_19_new_full_lgb_extra.csv"
  "2026-10-20 18:01|variants/p03_20_ext2.5_v4_fac3_full_only.csv"
  "2026-10-21 07:01|variants/p03_21_ext0.5_xgb_deep_v4_xgb_only.csv"
  "2026-10-21 12:01|variants/p03_22_ext2.0_v4_fac_lgb_full.csv"
  "2026-10-21 18:01|variants/p03_23_new_full_cat_deep.csv"
  "2026-10-22 07:01|variants/p03_24_ext1.5_v4_fac3_main.csv"
  "2026-10-22 12:01|variants/p03_25_ext1.5_v4_fac3_cat_full.csv"
  "2026-10-22 18:01|variants/p03_26_ext0.5_full_lgb_ff06_lgb_ff06.csv"
  "2026-10-23 07:01|variants/p03_27_ext0.5_full_lgb_extra_lgb_extra.csv"
  "2026-10-23 12:01|variants/p03_28_new_full_xgb_deep.csv"
  "2026-10-23 18:01|variants/p03_29_new_lgb_mc800.csv"
  "2026-10-24 07:01|variants/p03_30_ext3.0_v4_fac_lgb_full.csv"
  "2026-10-24 12:01|variants/p03_31_new_cat_deep.csv"
  "2026-10-24 18:01|variants/p03_32_mix2_ext1.0_v4_fac_xgb_full_ext0.5_v4_fac3_xgb_full.csv"
  "2026-10-25 07:01|variants/p03_33_ext1.5_v4_fac3_xgb_full.csv"
  "2026-10-25 12:01|variants/p03_34_ext1.5_v4_full_lean.csv"
  "2026-10-25 18:01|variants/p03_35_new_lgb_extra31.csv"
  "2026-10-26 07:01|variants/p03_36_new_lgb_deep.csv"
  "2026-10-26 12:01|variants/p03_37_ext0.5_full_lgb_shallow_lgb_shallow.csv"
  "2026-10-26 18:01|variants/p03_38_ext0.5_v4_fac_top_only.csv"
  "2026-10-27 07:01|variants/p03_39_ext0.5_v4_fac_lgb_only.csv"
  "2026-10-27 12:01|variants/p03_40_ext0.5_full_cat_deep_cat_deep.csv"
  "2026-10-27 18:01|variants/p03_41_mix2_ext2.0_v4_fac_full_only_mix_full_xgb_deep.csv"
  "2026-10-28 07:01|variants/p03_42_mix2_ext2.0_v4_fac_full_only_ext1.0_v4full_faclean.csv"
  "2026-10-28 12:01|variants/p03_43_ext2.0_v4_fac_xgb_full.csv"
  "2026-10-28 18:01|variants/p03_44_new_cat_lossguide.csv"
  "2026-10-29 07:01|variants/p03_45_new_lgb_ff015.csv"
  "2026-10-29 12:01|variants/p03_46_ext1.5_v4_fac_xgb_only.csv"
  "2026-10-29 18:01|variants/p03_47_new_lgb_extra.csv"
  "2026-10-30 07:01|variants/p03_48_ext1.0_full_lgb_ff06_lgb_ff06.csv"
  "2026-10-30 12:01|variants/p03_49_ext2.0_v4_fac3_full_only.csv"
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
  # a failed submit (no internet after waking up) is retried every 5 minutes for up to 3 hours
  ok=0
  for t in {1..36}; do
    $K competitions submit -f $f -m "harvest ${f:t:r}" && { ok=1; break }
    echo "$(date '+%F %T') submit failed for $f, retry in 5 minutes"; sleep 300
  done
  (( ok )) || continue
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
