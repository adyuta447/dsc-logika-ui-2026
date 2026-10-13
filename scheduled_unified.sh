#!/bin/zsh
# Unified queue (extrapolations, new model types, variants): submits one file at a time (three a day, hours apart), checking the leaderboard before and after each.
cd /Users/yutaaa/Developments/dsc-logikaui
K=.venv/bin/kaggle
queue=(
  "2026-10-14 10:01|variants/qd_00_mix2_ext2.0_v4_fac_full_only_new_full_lgb_extra.csv"
  "2026-10-14 15:01|variants/qd_01_ext0.5_full_lgb_extra_lgb_extra.csv"
  "2026-10-14 20:01|variants/qd_02_ext1.0_lgb_extra_v4_lgb_only.csv"
  "2026-10-15 07:01|variants/qd_03_ext1.0_full_lgb_extra31_lgb_extra31.csv"
  "2026-10-15 12:01|variants/qd_04_ext2.0_v4_fac_full_only.csv"
  "2026-10-15 18:01|variants/qd_05_ext1.0_full_lgb_shallow_lgb_shallow.csv"
  "2026-10-16 07:01|variants/qd_06_ext0.5_full_cat_deep_cat_deep.csv"
  "2026-10-16 12:01|variants/qd_07_v4_lgb_only.csv"
  "2026-10-16 18:01|variants/qd_08_ext0.5_full_lgb_mc800_lgb_mc800.csv"
  "2026-10-17 07:01|variants/qd_09_new_cat_deep.csv"
  "2026-10-17 12:01|variants/qd_10_new_full_lgb_extra.csv"
  "2026-10-17 18:01|variants/qd_11_ext1.5_v4_fac3_main.csv"
  "2026-10-18 07:01|variants/qd_12_mix2_ext2.0_v4_fac_full_only_mix_full_lgb_extra31.csv"
  "2026-10-18 12:01|variants/qd_13_ext2.5_v4_fac3_full_only.csv"
  "2026-10-18 18:01|variants/qd_14_mix_full_lgb_shallow.csv"
  "2026-10-19 07:01|variants/qd_15_ext0.5_xgb_deep_v4_xgb_only.csv"
  "2026-10-19 12:01|variants/qd_16_ext1.5_v4_fac3_full_only.csv"
  "2026-10-19 18:01|variants/qd_17_ext1.5_v4_fac3_xgb_only.csv"
  "2026-10-20 07:01|variants/qd_18_ext0.5_v4_fac3_full_only.csv"
  "2026-10-20 12:01|variants/qd_19_ext2.0_v4_fac3_lean_only.csv"
  "2026-10-20 18:01|variants/qd_20_new_full_lgb_extra31.csv"
  "2026-10-21 07:01|variants/qd_21_ext1.5_v4_fac_main.csv"
  "2026-10-21 12:01|variants/qd_22_ext0.5_v4_fac3_xgb_only.csv"
  "2026-10-21 18:01|variants/qd_23_fac3_top_only.csv"
  "2026-10-22 07:01|variants/qd_24_ext0.5_full_xgb_deep_xgb_deep.csv"
  "2026-10-22 12:01|variants/qd_25_mix2_ext1.5_v4_fac_full_only_ext2.0_v4_fac_full_only.csv"
  "2026-10-22 18:01|variants/qd_26_mix_xgb_shallow.csv"
  "2026-10-23 07:01|variants/qd_27_mix2_ext2.0_v4_fac_full_only_new_full_lgb_mc800.csv"
  "2026-10-23 12:01|variants/qd_28_ext0.5_full_lgb_shallow_lgb_shallow.csv"
  "2026-10-23 18:01|variants/qd_29_mix_full_cat_deep.csv"
  "2026-10-24 07:01|variants/qd_30_new_lgb_mc800.csv"
  "2026-10-24 12:01|variants/qd_31_new_lgb_goss.csv"
  "2026-10-24 18:01|variants/qd_32_mix_full_lgb_extra31.csv"
  "2026-10-25 07:01|variants/qd_33_ext0.5_full_lgb_extra31_lgb_extra31.csv"
  "2026-10-25 12:01|variants/qd_34_new_full_cat_deep.csv"
  "2026-10-25 18:01|variants/qd_35_ext2.0_v4_fac_lgb_only.csv"
  "2026-10-26 07:01|variants/qd_36_ext1.0_full_lgb_extra_lgb_extra.csv"
  "2026-10-26 12:01|variants/qd_37_new_lgb_ff06.csv"
  "2026-10-26 18:01|variants/qd_38_ext1.0_full_xgb_shallow_xgb_shallow.csv"
  "2026-10-27 07:01|variants/qd_39_ext1.0_full_lgb_ff06_lgb_ff06.csv"
  "2026-10-27 12:01|variants/qd_40_ext0.5_lgb_extra_v4_lgb_only.csv"
  "2026-10-27 18:01|variants/qd_41_ext1.5_v4_fac_top_mix.csv"
  "2026-10-28 07:01|variants/qd_42_ext1.0_v4_fac_lgb_only.csv"
  "2026-10-28 12:01|variants/qd_43_ext2.0_v4_fac3_full_only.csv"
  "2026-10-28 18:01|variants/qd_44_ext1.0_full_xgb_deep_xgb_deep.csv"
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
