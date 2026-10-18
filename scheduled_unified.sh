#!/bin/zsh
# Unified queue (mixes of Agas's v61 from experiments/v61_plan.py first, then the p22 plan) (extrapolations, new model types, variants): submits one file at a time (three a day, hours apart), checking the leaderboard before and after each.
cd /Users/yutaaa/Developments/dsc-logikaui
K=.venv/bin/kaggle
files=(
  variants/v61_03_mix0.5_v61_p17_00.csv
  variants/v61_14_mix0.7_v61_av64cat.csv
  variants/v61_12_mix0.5_v61_qc00.csv
  variants/v61_06_mix0.85_v61_p19_00.csv
  variants/team_stack.csv
  variants/p22_07_ext1.0_plus_prof_lgb_extra_lgb_extra.csv
  variants/p22_08_ext1.0_full_lgb_extra_facfull_lgb_extra.csv
  variants/v61_02_mix0.5_v61_p19_00.csv
  variants/v61_13_ext0.5_v61_v63f.csv
  variants/v61_08_ext0.25_v61_p21_00.csv
  variants/v61_07_mix0.85_v61_v4full.csv
  variants/v61_01_mix0.85_v61_qc00.csv
  variants/v61_05_mix0.7_v61_qc00.csv
  variants/v61_10_mix0.85_v61_v63f.csv
  variants/v61_09_mix0.7_v61_v63f.csv
  variants/p22_00_new_minus_ec_xgb_deep.csv
  variants/p22_01_new_minus_time_lgb_extra31.csv
  variants/p22_02_mix2_mix2_ext1.0_v4_fac_xgb_full_ext1.5_v4full_faclean_ext0.5_full_lgb_extra_lgb_extra.csv
  variants/p22_03_ext1.0_plus_prof_lgb_ff015_lgb_ff015.csv
  variants/p22_04_ext1.0_plus_prof_xgb_deep_xgb_deep.csv
  variants/p22_05_ext1.0_full_lgb_ff015_facfull_lgb_ff015.csv
  variants/p22_06_mix2_ext2.0_v4_fac_full_only_new_minus_time_lgb_extra31.csv
  variants/p22_09_ext1.5_v4full_faclean.csv
  variants/p22_10_mix2_new_minus_ec_xgb_deep_ext0.5_full_lgb_extra_lgb_extra.csv
  variants/p22_11_ext1.0_plus_cs_lgb_extra_lgb_extra.csv
  variants/p22_12_ext3.0_v4_full_lean.csv
  variants/p22_13_ext1.0_plus_prof_lgb_shallow_lgb_shallow.csv
  variants/p22_14_mix2_ext2.0_v4_fac_full_only_ext0.5_full_lgb_extra_lgb_extra.csv
  variants/p22_15_new_plus_cs_lgb_ff015.csv
  variants/p22_16_new_full_cat_lossguide.csv
  variants/p22_17_new_minus_time_xgb_deep.csv
  variants/p22_18_mix2_new_minus_ec_xgb_deep_new_minus_time_lgb_extra31.csv
  variants/p22_19_new_plus_cs_lgb_extra.csv
  variants/p22_20_ext2.0_plus_prof_lgb_extra_lgb_extra.csv
  variants/p22_21_ext0.5_full_lgb_ff015_lgb_ff015.csv
  variants/p22_22_ext2.5_v4_4_fac_full_only.csv
  variants/p22_23_ext1.0_v4_fac_lgb_full.csv
  variants/p22_24_ext1.5_v4_4_fac_main.csv
  variants/p22_25_ext1.5_v4_4_full_lean.csv
  variants/p22_26_new_facfull_cat_lossguide.csv
  variants/p22_27_ext1.0_full_lgb_mc800_facfull_lgb_mc800.csv
  variants/p22_28_new_minus_ec_xgb_shallow.csv
  variants/p22_29_ext2.0_plus_prof_lgb_shallow_lgb_shallow.csv
  variants/p22_30_mix2_mix2_ext1.0_v4_fac_xgb_full_ext1.5_v4full_faclean_new_minus_time_lgb_extra31.csv
  variants/p22_31_new_plus_ec_lgb_deep.csv
  variants/p22_32_ext3.0_v4_fac_lgb_full.csv
  variants/p22_33_ext2.0_plus_prof_xgb_shallow_xgb_shallow.csv
  variants/p22_34_ext0.5_v4_fac3_cat_full.csv
  variants/p22_35_ext2.0_full_lgb_extra_minus_prof_lgb_extra.csv
  variants/p22_36_mix2_ext2.0_v4_fac_full_only_new_minus_ec_xgb_deep.csv
  variants/p22_37_new_plus_time_lgb_extra31.csv
  variants/p22_38_ext0.5_full_xgb_lossguide_xgb_lossguide.csv
  variants/p22_39_ext2.0_v4_fac_full_only.csv
  variants/p22_40_mix2_ext1.0_v4_fac_full_only_new_minus_time_lgb_extra31.csv
  variants/p22_41_ext1.5_full_lgb_ff015_facfull_lgb_ff015.csv
)
slots=(${(f)"$(.venv/bin/python experiments/slots.py)"})
valid() { .venv/bin/python -c "
import pandas as pd,sys
s=pd.read_csv(sys.argv[1]);t=pd.read_csv('data/test_applications.csv')
assert len(s)==len(t) and s.application_id.is_unique and set(s.application_id)==set(t.application_id)
assert s.default_probability.between(0,1).all()" "$1" }
our_rank() { d=$(mktemp -d); $K competitions leaderboard -d -p $d >/dev/null 2>&1; unzip -o -q $d/*.zip -d $d 2>/dev/null
  .venv/bin/python -c "import pandas as pd,glob,sys; t=pd.read_csv(glob.glob(sys.argv[1]+'/*.csv')[0]); r=t.Rank[t.TeamName=='kali ini beneran deh']; print(int(r.iloc[0]) if len(r) else '')" $d; rm -rf $d }
j=1
for when in $slots; do
  (( j > $#files )) && break
  f=$files[j]
  target=$(date -j -f "%Y-%m-%d %H:%M" "$when" +%s)
  # compares wall clock time, so a slot missed while the laptop slept runs as soon as it wakes
  while (( $(date +%s) < target )); do sleep 30; done
  echo "$(date '+%F %T') slot $when, rank before: $(our_rank)"
  if ! valid $f; then echo "invalid file $f, skipped"; (( j++ )); continue; fi
  # a failed submit (no internet, or the shared daily quota already used) is retried every 5 minutes for up to 3 hours,
  # then the same file waits for the next slot
  ok=0
  for t in {1..36}; do
    $K competitions submit -f $f -m "harvest ${f:t:r}" && { ok=1; break }
    echo "$(date '+%F %T') submit failed for $f, retry in 5 minutes"; sleep 300
  done
  (( ok )) || { echo "$(date '+%F %T') $f kept for the next slot"; continue }
  (( j++ ))
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
