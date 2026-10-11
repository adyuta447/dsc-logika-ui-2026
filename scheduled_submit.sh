#!/bin/zsh
# Submits next_submission.csv once the Kaggle daily quota resets (00:00 UTC = 07:00 WIB).
cd /Users/yutaaa/Developments/dsc-logikaui
target=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-11 07:01:00" +%s)
sleep $(( target - $(date +%s) ))
date
if [[ ! -f next_submission.csv ]]; then echo "no next_submission.csv, skipped"; exit 0; fi
msg=$(cat next_submission.txt 2>/dev/null || echo "scheduled submission")
.venv/bin/kaggle competitions submit -f next_submission.csv -m "$msg"
sleep 60
.venv/bin/kaggle competitions submissions | head -5
.venv/bin/kaggle competitions leaderboard -s | head -12
