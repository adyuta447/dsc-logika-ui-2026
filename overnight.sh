#!/bin/zsh
# Overnight: after each training pass, rerun the public split simulation and put its picks on the remaining submit slots.
cd /Users/yutaaa/Developments/dsc-logikaui
S=/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/
requeue() {
  local now=$(date +%H%M)
  if [[ $now > 0630 && $now < 0715 ]]; then echo "$(date '+%F %T') skip $1, too close to the 07:01 submit"; return; fi
  B=200000 KEEP=3000 NSLOTS=$(.venv/bin/python $S/slots.py count) PREFIX=$1 .venv/bin/python $S/harvest_bayes.py $S > logs/harvest_bayes_$1.log 2>&1 || { echo "$(date '+%F %T') simulation $1 failed"; return; }
  # the new scheduler is written to a separate file first, so a failed rebuild leaves the running one alone
  PREFIX=$1 .venv/bin/python $S/rebuild_qc.py $S scheduled_unified.next || { echo "$(date '+%F %T') rebuild $1 failed"; return; }
  zsh -n scheduled_unified.next || return
  pkill -f "scheduled_unifie[d].sh"; sleep 2
  cp logs/scheduled_unified.log logs/scheduled_unified_before_$1.log
  mv scheduled_unified.next scheduled_unified.sh
  (nohup caffeinate -i ./scheduled_unified.sh > logs/scheduled_unified.log 2>&1 &)
  git add scheduled_unified.sh && git commit -q -m "Overnight requeue $1" && git push -q
  echo "$(date '+%F %T') requeued with $1: $(grep -m1 -o 'variants/[^"]*' scheduled_unified.sh)"
}
while kill -0 26113 2>/dev/null; do sleep 60; done
requeue qd
until grep -q "fourth pass done" logs/diverse_train4.log 2>/dev/null; do sleep 60; done
requeue qe
PASS5=1 PYTHONPATH=. .venv/bin/python $S/diverse_train.py $S >> logs/diverse_train5.log 2>&1 &
p=$!; ./battery_guard.sh $p > logs/battery_guard_div5.log 2>&1 &
wait $p
requeue qf
echo "$(date '+%F %T') overnight done"
