#!/bin/zsh
# Re-plans the submit queue whenever new information arrives: a new public score in PROGRESS.md, or a finished training run.
cd /Users/yutaaa/Developments/dsc-logikaui
S=/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/
info() { print -- $(grep -c ': public 0\.' PROGRESS.md) $(cat logs/diverse_train*.log logs/nb_v4_s21.log 2>/dev/null | grep -cE 'pass done|v4_s21 exit 0') }
requeue() {
  NSLOTS=$(.venv/bin/python $S/slots.py count) PREFIX=$1 .venv/bin/python $S/harvest_kf.py $S > logs/harvest_kf_$1.log 2>&1 || { echo "$(date '+%F %T') simulation $1 failed"; return 1 }
  # never swap the scheduler from 15 minutes before to 40 minutes after a submit slot, nor while it retries a failed submit;
  # a slot passing meanwhile fails the rebuild, and the next loop plans again
  until .venv/bin/python $S/slots.py busy && ! tail -1 logs/scheduled_unified.log | grep -q "retry in 5 minutes"; do sleep 60; done
  PREFIX=$1 .venv/bin/python $S/rebuild_qc.py $S scheduled_unified.next || { echo "$(date '+%F %T') rebuild $1 failed"; return 1 }
  zsh -n scheduled_unified.next || return 1
  pkill -f "\./scheduled_unifie[d]\.sh$"; sleep 2
  cp logs/scheduled_unified.log logs/scheduled_unified_before_$1.log
  chmod +x scheduled_unified.next; mv scheduled_unified.next scheduled_unified.sh
  (nohup caffeinate -i ./scheduled_unified.sh > logs/scheduled_unified.log 2>&1 &)
  git add scheduled_unified.sh && git commit -q -m "Requeue $1" && git push -q
  echo "$(date '+%F %T') requeued with $1: $(grep -m1 -o 'variants/[^"]*' scheduled_unified.sh)"
}
# a restart re-plans at once and continues the prefix numbering
last=""; i=1; while ls variants/p$(printf %02d $i)_* >/dev/null 2>&1; do (( i++ )); done
while true; do
  now=$(info)
  if [[ $now != $last ]]; then
    echo "$(date '+%F %T') new information ($now), planning p$(printf %02d $i)"
    requeue p$(printf %02d $i) && last=$now || sleep 600
    (( i++ ))
  fi
  sleep 120
done
