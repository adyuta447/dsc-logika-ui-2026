#!/bin/zsh
# Waits for overnight.sh to finish its first requeue, then hands the rest of the night to overnight2.sh.
cd /Users/yutaaa/Developments/dsc-logikaui
until grep -qE "with qd|qd failed|skip qd" logs/overnight.log 2>/dev/null; do sleep 30; done
sleep 5; pkill -f "/overnigh[t]\.sh"; sleep 2
nohup caffeinate -i ./overnight2.sh >> logs/overnight.log 2>&1 &
echo "$(date '+%F %T') handed over to overnight2.sh" >> logs/overnight.log
