#!/bin/zsh
# Waits for the 07:01 submit to finish, then restarts the unified scheduler with today's tail optimized order.
cd /Users/yutaaa/Developments/dsc-logikaui
until grep -q "queue_00_ext1.5_v4_fac_trees_only.csv public" logs/scheduled_unified.log; do sleep 20; done
sleep 5
pkill -f "scheduled_unifie[d].sh"; sleep 2
cp logs/scheduled_unified.log logs/scheduled_unified_part1.log
.venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/reorder_today.py /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/ > logs/reorder_today.log 2>&1 && zsh -n scheduled_unified.sh && nohup caffeinate -i ./scheduled_unified.sh > logs/scheduled_unified.log 2>&1 &
