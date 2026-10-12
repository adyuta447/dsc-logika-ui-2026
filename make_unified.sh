#!/bin/zsh
# After the second training pass: pick the remaining submissions with the unified simulation, then swap the first scheduler for the unified one.
cd /Users/yutaaa/Developments/dsc-logikaui
until grep -q "second pass done" logs/diverse_train.log; do sleep 60; done
rm -f variants/queue_*.csv
NSLOTS=$(.venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/slots.py count) .venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/harvest_unified.py /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/ > logs/harvest_unified.log 2>&1
.venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/build_unified.py /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/ >> logs/harvest_unified.log 2>&1
chmod +x scheduled_unified.sh && zsh -n scheduled_unified.sh || exit 1
pkill -f "scheduled_harves[t].sh"
nohup caffeinate -i ./scheduled_unified.sh > logs/scheduled_unified.log 2>&1 &
echo "unified scheduler started $(date)" >> logs/harvest_unified.log
