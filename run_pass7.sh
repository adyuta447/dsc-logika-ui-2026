#!/bin/zsh
# Starts the seventh training pass once the sixth one has finished, with the battery guard attached.
cd /Users/yutaaa/Developments/dsc-logikaui
S=/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/
until grep -q "sixth pass done" logs/diverse_train6.log 2>/dev/null; do sleep 60; done
PASS7=1 PYTHONPATH=. .venv/bin/python $S/diverse_train.py $S >> logs/diverse_train7.log 2>&1 &
p=$!; ./battery_guard.sh $p > logs/battery_guard_div7.log 2>&1 &
wait $p; echo "seventh pass done $(date)" >> logs/diverse_train7.log
