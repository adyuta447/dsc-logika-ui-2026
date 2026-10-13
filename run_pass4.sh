#!/bin/zsh
# Starts the fourth training pass once the third one has finished, with the battery guard attached.
cd /Users/yutaaa/Developments/dsc-logikaui
S=/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/
while kill -0 26113 2>/dev/null; do sleep 30; done
PYTHONPATH=. .venv/bin/python $S/diverse_train.py $S >> logs/diverse_train4.log 2>&1 &
p=$!; ./battery_guard.sh $p > logs/battery_guard_div4.log 2>&1 &
wait $p; echo "fourth pass done $(date)" >> logs/diverse_train4.log
