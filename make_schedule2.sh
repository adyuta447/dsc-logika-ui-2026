#!/bin/zsh
# After the second diverse training pass: rerun the harvest selection on every model, then start the second submit queue.
cd /Users/yutaaa/Developments/dsc-logikaui
until grep -q "second pass done" logs/diverse_train.log; do sleep 60; done
rm -f variants/harvest2_*.csv
sed -i '' 's/for step in range(20):/for step in range(37):/' /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/harvest3.py
.venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/harvest3.py /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/ > logs/harvest3.log 2>&1
.venv/bin/python /private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/build_queue.py scheduled_harvest2.sh >> logs/harvest3.log 2>&1
chmod +x scheduled_harvest2.sh && zsh -n scheduled_harvest2.sh && nohup caffeinate -i ./scheduled_harvest2.sh > logs/scheduled_harvest2.log 2>&1 &
