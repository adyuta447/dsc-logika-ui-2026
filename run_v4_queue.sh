#!/bin/zsh
# Runs the season and rhythm feature notebooks one after another.
cd /Users/yutaaa/Developments/dsc-logikaui
while pgrep -f "run_seed_queu[e].sh" >/dev/null; do sleep 60; done
for r in v4_s42 v4_s7 v4_s13 v4_lr005; do
  (cd run_$r && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb > ../logs/nb_$r.log 2>&1)
  echo "$r exit $? $(date)"
done
