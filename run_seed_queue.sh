#!/bin/zsh
# Runs the fold seed variants one after another once the seed 7 run is done.
cd /Users/yutaaa/Developments/dsc-logikaui
while pgrep -f notebook_s7.ipynb >/dev/null; do sleep 60; done
for s in 13 21; do
  (cd run_s$s && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook_s$s.ipynb > ../logs/nb_s$s.log 2>&1)
  echo "seed $s exit $? $(date)"
done
