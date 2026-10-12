#!/bin/zsh
# Runs the learning rate 0.005 notebook once the fold seed queue has finished.
cd /Users/yutaaa/Developments/dsc-logikaui
while pgrep -f run_seed_queue.sh >/dev/null; do sleep 60; done
(cd run_lr005 && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook_lr005.ipynb > ../logs/nb_lr005.log 2>&1)
echo "lr005 exit $? $(date)"
