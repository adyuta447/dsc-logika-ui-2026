#!/bin/zsh
# After the final notebook check, adds fold seed 21 to v4 so v4 and fac have the same four fold seeds for the extrapolations.
cd /Users/yutaaa/Developments/dsc-logikaui
until grep -q "^exit" logs/run_final.log 2>/dev/null; do sleep 60; done
mkdir -p run_v4_s21 && ln -sfn ../data run_v4_s21/data
.venv/bin/python run1/make_nb_before_bag.py run_v4_s21/notebook.ipynb 10 1 21 0.01 || exit 1
cd run_v4_s21
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb > ../logs/nb_v4_s21.log 2>&1 &
p=$!; sleep 90; k=$(pgrep -P $p | head -1)
[[ -n $k ]] && ../battery_guard.sh $k > ../logs/battery_guard_s21.log 2>&1 &
wait $p; echo "v4_s21 exit $? $(date)" >> ../logs/nb_v4_s21.log
