#!/bin/zsh
# Runs the final notebook (three fold splits) and checks its submission against the submitted bag file.
cd /Users/yutaaa/Developments/dsc-logikaui/run_final
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb > ../logs/nb_final.log 2>&1
echo "exit $? $(date)"
cmp "DSC0152_kali ini beneran deh_Submisi.csv" ../variants/bag_run_v4_s42_top_mix_3.csv && echo "bit identical to the submitted bag v4 top_mix"
