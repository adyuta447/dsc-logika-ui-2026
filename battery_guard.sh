#!/bin/zsh
# Keeps the final notebook running, but pauses it under 20% on battery so the scheduled submits keep power; resumes on the charger.
k=$1
while kill -0 $k 2>/dev/null; do
  b=$(pmset -g batt); pct=$(echo $b | grep -o '[0-9]*%' | head -1 | tr -d %)
  if [[ $b == *"Battery Power"* ]] && (( pct < 20 )); then kill -STOP $k; else kill -CONT $k; fi
  sleep 60
done
