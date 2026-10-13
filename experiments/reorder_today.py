import glob, re, subprocess, sys
NEW_RANK = 'our_rank() { d=$(mktemp -d); $K competitions leaderboard -d -p $d >/dev/null 2>&1; unzip -o -q $d/*.zip -d $d 2>/dev/null\n  .venv/bin/python -c "import pandas as pd,glob,sys; t=pd.read_csv(glob.glob(sys.argv[1]+\'/*.csv\')[0]); r=t.Rank[t.TeamName==\'kali ini beneran deh\']; print(int(r.iloc[0]) if len(r) else \'\')" $d; rm -rf $d }'
# Rebuilds the unified queue for the slots still ahead: the two tail picks first, then the rest in their original order, skipping what was sent.
S = sys.argv[1]
sent = set(re.findall(r"(queue_\d+_[\w.]+?\.csv) public", open("logs/scheduled_unified.log").read()))
files = sorted(glob.glob("variants/queue_*.csv"))
first = ["variants/queue_03_ext2.0_v4_fac_full_only.csv", "variants/queue_10_ext1.0_v4_fac_lgb_only.csv"]
order = [f for f in first if f.split("/")[-1] not in sent] + [f for f in files if f not in first and f.split("/")[-1] not in sent]
slots = [s for s in subprocess.run([sys.executable, S + "slots.py"], capture_output=True, text=True).stdout.split("\n") if s]
queue = "queue=(\n" + "".join(f'  "{s}|{f}"\n' for s, f in zip(slots, order)) + ")\n"
src = re.sub(r"queue=\(\n.*?\n\)\n", lambda m: queue, open("scheduled_unified.sh").read(), count=1, flags=re.S)
# the shown leaderboard stops at 20 teams, so the rank comes from the full download
src = re.sub(r"our_rank\(\) \{.*?\}\n", lambda m: NEW_RANK + "\n", src, count=1, flags=re.S)
open("scheduled_unified.sh", "w").write(src)
print("sent:", sorted(sent), "| next:", list(zip(slots, order))[:3], "| queued", min(len(slots), len(order)))
