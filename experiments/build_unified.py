import glob, re, subprocess, sys
# Writes the unified scheduler: the submit loop of scheduled_harvest.sh with every remaining slot filled from variants/queue_*.csv.
S = sys.argv[1]
files = sorted(glob.glob("variants/queue_*.csv"))
slots = subprocess.run([sys.executable, S + "slots.py"], capture_output=True, text=True).stdout.split("\n")
queue = "queue=(\n" + "".join(f'  "{s}|{f}"\n' for s, f in zip(slots, files) if s) + ")\n"
src = re.sub(r"queue=\(\n.*?\n\)\n", lambda m: queue, open("scheduled_harvest.sh").read(), count=1, flags=re.S)
src = src.replace("# Submits the diverse variant queue", "# Unified queue (extrapolations, new model types, variants): submits")
open("scheduled_unified.sh", "w").write(src)
print(min(len(files), len([s for s in slots if s])), "files queued from", slots[0])
