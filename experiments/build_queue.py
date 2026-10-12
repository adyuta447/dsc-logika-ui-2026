import glob, re, sys
# Writes scheduled_harvest2.sh: the same submit loop as scheduled_harvest.sh with a queue of the harvest2 files, three slots a day.
out = sys.argv[1] if len(sys.argv) > 1 else "scheduled_harvest2.sh"
files = sorted(glob.glob("variants/harvest2_*.csv"))
slots = ["2026-10-16 18:01"] + [f"2026-10-{d} {h}" for d in range(17, 29) for h in ("07:01", "12:01", "18:01")]
queue = "queue=(\n" + "".join(f'  "{s}|{f}"\n' for s, f in zip(slots, files)) + ")\n"
src = open("scheduled_harvest.sh").read()
src = re.sub(r"queue=\(\n.*?\n\)\n", lambda m: queue, src, count=1, flags=re.S)
src = src.replace("# Submits the diverse variant queue", "# Second queue: submits the new model variants")
open(out, "w").write(src)
print(len(files), "files queued, first", slots[0], "last", slots[min(len(files), len(slots)) - 1])
