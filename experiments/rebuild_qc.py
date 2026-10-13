import glob, re, subprocess, sys
# Puts the new qc files on the remaining slots of scheduled_unified.sh (run only while the scheduler is stopped).
S = sys.argv[1]
files = sorted(glob.glob(f"variants/{__import__('os').environ.get('PREFIX', 'qc')}_*.csv"))
slots = [s for s in subprocess.run([sys.executable, S + "slots.py"], capture_output=True, text=True).stdout.split("\n") if s]
assert len(files) == len(slots), (len(files), len(slots))
queue = "queue=(\n" + "".join(f'  "{s}|{f}"\n' for s, f in zip(slots, files)) + ")\n"
src = open("scheduled_unified.sh").read()
new = re.sub(r"queue=\(\n.*?\n\)\n", lambda m: queue, src, count=1, flags=re.S)
assert new != src and "our_rank() { d=$(mktemp -d)" in new
open(sys.argv[2] if len(sys.argv) > 2 else "scheduled_unified.sh", "w").write(new)
print(len(files), "files from", slots[0], files[0])
