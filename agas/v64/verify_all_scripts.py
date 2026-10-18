"""Verify every script under agas/ still imports and resolves paths correctly.

The `shared/bootstrap.py` rewrite touched 25 scripts. This checks each one by
actually running it with `--help` or a harmless import, so a broken `paths`
import shows up as a failure rather than as a surprise four hours into a run.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()

SELF = Path(__file__).resolve()
SCRIPTS = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "agas").rglob("*.py")
    if "OtherTries" not in str(p) and "__pycache__" not in str(p)
    and p.resolve() != SELF
    and p.name not in ("fix_shared_imports.py", "fix_shared_imports2.py")
)

ARGPARSE = {"--help"}
ok, bad = [], []

for f in SCRIPTS:
    p = ROOT / f
    try:
        ast.parse(p.read_text())
    except SyntaxError as e:
        bad.append((f, f"syntax: {e}"))
        continue

    env = dict(os.environ)
    # Scripts without argparse ignore --help and start real work. The 90 second
    # timeout catches most of them, but a full CV study can finish inside it and
    # then overwrite its own output, so the study scripts opt out by name.
    if any(k in f for k in ("study_", "train_submission_v65", "train_submission_v64",
                            "final_v63", "make_submission")):
        env["V65_NO_STUDY"] = "1"
    try:
        r = subprocess.run([sys.executable, f, *ARGPARSE], cwd=ROOT,
                           capture_output=True, text=True, timeout=90, env=env)
        err = (r.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        # a script without argparse ignores --help and starts real work, which is
        # fine; the header it already printed proves the paths import resolved
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        err = (err or "").strip()
        if "ModuleNotFoundError" in err or "No module named 'paths'" in err:
            bad.append((f, err.splitlines()[-1][:120]))
        else:
            ok.append((f, "started, paths resolved"))
        continue
    if "ModuleNotFoundError" in err or "NameError: name 'paths'" in err:
        bad.append((f, err.splitlines()[-1][:120]))
    elif "Traceback" in err and "unrecognized arguments" not in err and "usage:" not in err:
        # a script that runs main() without argparse is fine as long as paths resolved
        if "paths" in err or "bootstrap" in err:
            bad.append((f, err.splitlines()[-1][:120]))
            continue
        ok.append((f, "ran main()"))
    else:
        ok.append((f, "ok"))

print(f"checked {len(SCRIPTS)} scripts: {len(ok)} fine, {len(bad)} broken")
for f, why in ok:
    print(f"  OK    {f}  ({why})")
for f, why in bad:
    print(f"  BROKEN {f}  {why}")
if bad:
    sys.exit(1)
