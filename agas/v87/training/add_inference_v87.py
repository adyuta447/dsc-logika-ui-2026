"""v87: give the executed committee notebook an inference-only mode.

The committee asks teams whose training runs 30 minutes or more to hand in the
trained artefact with a link and let the notebook load it. Training here takes
7 h, so the notebook now defaults to RUN_MODE = "inference": it fetches the
trained out-of-fold and test predictions of the seven legs (preds_retrained.npz,
written by the full run), checks their SHA-256, and blends them into the
submission. RUN_MODE = "train" keeps the full retraining path unchanged.

Usage: python3 add_inference_v87.py "<public Google Drive link or ''>"
Patches the root notebook in place; run it on the notebook as committed.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "DSC0152_kali ini beneran deh_Notebook.ipynb"
WEIGHTS_SHA256 = hashlib.sha256((ROOT / "preds_retrained.npz").read_bytes()).hexdigest()
URL = sys.argv[1]


def src(text):
    return text.strip("\n").splitlines(keepends=True)


def replace(cell, old, new):
    s = "".join(cell["source"])
    assert old in s, old[:60]
    cell["source"] = src(s.replace(old, new))


nb = json.loads(NB.read_text())
cells = nb["cells"]

replace(cells[0], """Run all cells from top to bottom (*Run All*). Nothing has to be configured and
no file other than this notebook and the competition data is needed.

The notebook builds every feature from the raw CSVs, trains all seven legs,
blends them, and writes the submission. That takes about 6.5 hours on a
10-core CPU; the outputs saved in this notebook are from such a complete run.""",
"""Run all cells from top to bottom (*Run All*). Nothing has to be configured.

Training all seven legs takes about **7 hours** on a 10-core CPU, so the notebook
runs **inference-only** by default (`RUN_MODE = "inference"`, cell 1.1): it
downloads the trained predictions of the seven legs from `WEIGHTS_URL` (a public
Google Drive link), checks their SHA-256, blends them and writes the submission
in a few minutes. Set `RUN_MODE = "train"` to retrain everything from the raw
CSVs instead; that path produced the downloaded file and gives the same
submission.""")

replace(cells[0], "and `training_log.txt`, written", "(plus `training_log.txt` in train mode), written")
replace(cells[0], """| **Input** | the eight competition CSVs, nothing else |""",
"""| **Input** | the eight competition CSVs |
| **Trained model output** | `preds_retrained.npz`, the out-of-fold and test predictions of the seven trained legs (6 MB), downloaded from `WEIGHTS_URL`; a copy next to the notebook or under `/kaggle/input` is used instead when present |""")

replace(cells[2], '''SUBMITTED_SHA256 = "35260db3730a3eec7bc2cce9fb3b540d047c64d5d980e39ab19b07ebf07c733b"''',
f'''SUBMITTED_SHA256 = "35260db3730a3eec7bc2cce9fb3b540d047c64d5d980e39ab19b07ebf07c733b"

RUN_MODE = "inference"  # "inference": load the trained predictions; "train": retrain all legs (~7 h)
WEIGHTS_URL = "{URL}"
WEIGHTS_NAME = "preds_retrained.npz"
WEIGHTS_SHA256 = "{WEIGHTS_SHA256}"''')

cells[21]["source"] = src("""
### 6.2 Train the seven legs, or load them

**`RUN_MODE = "train"`** trains every leg and prints one line per leg and fold
partition with its out-of-fold AUC and KS (42 lines), also written to
`training_log.txt`. The predictions are saved to `preds_retrained.npz` next to
the notebook.

**`RUN_MODE = "inference"`** (default) loads that same file as produced by the
full run: a copy next to the notebook or under `/kaggle/input` if there is one,
otherwise a download from `WEIGHTS_URL`. Its SHA-256 is checked before it is
opened.
""")

replace(cells[22], """t0 = time.time()
preds = trainer.train_legs(frame, fe_base, log=log)
np.savez_compressed(WORK / "preds_retrained.npz", **preds)
log(f"training done in {(time.time() - t0) / 3600:.2f} h")""",
"""def find_weights():
    \"\"\"Trained predictions: a local copy if there is one, else the public link.\"\"\"
    import re
    import urllib.request

    local = [WORK / WEIGHTS_NAME]
    if Path("/kaggle/input").exists():
        local += sorted(Path("/kaggle/input").rglob(WEIGHTS_NAME))
    for p in local:
        if p.exists():
            return p
    m = re.search(r"(?:/d/|id=)([\\w-]{20,})", WEIGHTS_URL)
    url = (f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t"
           if m else WEIGHTS_URL)
    print(f"downloading {WEIGHTS_URL}")
    urllib.request.urlretrieve(url, WORK / WEIGHTS_NAME)
    return WORK / WEIGHTS_NAME


if RUN_MODE == "train":
    t0 = time.time()
    preds = trainer.train_legs(frame, fe_base, log=log)
    np.savez_compressed(WORK / WEIGHTS_NAME, **preds)
    log(f"training done in {(time.time() - t0) / 3600:.2f} h")
else:
    weights = find_weights()
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == WEIGHTS_SHA256, f"{weights} is not the trained file (sha256 {digest})"
    preds = dict(np.load(weights, allow_pickle=True))
    print(f"loaded {weights.name}: sha256 {digest[:12]}..., {len(preds)} arrays, "
          f"{len(preds['train_ids']):,} train and {len(preds['test_ids']):,} test rows")""")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"patched {NB.name}; weights sha256 {WEIGHTS_SHA256}")
