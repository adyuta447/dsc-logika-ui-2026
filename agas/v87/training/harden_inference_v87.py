"""v87: make the inference-only run independent of the library versions.

* preds_retrained.npz stored its ids as pickled object arrays, which an older
  numpy may fail to unpickle; they become plain unicode arrays, loaded without
  allow_pickle. WEIGHTS_SHA256 follows the rewritten file.
* the feature frame (about 5 min, written against pandas 3.0) is only needed to
  train, so inference skips it.

Usage: python3 harden_inference_v87.py   (after add_inference_v87.py)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "DSC0152_kali ini beneran deh_Notebook.ipynb"
NPZ = ROOT / "preds_retrained.npz"

z = dict(np.load(NPZ, allow_pickle=True))
old = {k: v.copy() for k, v in z.items()}
for k in ("train_ids", "test_ids"):
    z[k] = z[k].astype(str)
np.savez_compressed(NPZ, **z)
back = np.load(NPZ)  # no allow_pickle: must open without it
for k in old:
    assert np.array_equal(back[k], old[k].astype(back[k].dtype)), k
sha = hashlib.sha256(NPZ.read_bytes()).hexdigest()


def src(text):
    return text.strip("\n").splitlines(keepends=True)


def replace(cell, a, b):
    s = "".join(cell["source"])
    assert a in s, a[:60]
    cell["source"] = src(s.replace(a, b))


nb = json.loads(NB.read_text())
cells = nb["cells"]
s2 = "".join(cells[2]["source"])
prev = s2.split('WEIGHTS_SHA256 = "')[1].split('"')[0]
replace(cells[2], prev, sha)
replace(cells[22], "np.load(weights, allow_pickle=True)", "np.load(weights)")

s18 = "".join(cells[18]["source"])
assert s18.startswith("t0 = time.time()\nframe = build_feature_frame()")
body = "".join("    " + l if l.strip() else l for l in s18.splitlines(keepends=True))
cells[18]["source"] = src('if RUN_MODE == "train":\n' + body + '\nelse:\n'
                          '    print("inference mode: the feature frame is only needed to train, skipped")')

md = "".join(cells[17]["source"])
cells[17]["source"] = src(md + "\n\nIn inference mode this step is skipped: the frame is only used to train.")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"npz rewritten without pickled ids, sha256 {sha}")
