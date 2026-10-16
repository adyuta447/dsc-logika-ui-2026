"""Shared path resolution for every script under `agas/`.

Layout after the v62 restructure:

    agas/
      README.md, progress.md, limit.md
      qc_00_ext1.5_v4_fac_full_only.csv     historical public anchor, 0.56171
      shared/
        paths.py                             this file
        feat_extra.py, build_extra.py, cv_harness.py, factor2.py
        cache/                               feature caches used by every version
      v57/ v58ext/ v59/ v60/ v61/ v62/
        training/                            scripts producing this version
        notebook/                            reproducible notebooks
        output/                              submissions, predictions, summaries, logs

Every script therefore needs three things: the repository root so `features.py`
and `data/` resolve, its own version output directory, and the shared cache for
feature frames. `version_of` and `version_output` derive the middle one from the
script location so no script has to hardcode its own folder name.
"""

from pathlib import Path

SCRIPT = Path(__file__).resolve()

AGAS = SCRIPT.parents[1]           # agas/
ROOT = AGAS.parent                 # repository root
SHARED = AGAS / "shared"
CACHE = SHARED / "cache"
ANCHOR = AGAS / "qc_00_ext1.5_v4_fac_full_only.csv"

FEATURES_CACHE = CACHE / "features_cache.pkl"
EXTRA_CACHE = CACHE / "extra_cache.pkl"
EXBLOCKS = CACHE / "exblocks"

# scripts sit in agas/<version>/training, tools sit directly in agas/shared
SUBDIRS = ("training", "notebook")


def version_dir(script_path=None):
    """Directory that owns a script: `agas/<version>` or `agas/shared`."""
    p = Path(script_path or SCRIPT).resolve()
    return p.parent.parent if p.parent.name in SUBDIRS else p.parent


def version_output(script_path=None):
    """Output directory that owns a script."""
    return version_dir(script_path) / "output"


def version_output_of(name):
    """Output directory of a named version, for example `v61`."""
    return AGAS / name / "output"


def version_notebook(name):
    return AGAS / name / "notebook"


def training_dir(name):
    return AGAS / name / "training"
