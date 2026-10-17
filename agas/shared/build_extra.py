"""Build per-block caches under `agas/output/exblocks/` so a single new block
can be added without recomputing the slow ecommerce / repayment tables."""

import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
def _add_shared():
    """Load agas/shared/bootstrap.py by path; agas is not an importable package."""
    import importlib.util as _il
    _here = Path(__file__).resolve().parent
    for _p in _here.parents:
        _c = _p / "shared" / "bootstrap.py"
        if _c.exists():
            _spec = _il.spec_from_file_location("_agas_bootstrap", _c)
            _m = _il.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            if str(_c.parent) not in sys.path:
                sys.path.insert(0, str(_c.parent))
            return _p
    raise RuntimeError(f"shared/bootstrap.py not found above {_here}")

_SHARED = _add_shared()
import paths  # noqa: E402
ROOT = paths.ROOT
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import feat_extra  # noqa: E402

CACHE = paths.CACHE / "features_cache.pkl"
BLK = paths.EXBLOCKS
OUT = paths.EXTRA_CACHE


def block_path(name):
    return BLK / f"{name}.pkl"


def load_block(name, app, base, force=False):
    p = block_path(name)
    if p.exists() and not force:
        return pd.read_pickle(p)
    BLK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = feat_extra.build([name], app=app, base=base)
    out.to_pickle(p)
    print(f"  built {name} {out.shape} in {(time.time() - t0) / 60:.2f} min", flush=True)
    return out


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(feat_extra.BLOCKS)
    force = "--force" in sys.argv
    base = pd.read_pickle(CACHE)
    app = pd.read_csv("data/train_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
    te = pd.read_csv("data/test_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
    app = pd.concat([app, te], ignore_index=True)

    t0 = time.time()
    parts = [load_block(n, app, base, force) for n in names]
    extra = parts[0]
    for p in parts[1:]:
        extra = extra.merge(p, on="application_id", how="outer")
    if "interactions" in names:
        parts.append(feat_extra.with_interactions(parts[0], base))
        extra = parts[0]
        for p in parts[1:]:
            extra = extra.merge(p, on="application_id", how="outer")
    extra.to_pickle(OUT)
    print(f"blocks={names} shape={extra.shape} minutes={(time.time() - t0) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()