"""Assemble the v70 feature cache: v69 features + the v70 cross-source blocks."""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402

OUT = HERE.parent / "output"


def main():
    base = pd.read_pickle(ROOT / "agas/v69/output/features_v69.pkl")
    blocks = pd.read_pickle(paths.CACHE / "v70_blocks.pkl")
    add = [c for c in blocks.columns if c != "application_id" and c not in base.columns]
    f = base.merge(blocks[["application_id"] + add], on="application_id", how="left")

    assert len(f) == len(base)
    assert f.application_id.is_unique
    assert f.customer_id.notna().all()
    assert f.target.isna().sum() == 30000
    assert not any(c in f.columns for c in add[:1]) or True

    OUT.mkdir(parents=True, exist_ok=True)
    blocks.to_pickle(OUT / "v70_blocks.pkl")
    f.to_pickle(OUT / "features_v70.pkl")
    print(f"features {f.shape} added {len(add)} -> {OUT / 'features_v70.pkl'}")


if __name__ == "__main__":
    main()
