"""Assemble the v69 feature cache: v66 base + v66 extra + v69 blocks."""

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
    base = pd.read_pickle(paths.FEATURES_CACHE)
    extra = pd.read_pickle(paths.EXTRA_CACHE)
    blocks_path = paths.CACHE / "v69_blocks.pkl"
    blocks = pd.read_pickle(blocks_path)

    add = [c for c in extra.columns if c != "application_id" and c not in base.columns]
    f = base.merge(extra[["application_id"] + add], on="application_id", how="left")
    add = [c for c in blocks.columns if c != "application_id" and c not in f.columns]
    f = f.merge(blocks[["application_id"] + add], on="application_id", how="left")

    assert len(f) == len(base)
    assert f.application_id.is_unique
    assert f.customer_id.notna().all()
    assert f.target.isna().sum() == 30000

    OUT.mkdir(parents=True, exist_ok=True)
    blocks.to_pickle(OUT / "v69_blocks.pkl")
    f.to_pickle(OUT / "features_v69.pkl")
    print(f"features {f.shape} blocks {blocks.shape} -> {OUT / 'features_v69.pkl'}")


if __name__ == "__main__":
    main()
