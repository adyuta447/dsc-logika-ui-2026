"""Check the notebook's full path against the production artefacts, without a 6 h run.

    features  the notebook's feature code, run from the raw CSVs, must equal the
              production cache (features_cache.pkl + extra_cache.pkl) exactly,
              column order included (LightGBM column sampling depends on it).
    trainer   the notebook's trainer and the original v66 script, both on a small
              config (2 folds, fold seed 42, LightGBM seed 0) and the same frame,
              must give identical predictions for all seven legs.

Usage: python3 verify_full_path_v86.py features|trainer
Needs the executed notebook's work/modules folder.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "agas/v86/notebook/work"
CACHE = ROOT / "agas/shared/cache"
sys.path.insert(0, str(WORK / "modules"))

import fe_base, fe_extra, fe_v65  # noqa: E402

for m in (fe_base, fe_extra, fe_v65):
    m.D = f"{ROOT / 'data'}/"

EXTRA_BLOCKS = ["bureau", "repayment", "previous", "account", "ecommerce", "gradenorm",
                "rxtra", "bxtra", "axtra", "ecommerce2", "consistency", "dataquality",
                "stress", "interactions"]


def cached_frame():
    base = pd.read_pickle(CACHE / "features_cache.pkl")
    extra = pd.read_pickle(CACHE / "extra_cache.pkl")
    add = [c for c in extra.columns if c != "application_id" and c not in base.columns]
    return base.merge(extra[["application_id"] + add], on="application_id", how="left")


def check_features():
    base = fe_base.build()
    app = fe_base.load_apps()
    parts = [fe_extra.build([n], app=app, base=base) for n in EXTRA_BLOCKS]
    parts.append(fe_extra.with_interactions(parts[0], base))
    extra = parts[0]
    for p in parts[1:]:
        extra = extra.merge(p, on="application_id", how="outer")
    add = [c for c in extra.columns if c != "application_id" and c not in base.columns]
    built = base.merge(extra[["application_id"] + add], on="application_id", how="left")
    ref = cached_frame()
    assert list(built.columns) == list(ref.columns), "column order differs"
    pd.testing.assert_frame_equal(built, ref, check_exact=True)
    print(f"features: PASS, {built.shape} identical to the production cache")


def check_trainer():
    import trainer
    out = ROOT / "agas/v66/output/preds_smoke86.npz"
    rc = subprocess.call([sys.executable, "agas/v66/training/train_submission_v66.py",
                          "--folds", "2", "--fold-seeds", "42", "--lgb-seeds", "0",
                          "--out-prefix", "smoke86"], cwd=ROOT)
    assert rc == 0
    old = np.load(out, allow_pickle=True)
    new = trainer.train_legs(cached_frame(), fe_base, folds=2, fold_seeds="42", lgb_seeds="0")
    for k in old.files:
        assert np.array_equal(old[k], new[k]), f"{k} differs"
    print(f"trainer: PASS, {len(old.files)} arrays identical to the v66 script")
    for p in ROOT.glob("agas/v66/output/*smoke86*"):
        p.unlink()


if __name__ == "__main__":
    {"features": check_features, "trainer": check_trainer}[sys.argv[1]]()
