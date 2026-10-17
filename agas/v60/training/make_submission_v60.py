"""Build conservative candidate submissions around the proven public leader.

The public-best file is retained as the anchor. New candidates add small,
rank-space perturbations from independent local models. This avoids the
large, unvalidated extrapolation that produced public 0.557 in v57.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


SCRIPT = Path(__file__).resolve()
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
AGAS = paths.AGAS
OUT = paths.version_output(SCRIPT)
OUT.mkdir(parents=True, exist_ok=True)
BASE = AGAS / "qc_00_ext1.5_v4_fac_full_only.csv"


def rank01(x):
    return rankdata(x) / len(x)


def load(path):
    return pd.read_csv(path).sort_values("application_id").reset_index(drop=True)


def main():
    base = load(BASE)
    test = pd.read_csv(ROOT / "data/test_applications.csv").sort_values("application_id").reset_index(drop=True)
    candidates = {
        "v58_ext15": load(paths.version_output_of("v58ext") / "submission_v58ext_ext1.5_v4_fac_full_only.csv"),
        "v58_ext1": load(paths.version_output_of("v58ext") / "submission_v58ext_ext1_v4_fac_full_only.csv"),
        "v58_v4": load(paths.version_output_of("v58ext") / "submission_v58ext_v4_full_only.csv"),
        "v57_top": load(paths.version_output_of("v57") / "submission_v57_top_mix.csv"),
        "v59_xgb": load(paths.version_output_of("v59") / "submission_v59boost_xgb_lossguide.csv"),
    }
    for name, frame in candidates.items():
        assert np.array_equal(frame.application_id, base.application_id), name

    b = rank01(base.default_probability.to_numpy())
    r = {name: rank01(frame.default_probability.to_numpy()) for name, frame in candidates.items()}
    predictions = {
        "anchor_qc": b,
        "qc_98_v58_ext15_2_v58_v4": 0.98 * b + 0.02 * r["v58_v4"],
        "qc_95_v58_ext15_5_v58_v4": 0.95 * b + 0.05 * r["v58_v4"],
        "qc_98_v58_ext1_2_v58_v4": 0.98 * b + 0.02 * r["v58_ext1"],
        "qc_95_v58_ext1_5_v58_v4": 0.95 * b + 0.05 * r["v58_ext1"],
        "qc_98_v59_xgb_2": 0.98 * b + 0.02 * r["v59_xgb"],
        "qc_95_v59_xgb_5": 0.95 * b + 0.05 * r["v59_xgb"],
        "mix_v58_ext1_v4": 0.50 * r["v58_ext1"] + 0.50 * r["v58_v4"],
        "mix_qc_v58_ext1": 0.50 * b + 0.50 * r["v58_ext1"],
        "mix_qc_v58_v4": 0.50 * b + 0.50 * r["v58_v4"],
    }

    files = {}
    for name, pred in predictions.items():
        ordered = pd.DataFrame({
            "application_id": base.application_id,
            "default_probability": rank01(pred),
        })
        path = OUT / f"submission_v60_{name}.csv"
        ordered.to_csv(path, index=False)
        check = pd.read_csv(path)
        assert len(check) == len(test)
        assert check.application_id.is_unique
        assert set(check.application_id) == set(test.application_id)
        assert check.default_probability.between(0, 1).all()
        files[name] = str(path)

    correlations = {
        name: float(spearmanr(b, pred).statistic)
        for name, pred in predictions.items()
    }
    summary = {
        "anchor": str(BASE),
        "anchor_public_score_known": 0.56171,
        "method": "conservative rank-space perturbations around proven public anchor",
        "warning": "No public score can be guaranteed without Kaggle submission.",
        "files": files,
        "spearman_to_anchor": correlations,
    }
    (OUT / "summary_v60.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
