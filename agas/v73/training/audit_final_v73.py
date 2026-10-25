"""Final v73 audit: validate the frozen v66_04 anchor.

This script deliberately does not train anything. It checks the exact CSV that
is proposed for the final submission, rebuilds the same rank vector from the
stored v66 predictions, and writes a small manifest under ``agas/v73/output``.

Run from the repository root:

    python3 agas/v73/training/audit_final_v73.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "data").is_dir(), f"repo root salah: {ROOT}"
assert (ROOT / "agas").is_dir(), f"repo root salah: {ROOT}"

OUT = ROOT / "agas/v73/output"
ANCHOR = ROOT / "agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv"
PREDS = ROOT / "agas/v66/output/preds_v66.npz"
V72_CONTROL = ROOT / "agas/v72/output/submission_v72_01_anchor.csv"
V73_COPY = OUT / "submission_v73_01_anchor.csv"
AUDIT_JSON = OUT / "audit_v73.json"

EXPECTED_ANCHOR_SHA256 = (
    "acfb8ea882a92a414e338e89801ad7602e3c6c0faa9917887f6e750f0853c5c9"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values) / len(values)


def load_aligned(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["application_id", "default_probability"]:
        raise AssertionError(f"kolom tidak sesuai di {path}: {list(frame.columns)}")
    if frame["application_id"].duplicated().any():
        raise AssertionError(f"application_id duplikat di {path}")
    series = frame.set_index("application_id")["default_probability"]
    missing = set(ids) - set(series.index)
    if missing:
        raise AssertionError(f"{len(missing)} ID test hilang dari {path}")
    return series.loc[ids].to_numpy(dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(ROOT / "data/test_applications.csv", usecols=["application_id"])
    test_ids = test["application_id"].astype(str).to_numpy()
    if len(test_ids) != 30_000 or len(set(test_ids)) != 30_000:
        raise AssertionError("test_applications.csv tidak berisi 30.000 ID unik")

    anchor = pd.read_csv(ANCHOR)
    if list(anchor.columns) != ["application_id", "default_probability"]:
        raise AssertionError(f"kolom anchor tidak sesuai: {list(anchor.columns)}")
    anchor_ids = anchor["application_id"].astype(str).to_numpy()
    anchor_prob = pd.to_numeric(anchor["default_probability"], errors="coerce").to_numpy(
        dtype=float
    )

    checks: dict[str, bool] = {
        "rows_30000": len(anchor) == 30_000,
        "columns_expected": list(anchor.columns)
        == ["application_id", "default_probability"],
        "ids_unique": len(set(anchor_ids)) == 30_000,
        "id_set_matches_test": set(anchor_ids) == set(test_ids),
        "probability_numeric": not np.isnan(anchor_prob).any(),
        "probability_finite": bool(np.isfinite(anchor_prob).all()),
        "probability_in_0_1": bool(
            (anchor_prob >= 0.0).all() and (anchor_prob <= 1.0).all()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise AssertionError(f"validasi anchor gagal: {failed}")

    preds = np.load(PREDS, allow_pickle=True)
    pred_ids = preds["test_ids"].astype(str)
    checks["preds_test_id_order_matches_csv"] = bool(
        np.array_equal(pred_ids, anchor_ids)
    )
    if not checks["preds_test_id_order_matches_csv"]:
        raise AssertionError("urutan test_ids di preds_v66.npz tidak sama dengan CSV")

    # Rebuild the exact v66 chain from the stored test predictions.
    rt = {
        key[5:]: rank01(preds[key])
        for key in preds.files
        if key.startswith("test_") and key != "test_ids"
    }
    trees = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core = (2 * trees + rt["lgbx_full"]) / 3
    tm30 = 0.7 * core + 0.3 * rt["linear"]
    rebuilt = rank01(tm30 + 0.5 * (tm30 - rt["lgb_lean"]))

    max_abs_diff = float(np.max(np.abs(anchor_prob - rebuilt)))
    rho = float(spearmanr(anchor_prob, rebuilt).statistic)
    rank_changes = int(np.sum(~np.isclose(anchor_prob, rebuilt, rtol=0, atol=1e-12)))
    checks["rebuilt_matches_csv"] = max_abs_diff <= 1e-12
    checks["rebuilt_rank_unchanged"] = rank_changes == 0
    checks["spearman_one"] = abs(rho - 1.0) <= 1e-12

    anchor_sha = sha256(ANCHOR)
    checks["anchor_sha256"] = anchor_sha == EXPECTED_ANCHOR_SHA256

    v72_sha = sha256(V72_CONTROL) if V72_CONTROL.exists() else None
    checks["v72_control_byte_identical"] = v72_sha == anchor_sha
    if V72_CONTROL.exists():
        v72_prob = load_aligned(V72_CONTROL, pred_ids)
        checks["v72_control_values_identical"] = bool(
            np.allclose(v72_prob, anchor_prob, rtol=0, atol=0)
        )
    else:
        checks["v72_control_values_identical"] = False

    # Package an exact copy for the final v73 folder. It is not a new candidate.
    shutil.copyfile(ANCHOR, V73_COPY)
    v73_sha = sha256(V73_COPY)
    checks["v73_copy_byte_identical"] = v73_sha == anchor_sha

    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise AssertionError(f"audit final gagal: {failed}")

    summary = {
        "anchor": str(ANCHOR.relative_to(ROOT)),
        "anchor_sha256": anchor_sha,
        "expected_anchor_sha256": EXPECTED_ANCHOR_SHA256,
        "rows": int(len(anchor)),
        "unique_ids": int(len(set(anchor_ids))),
        "min_probability": float(anchor_prob.min()),
        "max_probability": float(anchor_prob.max()),
        "rebuild_max_abs_diff": max_abs_diff,
        "rebuild_spearman": rho,
        "rebuild_rank_changes": rank_changes,
        "v72_control": str(V72_CONTROL.relative_to(ROOT)),
        "v72_control_sha256": v72_sha,
        "v73_copy": str(V73_COPY.relative_to(ROOT)),
        "v73_copy_sha256": v73_sha,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    AUDIT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"anchor        : {ANCHOR.relative_to(ROOT)}")
    print(f"sha256        : {anchor_sha}")
    print(f"rows          : {len(anchor):,}")
    print(f"range         : [{anchor_prob.min():.6f}, {anchor_prob.max():.6f}]")
    print(f"rebuild diff  : {max_abs_diff:.3e}")
    print(f"rebuild rho   : {rho:.12f}")
    print(f"rank changes  : {rank_changes}")
    print(f"v73 copy      : {V73_COPY.relative_to(ROOT)}")
    print(f"audit manifest: {AUDIT_JSON.relative_to(ROOT)}")
    print("status        : PASS")


if __name__ == "__main__":
    main()
