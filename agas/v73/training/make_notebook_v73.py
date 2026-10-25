"""Generate the final v73 notebook.

The notebook is a reproducibility/audit notebook for the frozen v66_04 anchor.
It does not train a model: ``RUN_TRAINING`` is always False in the executed
copy. The original v66 training command is kept visible for provenance.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "agas/v73/notebook/notebook_v73_final.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


cells = [
    markdown(
        """
        # DSC0152 — Final notebook v73

        **Status:** final freeze, dijalankan penuh dengan `RUN_TRAINING=False`.

        **Anchor final:** `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv`
        **Public anchor:** **0.56424**
        **Tanggal keputusan:** 2026-10-25

        Notebook ini tidak melatih model baru. Tujuannya adalah membuktikan bahwa
        submission final yang dipilih adalah vektor `v66_04` yang sudah terbukti di
        public, lalu memvalidasi artefaknya secara reproducible.
        """
    ),
    markdown(
        """
        ## Keputusan final

        `submission_v72_02_anchor_seq05.csv` mendapat public **0.55968**, yaitu
        **−0.00456** dari anchor `v66_04` (**0.56424**). OOF seq05 sedikit lebih
        tinggi (AUC 0.846968, KS 0.528844, Spearman 0.999501 terhadap anchor),
        tetapi kenaikan itu tidak transfer ke public.

        Semua kandidat pasca-anchor yang diuji gagal melewati 0.56424:
        `core71` 0.56111, `v70_06` 0.56040, `v69_10` 0.55989, `v70_01` 0.55801.
        Karena itu:

        - **final = `v66_04`**;
        - **v72 ditutup sebagai negative result**;
        - tidak ada probe model baru, tidak ada submit baru di notebook ini;
        - skor `0.57768` adalah skor peserta lain, bukan kandidat kita.

        ## Audit cutoff dan leakage

        - Notebook ini tidak membuat fitur baru dan tidak memakai data setelah
          `decision_datetime`.
        - Anchor memakai artefak v66 yang sudah memakai cutoff event-time.
        - Sequence v72 **tidak** dipakai pada kandidat final.
        - Tidak ada data eksternal, AutoML, atau AutoFE.
        """
    ),
    markdown(
        """
        ## Model yang dipakai: v66

        v66 mengembalikan arsitektur v61 dan hanya menambah jumlah training.
        Konfigurasi final: 10 fold `StratifiedKFold`, 6 fold seed
        (`42,7,13,21,5,101`), dan 2 LightGBM seed. Leg yang dilatih adalah
        `lgb_full`, `lgb_lean`, `fac_full`, `cat_full`, `xgb_full`, `lgbx_full`,
        dan `linear`; factor features dihitung fold-safe. Tidak ada feature
        selection, karena seleksi gain per fold adalah salah satu penyebab
        kemunduran v65.

        Blend yang menghasilkan `v66_04`:

        ```text
        trees4 = mean(rank(lgb_full, lgb_lean, cat_full, xgb_full))
        core   = (2 * trees4 + lgbx_full) / 3
        tm30   = 0.7 * core + 0.3 * linear
        anchor = rank01(tm30 + 0.5 * (tm30 - lgb_lean))
        ```

        `v66_04` adalah ekstrapolasi `ext0.5` dari `tm30` menjauhi `lgb_lean`,
        arah yang sama dengan v61. Pada OOF, resep ini menghasilkan
        AUC 0.846840 dan KS 0.528713; di public ia menghasilkan **0.56424**.
        Kandidat v68–v72 yang menambah fold, fitur, MLP, atau sequence tidak
        melewati angka itu.
        """
    ),
    code(
        """
        import hashlib
        import json
        import os
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from scipy.stats import ks_2samp, rankdata, spearmanr
        from sklearn.metrics import roc_auc_score

        RUN_TRAINING = False
        FOLDS = 10
        FOLD_SEEDS = "42,7,13,21,5,101"
        LGB_SEEDS = "0,1"


        def find_root(start: Path) -> Path:
            start = start.resolve()
            for candidate in [start, *start.parents]:
                if (candidate / "data").is_dir() and (candidate / "agas").is_dir():
                    return candidate
            raise RuntimeError("repo root tidak ditemukan")


        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def rank01(values) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            return rankdata(values) / len(values)


        def load_aligned(path: Path, ids: np.ndarray) -> np.ndarray:
            frame = pd.read_csv(path)
            series = frame.set_index("application_id")["default_probability"]
            return series.loc[ids].to_numpy(dtype=float)


        ROOT = find_root(Path.cwd())
        AGAS = ROOT / "agas"
        V66 = AGAS / "v66"
        V72 = AGAS / "v72"
        V73 = AGAS / "v73"
        ANCHOR = V66 / "output/submission_v66_04_ext0.5_tm30_lgb_lean.csv"
        PREDS = V66 / "output/preds_v66.npz"
        V72_CONTROL = V72 / "output/submission_v72_01_anchor.csv"
        V73_COPY = V73 / "output/submission_v73_01_anchor.csv"

        os.chdir(ROOT)
        print("repo root  :", ROOT)
        print("RUN_TRAINING:", RUN_TRAINING)
        print("anchor     :", ANCHOR.relative_to(ROOT))
        """
    ),
    code(
        """
        test_ids = pd.read_csv(
            ROOT / "data/test_applications.csv", usecols=["application_id"]
        ).application_id.astype(str).to_numpy()

        anchor_frame = pd.read_csv(ANCHOR)
        anchor_ids = anchor_frame.application_id.astype(str).to_numpy()
        anchor_vec = anchor_frame.default_probability.to_numpy(dtype=float)

        checks = {
            "rows_30000": len(anchor_frame) == 30_000,
            "columns_expected": list(anchor_frame.columns)
            == ["application_id", "default_probability"],
            "ids_unique": len(set(anchor_ids)) == 30_000,
            "id_set_matches_test": set(anchor_ids) == set(test_ids),
            "probability_finite": bool(np.isfinite(anchor_vec).all()),
            "probability_in_0_1": bool(
                (anchor_vec >= 0).all() and (anchor_vec <= 1).all()
            ),
        }
        print(pd.Series(checks, name="ok").to_string())
        assert all(checks.values())
        print(f"rows={len(anchor_frame):,}  min={anchor_vec.min():.6f}  max={anchor_vec.max():.6f}")
        print("sha256=", sha256(ANCHOR))
        """
    ),
    code(
        """
        z = np.load(PREDS, allow_pickle=True)
        assert np.array_equal(z["test_ids"].astype(str), anchor_ids)

        rt = {
            key[5:]: rank01(z[key])
            for key in z.files
            if key.startswith("test_") and key != "test_ids"
        }
        trees = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
        core = (2 * trees + rt["lgbx_full"]) / 3
        tm30 = 0.7 * core + 0.3 * rt["linear"]
        rebuilt = rank01(tm30 + 0.5 * (tm30 - rt["lgb_lean"]))

        max_abs_diff = float(np.max(np.abs(anchor_vec - rebuilt)))
        rho = float(spearmanr(anchor_vec, rebuilt).statistic)
        rank_changes = int(np.sum(~np.isclose(anchor_vec, rebuilt, rtol=0, atol=1e-12)))
        print(f"rebuild max abs diff : {max_abs_diff:.3e}")
        print(f"rebuild Spearman     : {rho:.12f}")
        print(f"rank changes         : {rank_changes}")
        assert max_abs_diff <= 1e-12
        assert rank_changes == 0

        # Reproduce the OOF metrics of the same recipe for provenance.
        y = (
            pd.read_csv(
                ROOT / "data/train_applications.csv",
                usecols=["application_id", "target"],
            )
            .set_index("application_id")
            .loc[z["train_ids"], "target"]
            .to_numpy(dtype=int)
        )
        ro = {
            key[4:]: rank01(z[key])
            for key in z.files
            if key.startswith("oof_")
        }
        trees_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
        core_o = (2 * trees_o + ro["lgbx_full"]) / 3
        tm30_o = 0.7 * core_o + 0.3 * ro["linear"]
        anchor_oof = rank01(tm30_o + 0.5 * (tm30_o - ro["lgb_lean"]))
        oof_auc = float(roc_auc_score(y, anchor_oof))
        oof_ks = float(ks_2samp(anchor_oof[y == 1], anchor_oof[y == 0]).statistic)
        print(f"OOF AUC              : {oof_auc:.6f}")
        print(f"OOF KS               : {oof_ks:.6f}")
        """
    ),
    code(
        """
        train_cmd = (
            f"python3 {V66 / 'training/train_submission_v66.py'} "
            f"--folds {FOLDS} --fold-seeds {FOLD_SEEDS} "
            f"--lgb-seeds {LGB_SEEDS} --out-prefix v66"
        )
        print(train_cmd)
        if RUN_TRAINING:
            raise RuntimeError("Final notebook harus dijalankan dengan RUN_TRAINING=False")
        print("RUN_TRAINING=False: training dilewati; memakai preds_v66.npz")

        summary = json.loads((V66 / "output/summary_v66.json").read_text())
        print("folds       :", summary["folds"])
        print("fold seeds  :", summary["fold_seeds"])
        print("lgb seeds   :", summary["lgb_seeds"])
        print("columns     :", summary["n_full"], summary["n_lean"], summary["n_fac"])
        print("minutes     :", round(summary["minutes"], 1))
        """
    ),
    code(
        """
        ledger = pd.DataFrame(
            [
                ("v66_04_ext0.5_tm30_lgb_lean", 0.56424, "FINAL / anchor"),
                ("v70_07_ext0.5_tm30v70_lgb_lean", 0.56419, "near-anchor"),
                ("v68_04_ext0.5_tm30_lgb_lean", 0.56411, "near-anchor"),
                ("v71_01_core71", 0.56111, "rejected"),
                ("v70_06_plateau_tm30w7_v70_extra_lgb_lean", 0.56040, "rejected"),
                ("v69_10_ext0.5_tm30_extra", 0.55989, "rejected"),
                ("v72_02_anchor_seq05", 0.55968, "rejected / seq gain tidak transfer"),
                ("v70_01_leg_v70_extra", 0.55801, "rejected"),
            ],
            columns=["candidate", "public", "decision"],
        )
        print(ledger.to_string(index=False))
        """
    ),
    code(
        """
        controls = {
            "v72_01_anchor": V72_CONTROL,
            "v73_01_anchor": V73_COPY,
        }
        control_checks = {}
        for name, path in controls.items():
            vec = load_aligned(path, anchor_ids)
            control_checks[name] = {
                "sha256": sha256(path),
                "same_hash": sha256(path) == sha256(ANCHOR),
                "same_values": bool(np.allclose(vec, anchor_vec, rtol=0, atol=0)),
            }
            print(name, control_checks[name])
        assert all(v["same_hash"] and v["same_values"] for v in control_checks.values())
        """
    ),
    code(
        """
        final_ok = (
            all(checks.values())
            and max_abs_diff <= 1e-12
            and rank_changes == 0
            and abs(rho - 1.0) <= 1e-12
            and all(v["same_hash"] and v["same_values"] for v in control_checks.values())
        )
        assert final_ok
        print("FINAL SUBMISSION :", ANCHOR.relative_to(ROOT))
        print("PUBLIC           : 0.56424")
        print("STATUS           : PASS")
        print("No training, no Kaggle submission, and no GitHub push from this notebook.")
        """
    ),
    markdown(
        """
        ## Batas dan keputusan operasional

        - Notebook ini **tidak** mengirim submission ke Kaggle.
        - Notebook ini **tidak** melakukan push ke GitHub.
        - Kandidat final hanya `v66_04`; `seq05`, `core71`, dan seluruh blend
          sequence/MLP tidak boleh menggantikan anchor.
        - Deadline notebook: **30 Oktober 2026, 23:59 WIB**. Target upload
          internal: 29 Oktober 2026.
        - Nama file final mengikuti aturan committee:
          `ParticipantNumber_TeamName_Notebook.ipynb`.
        """
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells")
