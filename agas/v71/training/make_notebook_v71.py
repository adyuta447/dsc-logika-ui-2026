"""Generate `agas/v71/notebook/notebook_v71.ipynb`.

The notebook is a report over the artifacts in `agas/v71/output`; it does not
retrain.  Execute it with `RUN_TRAINING=False`.
"""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "output"
NB = HERE.parent / "notebook" / "notebook_v71.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


cells = []

cells.append(md(r"""
# DSC0152 — notebook v71

## Tujuan

Mencari informasi baru yang tidak dimiliki keluarga v66–v70, bukan menambah
blok fitur redundan. Tiga hal diuji:

1. **historical panel** — 74.918 pinjaman lama dengan outcome 12 bulan lengkap;
2. **auxiliary rate** — residual rate terhadap fitur risiko;
3. **model diversity** — CatBoost + MLP di atas feature set v70.

Hasil akhirnya: panel dan rate residual **tidak** menambah model sekarang;
MLP adalah lever yang benar-benar baru. Notebook ini mencatat ketiganya,
termasuk hasil negatifnya.
"""))

cells.append(md(r"""
## Setup
"""))

cells.append(code(r"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "agas").exists():
    ROOT = ROOT.parent
OUT = ROOT / "agas/v71/output"
V70 = ROOT / "agas/v70/output"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
print("root", ROOT)
print("v71 output", OUT)
"""))

cells.append(md(r"""
## Public ledger

Anchor tetap `v66_04` = **0.56424**. v70 menguji tiga arah dan tidak ada yang
melewatinya: `v70_07` 0.56419 (praktis seri), `v70_06` 0.56040 (OOF KS
tertinggi tetapi public turun), `v70_01` 0.55801 (OOF AUC tertinggi tetapi
public paling rendah). Ini bukti bahwa selisih OOF di bawah ~0.001 tidak
memprediksi public.
"""))

cells.append(code(r"""
ledger = pd.DataFrame([
    ("v66_04", "ext0.5|tm30|lgb_lean", 0.56424, "anchor"),
    ("v69_10", "ext0.5|tm30|extra", 0.55989, "pivot extra gagal"),
    ("v70_06", "plateau|tm30w7|v70_extra", 0.56040, "OOF KS tinggi, public turun"),
    ("v70_07", "ext0.5|tm30v70|lgb_lean", 0.56419, "praktis seri"),
    ("v70_01", "leg|v70_extra", 0.55801, "OOF AUC tinggi, public turun"),
], columns=["candidate", "chain", "public", "note"])
ledger
"""))

cells.append(md(r"""
## 1. Historical panel

### Konstruksi label

`previous_applications` yang berstatus `approved` dan punya `loan_id` digabung
dengan repayment history. Label adalah `max_days_past_due >= 90` pada 12
angsuran pertama. Baris hanya dipakai jika window 12 bulan sudah selesai
sebelum `decision_datetime` aplikasi sekarang.

Label ini adalah **aproksimasi** dari definisi resmi “90+ DPD dalam 12 bulan”:
data repayment tidak memberi timestamp harian kapan DPD 90 tercapai, jadi
angsuran pertama sampai ke-12 dipakai sebagai proksi. Batas ini dicatat, bukan
disembunyikan.

### Fitur

Semua fitur dihitung pada `decision_datetime` pinjaman historis:

- terms pinjaman historis dan decision lag;
- previous applications sebelum cut: approval/rejection, amount, rate, recency,
  gap, active count;
- repayment sebelum cut: late30/late90, DPD, unpaid, delay, tren, outcome per
  loan;
- bureau report milik application historis: score, provider disagreement,
  utilization, inquiries, delinquency;
- account behavior sebelum cut: inflow/outflow, net, saldo, low-balance days,
  tren 1/3/6/12 bulan;
- e-commerce sebelum cut: spend/order 30/90/180/365 hari, basket, kategori,
  payment method, cancel, tren. Agregasi harian per customer dilakukan lebih
  dulu agar memori tetap wajar.

Profil customer adalah snapshot saat ini. Umur dan tenure di-backdate; income
dipakai dengan catatan bahwa ini snapshot, bukan nilai pada saat pinjaman
historis. Tidak ada target aplikasi sekarang yang dipakai.
"""))

cells.append(code(r"""
build = json.loads((OUT / "summary_panel_build_v71.json").read_text())
panel = json.loads((OUT / "summary_panel_v71.json").read_text())
print(json.dumps(build, indent=2))
print("\npanel model:")
for k in ["hist_rows", "current_rows", "features", "hist_positive_rate",
          "panel_hist_auc", "panel_hist_ks", "panel_current_auc",
          "panel_current_ks", "minutes"]:
    print(f"  {k}: {panel[k]}")
print("\nper fold")
display(pd.DataFrame(panel["per_fold"]).T)
"""))

cells.append(md(r"""
### Hasil panel dan keputusan

Panel model memakai 5-fold **GroupKFold by customer**. Historical OOF AUC
0.7083 dan KS 0.2962; ketika skor panel dipakai untuk memprediksi target
aplikasi sekarang, AUC-nya 0.7562. Itu bukan model utama, tetapi cukup untuk
diuji sebagai fitur.

Screening current model memakai 5 fold × 3 fold seed, config `extra`, factor
aktif. Pembanding adalah cache v70 yang sama.
"""))

cells.append(code(r"""
rows = []
for name, path in [
    ("v70 baseline (no panel)", V70 / "summary_s4_all.json"),
    ("v71 panel score only", OUT / "summary_cv_v71_panel.json"),
    ("v71 panel score + 68 pn_ features + rf_", OUT / "summary_cv_v71_all.json"),
]:
    if path.exists():
        j = json.loads(path.read_text())
        rows.append({"run": name, "n_lean": j["n_lean"],
                     "mean_auc": j["mean_auc"], "sd_auc": j["sd_auc"],
                     "mean_ks": j["mean_ks"]})
pd.DataFrame(rows)
"""))

cells.append(md(r"""
**Keputusan:** panel score hanya 0.847202 vs baseline 0.847219 (selisih
−0.000017, jauh di dalam noise). Panel + 68 fitur + rate residual 0.847199.
Panel **tidak dipakai** sebagai fitur current model. Ini negative result yang
jelas: label historisnya terlalu lemah/noisy untuk menambah model yang sudah
mempunyai 684 fitur current.

## 2. Auxiliary rate

Model multiclass memprediksi tier rate (16/20/24/28/32%) dari fitur selain
rate dan installment. Tujuannya mendapat residual rate terhadap grade risiko.
Hasilnya: rate bisa diprediksi hampir sempurna dari fitur lain (AUC rate>=24%
= 1.0), residual hampir nol (median absolut ~2.7e-6). Ini bukan sinyal baru;
ini berarti rate adalah fungsi deterministik dari profil yang sudah terukur.
Fitur `rf_*` tidak dipakai di model final.
"""))

cells.append(code(r"""
bf = json.loads((OUT / "summary_build_features_v71.json").read_text())
print(json.dumps(bf["rate_model"], indent=2))
# residual distribution from the cache, if present
f = pd.read_pickle(OUT / "features_v71.pkl")
tr = f[f.target.notna()]
tr[["rf_rate_residual", "rf_rate_abs_residual", "rf_rate_entropy"]].describe(
    percentiles=[.01, .5, .99])
"""))

cells.append(md(r"""
## 3. Model diversity — lever yang berhasil

Feature set v70 adalah 684 kolom lean; factor features tetap fold-safe.
LightGBM sudah jenuh. Yang baru adalah dua geometri model:

- **CatBoost**: split/regularisasi berbeda, 15 fold × 3 fold seed;
- **MLP**: 771 kolom numerik, median imputation + clipping + standardisasi
  di dalam fold, arsitektur 256→64, dropout, early stopping.

MLP tidak memakai panel/rate. CatBoost memakai lean yang sama dengan LGBM.
"""))

cells.append(code(r"""
rows = []
for name, path in [
    ("MLP 256/64 (3 seeds × 3 fold seeds)", OUT / "summary_mlp_v71.json"),
    ("MLP wide 512/128 (15 fold, 3 fold seed)", OUT / "summary_mlp_wide_v71.json"),
    ("MLP wide 512/128 (screen 5-fold)", OUT / "summary_mlp_wide_screen.json"),
    ("CatBoost (screen 5-fold)", OUT / "summary_cat_v71_screen.json"),
    ("CatBoost (15 fold, 3 fold seed)", OUT / "summary_cat_v71.json"),
    ("XGBoost (screen 5-fold)", OUT / "summary_xgb_v71_screen.json"),
]:
    if path.exists():
        j = json.loads(path.read_text())
        rows.append({"leg": name, "mean_auc": j["mean_auc"],
                     "sd_auc": j["sd_auc"], "mean_ks": j["mean_ks"],
                     "minutes": j["minutes"]})
pd.DataFrame(rows)
"""))

cells.append(md(r"""
MLP final (15 fold × 3 fold seed × 3 seed) mencapai AUC **0.848433** dan KS
**0.529473**, sedikit di atas `v70_extra` (0.848369 / 0.528503). Tetapi nilai
sebenarnya ada di korelasi: rank MLP vs v70 hanya ~0.973, jauh lebih rendah
daripada korelasi antar-LGBM. Blend rank 60% MLP + 40% v70 mencapai
**AUC 0.850082 / KS 0.531973** pada OOF.

CatBoost final dan blend final dibaca dari artefak di bawah.
"""))

cells.append(code(r"""
# candidate table is the authoritative final blend/selection record
cand = json.loads((OUT / "summary_v71_candidates.json").read_text())
print("core weights:", {k: round(float(v), 3) for k, v in cand["core_weights"].items()})
print("core auc/ks:", round(cand["core_auc"], 6), round(cand["core_ks"], 6))
display(pd.DataFrame(cand["candidates"]))
"""))

cells.append(md(r"""
### Public-proxy bootstrap

Untuk setiap kandidat, 600 subset stratified berukuran 30.000 diambil dari OOF,
lalu dihitung `KS(candidate) - KS(anchor)` pada subset yang sama. Ini bukan
prediksi public; ini mengukur apakah kandidat menang karena perbedaan rank yang
konsisten, bukan karena satu argmax KS.

Yang dicari adalah kandidat dengan `P(KS lebih baik) > 0.55` dan AUC yang masih
masuk lantai 0.0020 di bawah kandidat terbaik. Kandidat yang rank-nya >0.9995
sama dengan kandidat yang sudah dipilih dideduplikasi.
"""))

cells.append(code(r"""
rows = []
for r in cand["candidates"]:
    rows.append({
        "slot": r["slot"], "name": r["name"], "auc": r["auc"], "ks": r["ks"],
        "dKS": r["proxy"]["mean_dks"], "P(ks)": r["proxy"]["p_ks_better"],
        "dAUC": r["proxy"]["mean_dauc"], "rho_v66": r["spearman_v66_04"],
        "file": r["file"], "valid": r["valid"],
    })
pd.DataFrame(rows)
"""))

cells.append(md(r"""
## Validasi CSV

Semua file harus 30.000 baris, `application_id` unik dan cocok dengan
`data/test_applications.csv`, probabilitas di [0, 1], dan tanpa NaN.
"""))

cells.append(code(r"""
test_ids = pd.read_csv(ROOT / "data/test_applications.csv").application_id
bad = []
for p in sorted(OUT.glob("submission_v71_*.csv")):
    d = pd.read_csv(p)
    ok = (len(d) == 30000 and d.application_id.is_unique
          and set(d.application_id) == set(test_ids)
          and d.default_probability.notna().all()
          and d.default_probability.between(0, 1).all())
    if not ok:
        bad.append(p.name)
    print(f"{p.name:60s} rows={len(d):5d} ok={ok}")
print("bad:", bad)
"""))

cells.append(md(r"""
## Reproduksi

Jalankan dari root repo:

```bash
python3 agas/v71/training/build_panel_v71.py
python3 agas/v71/training/train_panel_v71.py
python3 agas/v71/training/build_features_v71.py
python3 agas/v71/training/train_mlp_v71.py --drops pn_,rf_ --folds 15 \
  --fold-seeds 42,7,13 --seeds 0,1,2 --out-prefix mlp_v71
python3 agas/v71/training/train_cat_v71.py --drops pn_,rf_ --folds 15 \
  --fold-seeds 42,7,13 --seeds 0 --use-factor --out-prefix cat_v71
python3 agas/v71/training/train_mlp_v71.py --drops pn_,rf_ --folds 15 \
  --fold-seeds 42,7,13 --seeds 0 --hidden 512 --hidden2 128 \
  --dropout 0.35 --dropout2 0.20 --lr 0.001 --wd 0.0003 \
  --out-prefix mlp_wide_v71
python3 agas/v71/training/make_submission_v71.py
```

Notebook ini tidak menjalankan training. Set `RUN_TRAINING=True` hanya jika
ingin mencetak ulang perintah di atas.
"""))

cells.append(code(r"""
RUN_TRAINING = False
if RUN_TRAINING:
    print("See the markdown cell above for the full command list.")
else:
    print("report-only mode; all numbers read from agas/v71/output")
"""))

cells.append(md(r"""
## Kesimpulan

- **Panel historical** adalah negative result yang mahal tapi jelas: label
  historis 16.5% positif, model panel AUC 0.708, dan tidak menambah model
  current.
- **Rate residual** juga negative: rate deterministik dari fitur yang sudah ada.
- **MLP** adalah satu-satunya lever baru yang berhasil. OOF blend naik dari
  0.84837 (LGBM-only) ke ~0.8501, dengan KS ~0.5320.
- Public 0.577 tetap tidak dijamin. Kenaikan OOF ~0.0017 setara ~0.3 sd public;
  kandidat v71 lebih kuat dan lebih beragam, tetapi 0.577 masih membutuhkan
  favorable public variance atau structural break berikutnya.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB))
print(f"saved {NB} with {len(cells)} cells")
