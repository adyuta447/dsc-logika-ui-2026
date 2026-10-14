# Progress Eksperimen `agas/`

## 2026-10-14

### 1. Konteks dan target

- Skor tim saat ini: sekitar `0.561`
- Target: tembus `0.57` public leaderboard
- Constraint: tidak mengubah file lama, semua file baru hanya di `agas/`

### 2. Audit awal

- Baca aturan di `agas/limit.md`
- Audit baseline dari `PROGRESS.md` dan `make_nb.py`
- Konfirmasi library tersedia: `numpy`, `pandas`, `scipy`, `sklearn`, `lightgbm`, `xgboost`, `catboost`

### 3. Implementasi

Membuat script baru:

- `agas/train_submission_v57.py`

Isi utama:

1. Load fitur dari `features.py` (tanpa ubah source).
2. Build factor features per fold dengan `factor_inputs` + `factor_features`.
3. Training ensemble model multi-family (LGB/CAT/XGB/linear spline).
4. Build beberapa blend + satu blend agresif (`aggressive_145`).
5. Simpan beberapa kandidat submission sekaligus.

### 4. Eksekusi training

Command:

```bash
python3 agas/train_submission_v57.py --folds 5 --fold-seed 42 --lr 0.01 --lgb-seeds 0,1,2 --out-prefix v57
```

Durasi:

- `37.33` menit

### 5. Hasil CV (OOF)

Sumber: `agas/output/summary_v57.json`

- `top_mix`: AUC `0.846782`, KS `0.526606`
- `linear30`: AUC `0.846740`, KS `0.525565`
- `trees_only`: AUC `0.846534`, KS `0.525488`
- `extra31_mix`: AUC `0.846586`, KS `0.525972`
- `meta_logit`: AUC `0.846772`, KS `0.526435`
- `aggressive_145`: AUC `0.846725`, KS `0.526988` (**best KS run ini**)

### 6. Artefak yang dihasilkan

Folder output:

- `agas/output/submission_v57_best.csv`
- `agas/output/submission_v57_aggressive_145.csv`
- `agas/output/submission_v57_top_mix.csv`
- `agas/output/submission_v57_meta_logit.csv`
- `agas/output/submission_v57_linear30.csv`
- `agas/output/submission_v57_trees_only.csv`
- `agas/output/submission_v57_extra31_mix.csv`

Dokumentasi:

- `agas/output/train_log_v57.txt`
- `agas/output/summary_v57.json`

### 7. Validasi submission

Semua file `submission_v57_*.csv` lolos:

- baris `30000`
- `application_id` unik
- set `application_id` sesuai test
- `default_probability` valid di `[0,1]`

### 8. Keputusan run ini

- Kandidat utama untuk submit: `submission_v57_best.csv`
- Kandidat backup: `submission_v57_top_mix.csv`, `submission_v57_meta_logit.csv`
- Tidak ada perubahan file existing di luar `agas/`

## 2026-10-14 (lanjutan, sore)

### 1. Hasil submit v57 dan evaluasi ulang

- `submission_v57_best.csv` (varian `aggressive_145`) disubmit, **hasil public hanya 0.557**.
- Skor tim terbaik tetap `0.5617` (bukan 0.617, koreksi dari user), dari file lama tim `agas/qc_00_ext1.5_v4_fac_full_only.csv`.
- **Diagnosis kenapa `aggressive_145` gagal**: blend itu mengekstrapolasi (1.45x - 0.45x) antara dua varian blend hasil racikan saya sendiri (`top_mix` vs `meta_logit`), dipilih cuma berdasar OOF KS session ini. Arah ekstrapolasi itu **tidak pernah divalidasi ke public leaderboard**, jadi noise public (0.0015–0.003 per `PROGRESS.md`) bisa dominan dan merusak skor.
- File lama tim yang skornya `0.5617` dibangun dengan arah ekstrapolasi yang **sudah terbukti di public**: `ext1.5|v4-fac|full_only`, yaitu `rank(v4_full_only + 1.5*(v4_full_only - fac_full_only))`. Ditemukan sumbernya di `experiments/harvest_kf.py` fungsi `ext()`.
- `v4` = trees (LGB+CAT+XGB) dilatih pakai fitur lengkap **termasuk** 4 kelompok fitur musiman/rhythm yang ditambahkan belakangan: `dec_doy` (musim), `ab_inc_share` (porsi gaji), `ab_open_avg` (saldo awal), `pa_gap_mean`/`ec_gap_mean`/`ec_gap_last3` (jarak antar pengajuan/belanja).
- `fac` = trees yang sama tapi **tanpa** 6 kolom itu (fitur lama, pra-v4), factor model tetap ada di kedua sisi.

### 2. Keputusan strategi baru

Daripada bikin blend baru lagi yang belum tervalidasi, **replikasi resep yang sudah terbukti public (0.5617)** dengan kualitas model dasar yang lebih kuat:

- 3 fold seed (`42,7,13`) seperti resep asli.
- 3 LightGBM seed per varian (bukan 1) + CatBoost + XGBoost, untuk tiap sisi (`v4` dan `fac`).
- Rasio ekstrapolasi `1.5` (terbukti), plus `1.0` dan `2.0` sebagai pembanding/cadangan karena noise public leaderboard tinggi.

Script baru: `agas/train_submission_v58_ext.py`. Tidak mengubah `features.py`; kolom v4-only dihapus manual dari daftar kolom di script sendiri (bukan di source).

### 3. Eksekusi

Command:

```bash
python3 agas/train_submission_v58_ext.py --fold-seeds 42,7,13 --folds 5 --lr 0.01 --lgb-seeds 0,1,2 --ext-ratios 1.0,1.5,2.0 --out-prefix v58ext
```

Durasi: `84.39` menit (3 fold seed x 2 varian x 5 model per varian).

### 4. Hasil CV (OOF), sumber `agas/output/summary_v58ext.json`

| Varian | AUC | KS |
|---|---|---|
| `v4_full_only` (bag 3 fold seed) | 0.847073 | 0.526265 |
| `fac_full_only` (bag 3 fold seed) | 0.846160 | 0.525122 |
| `ext1.0_v4_fac_full_only` | 0.846846 | 0.527486 |
| **`ext1.5_v4_fac_full_only`** | 0.846298 | **0.527511** (terbaik) |
| `ext2.0_v4_fac_full_only` | 0.845463 | 0.526260 |

Catatan penting:

- Arah `v4 - fac` **konsisten positif di ketiga fold seed** (v4 selalu lebih baik dari fac di semua 3 seed, selisih KS 0.0017–0.0033), mengkonfirmasi arah ekstrapolasi ini bukan kebetulan satu fold.
- Rasio `1.5` memang OOF KS terbaik di antara 3 pilihan, cocok dengan temuan tim sebelumnya (dipakai di file `qc_00` yang skornya 0.5617).
- `ext1.5` run ini AUC OOF `0.846298` KS `0.527511`, vs resep lama yang (menurut `PROGRESS.md`) berbasis `run_v4_s42/s7/s13` + `run_fac/s7/s13/s21` (4 run untuk fac, bukan 3). KS run ini sedikit lebih tinggi dari estimasi `top_mix` run v57 sebelumnya (`0.526988`).

### 5. Validasi kesamaan arah dengan file lama

```python
# spearman antara submission_v58ext_ext1.5 vs qc_00 (file lama, skor 0.5617)
spearman = 0.999392
```

Korelasi rank sangat tinggi (0.9994) terhadap `agas/qc_00_ext1.5_v4_fac_full_only.csv` → ini benar replikasi arah yang sama, bukan blend baru yang belum teruji, dengan model dasar yang dilatih ulang (bukan file yang disalin).

### 6. Artefak baru

- `agas/train_submission_v58_ext.py` (script)
- `agas/output/submission_v58ext_v4_full_only.csv`
- `agas/output/submission_v58ext_fac_full_only.csv`
- `agas/output/submission_v58ext_ext1_v4_fac_full_only.csv`
- `agas/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` (**kandidat utama baru**)
- `agas/output/submission_v58ext_ext2_v4_fac_full_only.csv`
- `agas/output/summary_v58ext.json`, `agas/output/train_log_v58ext.txt`

Semua file lolos validasi: 30000 baris, `application_id` unik dan cocok test, `default_probability` di `[0,1]`.

### 7. Rekomendasi submit berikutnya

1. `submission_v58ext_ext1.5_v4_fac_full_only.csv` — replikasi resep terbukti 0.5617 dengan base model lebih kuat, OOF KS tertinggi sejauh ini.
2. `submission_v58ext_ext1_v4_fac_full_only.csv` — cadangan, rasio lebih konservatif kalau ekstrapolasi 1.5 ternyata overfit ke noise public pada split berikutnya.
3. Hindari submit ulang varian seperti `aggressive_145` (arah blend belum tervalidasi di public) kecuali ada bukti public baru yang mendukung arah itu.

### 8. Tidak dilakukan (di luar scope/waktu run ini)

- Tidak mencoba 10-fold (resep asli notebook pakai 10 fold, run ini pakai 5 fold untuk kecepatan). Bisa jadi langkah lanjutan kalau waktu submit masih ada sebelum deadline 30 Okt.
- Tidak mengubah file apa pun di luar `agas/`, tidak push ke GitHub.

