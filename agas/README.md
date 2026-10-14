# README - Eksperimen Submission `agas/`

Dokumen ini menjelaskan langkah kerja untuk menghasilkan kandidat submission baru tanpa mengubah file lama di repo.

## 1) Batasan yang dipatuhi

- Tidak push apa pun ke GitHub.
- Tidak mengubah file existing di luar `agas/`.
- Semua artefak baru diletakkan di `agas/`.
- Hanya memakai data kompetisi lokal di folder `data/`.

## 2) Metode yang dipakai

Script utama: `agas/train_submission_v57.py`

Ringkasan metode:

1. Memakai feature engineering existing dari `features.py` (termasuk factor features) tanpa edit file sumber.
2. Training model ensemble 5-fold:
   - LightGBM (`full`, `lean`, `top`) dengan seed `0,1,2`
   - CatBoost (`full`, `lean`, `top`)
   - XGBoost (`full`, `lean`, `top`)
   - LightGBM extra config (`lgbx_full`, `lgbx_lean`)
   - Spline Logistic
3. Membuat beberapa blend:
   - `top_mix`
   - `linear30`
   - `trees_only`
   - `extra31_mix`
   - `meta_logit`
   - `aggressive_145` (ekstrapolasi rank blend)
4. Memilih varian terbaik berdasarkan KS out-of-fold.

## 3) Cara menjalankan

Jalankan dari root repo:

```bash
python3 agas/train_submission_v57.py --folds 5 --fold-seed 42 --lr 0.01 --lgb-seeds 0,1,2 --out-prefix v57
```

## 4) Struktur output

Semua hasil ada di `agas/output/`:

- `features_cache.pkl` - cache fitur gabungan train+test
- `train_log_v57.txt` - log training detail
- `summary_v57.json` - ringkasan metrik semua blend
- `submission_v57_*.csv` - kandidat submission
- `submission_v57_best.csv` - kandidat terbaik otomatis

## 5) Validasi file submission

Semua file `submission_v57_*.csv` sudah lolos cek:

- jumlah baris = 30000
- `application_id` unik
- set `application_id` sama persis dengan `data/test_applications.csv`
- `default_probability` dalam rentang `[0,1]`

## 6) Saran urutan submit ke Kaggle (run v57, SUDAH DIEVALUASI — lihat bagian 7)

1. ~~`agas/output/submission_v57_best.csv` (varian `aggressive_145`)~~ — disubmit, hasil public **0.557**, lebih buruk dari baseline tim.
2. `agas/output/submission_v57_top_mix.csv`
3. `agas/output/submission_v57_meta_logit.csv`

Catatan: leaderboard public kompetisi ini memiliki noise tinggi, jadi varian dengan CV sedikit lebih rendah kadang bisa lebih tinggi di public.

---

## 7) Run kedua: replikasi resep terbukti (`train_submission_v58_ext.py`)

`aggressive_145` (bagian 6) gagal di public karena arahnya (ekstrapolasi antara dua blend buatan sendiri) tidak pernah divalidasi di leaderboard — hanya dipilih dari OOF KS session itu saja. Pelajaran: **ekstrapolasi rank hanya aman kalau arahnya sudah terbukti menaikkan skor public**, bukan sekadar menaikkan KS di CV sendiri.

Skor tim terbaik (`0.5617`, file tim `agas/qc_00_ext1.5_v4_fac_full_only.csv`) memakai arah yang sudah teruji:

```
ext1.5|v4-fac|full_only = rank(v4_full_only + 1.5 * (v4_full_only - fac_full_only))
```

- `v4_full_only`: trees (LGB+CAT+XGB) pada fitur lengkap, **termasuk** 6 kolom v4 (`dec_doy`, `ab_inc_share`, `ab_open_avg`, `pa_gap_mean`, `ec_gap_mean`, `ec_gap_last3`).
- `fac_full_only`: trees yang sama, **tanpa** 6 kolom itu (fitur pra-v4), factor model tetap dipakai di kedua sisi.
- Ditemukan di `experiments/harvest_kf.py` (fungsi `ext`), bukan dibuat ulang dari nol.

### Cara menjalankan

```bash
python3 agas/train_submission_v58_ext.py --fold-seeds 42,7,13 --folds 5 --lr 0.01 --lgb-seeds 0,1,2 --ext-ratios 1.0,1.5,2.0 --out-prefix v58ext
```

Perbedaan vs resep asli: 3 LightGBM seed per varian (bukan 1), dijalankan di 3 fold seed seperti resep asli, untuk ensemble dasar yang sedikit lebih stabil. Tidak mengubah `features.py` — kolom v4-only dibuang manual lewat daftar kolom di script ini saja.

### Hasil OOF (`agas/output/summary_v58ext.json`)

| Varian | AUC | KS |
|---|---|---|
| `v4_full_only` | 0.847073 | 0.526265 |
| `fac_full_only` | 0.846160 | 0.525122 |
| `ext1.0` | 0.846846 | 0.527486 |
| **`ext1.5`** | 0.846298 | **0.527511** |
| `ext2.0` | 0.845463 | 0.526260 |

Arah `v4 - fac` positif konsisten di semua 3 fold seed (bukan kebetulan satu split). Rasio `1.5` tetap rasio terbaik di OOF, sejalan dengan temuan tim sebelumnya.

Validasi arah: korelasi Spearman antara `submission_v58ext_ext1.5_v4_fac_full_only.csv` dan file lama tim `qc_00_ext1.5_v4_fac_full_only.csv` = **0.9994** — konfirmasi ini replikasi arah yang sama dengan model dasar dilatih ulang, bukan blend spekulatif baru.

### File output

- `agas/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` — **kandidat utama berikutnya**
- `agas/output/submission_v58ext_ext1_v4_fac_full_only.csv` — cadangan konservatif
- `agas/output/submission_v58ext_ext2_v4_fac_full_only.csv` — cadangan agresif
- `agas/output/submission_v58ext_v4_full_only.csv`, `..._fac_full_only.csv` — dua kaki mentah, untuk diagnosis kalau ekstrapolasi meleset

Semua lolos validasi format (30000 baris, id unik dan cocok test, probabilitas di `[0,1]`).

### Saran submit berikutnya

1. `submission_v58ext_ext1.5_v4_fac_full_only.csv`
2. `submission_v58ext_ext1_v4_fac_full_only.csv` (kalau 1.5 ternyata overfit noise public lagi)

Jangan submit varian blend spekulatif baru (seperti `aggressive_145`) tanpa validasi arah ke public dulu.

