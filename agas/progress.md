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

- `agas/v57/training/train_submission_v57.py`

Isi utama:

1. Load fitur dari `features.py` (tanpa ubah source).
2. Build factor features per fold dengan `factor_inputs` + `factor_features`.
3. Training ensemble model multi-family (LGB/CAT/XGB/linear spline).
4. Build beberapa blend + satu blend agresif (`aggressive_145`).
5. Simpan beberapa kandidat submission sekaligus.

### 4. Eksekusi training

Command:

```bash
python3 agas/v57/training/train_submission_v57.py --folds 5 --fold-seed 42 --lr 0.01 --lgb-seeds 0,1,2 --out-prefix v57
```

Durasi:

- `37.33` menit

### 5. Hasil CV (OOF)

Sumber: `agas/v57/output/summary_v57.json`

- `top_mix`: AUC `0.846782`, KS `0.526606`
- `linear30`: AUC `0.846740`, KS `0.525565`
- `trees_only`: AUC `0.846534`, KS `0.525488`
- `extra31_mix`: AUC `0.846586`, KS `0.525972`
- `meta_logit`: AUC `0.846772`, KS `0.526435`
- `aggressive_145`: AUC `0.846725`, KS `0.526988` (**best KS run ini**)

### 6. Artefak yang dihasilkan

Folder output:

- `agas/v57/output/submission_v57_best.csv`
- `agas/v57/output/submission_v57_aggressive_145.csv`
- `agas/v57/output/submission_v57_top_mix.csv`
- `agas/v57/output/submission_v57_meta_logit.csv`
- `agas/v57/output/submission_v57_linear30.csv`
- `agas/v57/output/submission_v57_trees_only.csv`
- `agas/v57/output/submission_v57_extra31_mix.csv`

Dokumentasi:

- `agas/v57/output/train_log_v57.txt`
- `agas/v57/output/summary_v57.json`

### 7. Validasi submission

Semua file `submission_v57_*.csv` lolos:

- baris `30000`
- `application_id` unik
- set `application_id` sesuai test
- `default_probability` valid di `[0,1]`

### 8. Keputusan run ini

- Kandidat utama untuk submit: `agas/v57/output/submission_v57_best.csv`
- Kandidat backup: `agas/v57/output/submission_v57_top_mix.csv`, `agas/v57/output/submission_v57_meta_logit.csv`
- Tidak ada perubahan file existing di luar `agas/`

## 2026-10-14 (lanjutan, sore)

### 1. Hasil submit v57 dan evaluasi ulang

- `agas/v57/output/submission_v57_best.csv` (varian `aggressive_145`) disubmit, **hasil public hanya 0.557**.
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

Script baru: `agas/v58ext/training/train_submission_v58_ext.py`. Tidak mengubah `features.py`; kolom v4-only dihapus manual dari daftar kolom di script sendiri (bukan di source).

### 3. Eksekusi

Command:

```bash
python3 agas/v58ext/training/train_submission_v58_ext.py --fold-seeds 42,7,13 --folds 5 --lr 0.01 --lgb-seeds 0,1,2 --ext-ratios 1.0,1.5,2.0 --out-prefix v58ext
```

Durasi: `84.39` menit (3 fold seed x 2 varian x 5 model per varian).

### 4. Hasil CV (OOF), sumber `agas/v58ext/output/summary_v58ext.json`

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

- `agas/v58ext/training/train_submission_v58_ext.py` (script)
- `agas/v58ext/output/submission_v58ext_v4_full_only.csv`
- `agas/v58ext/output/submission_v58ext_fac_full_only.csv`
- `agas/v58ext/output/submission_v58ext_ext1_v4_fac_full_only.csv`
- `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` (**kandidat utama baru**)
- `agas/v58ext/output/submission_v58ext_ext2_v4_fac_full_only.csv`
- `agas/v58ext/output/summary_v58ext.json`, `agas/v58ext/output/train_log_v58ext.txt`

Semua file lolos validasi: 30000 baris, `application_id` unik dan cocok test, `default_probability` di `[0,1]`.

### 7. Rekomendasi submit berikutnya

1. `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` — replikasi resep terbukti 0.5617 dengan base model lebih kuat, OOF KS tertinggi sejauh ini.
2. `agas/v58ext/output/submission_v58ext_ext1_v4_fac_full_only.csv` — cadangan, rasio lebih konservatif kalau ekstrapolasi 1.5 ternyata overfit ke noise public pada split berikutnya.
3. Hindari submit ulang varian seperti `aggressive_145` (arah blend belum tervalidasi di public) kecuali ada bukti public baru yang mendukung arah itu.

### 8. Tidak dilakukan (di luar scope/waktu run ini)

- Tidak mencoba 10-fold (resep asli notebook pakai 10 fold, run ini pakai 5 fold untuk kecepatan). Bisa jadi langkah lanjutan kalau waktu submit masih ada sebelum deadline 30 Okt.
- Tidak mengubah file apa pun di luar `agas/`, tidak push ke GitHub.

## 2026-10-15 — Model alternatif setelah v58ext public 0.559

### 1. Trigger dan target

- Submission `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` mendapat public score `0.559`.
- Skor ini turun dari skor terbaik tim `0.5617`.
- Target eksperimen tetap `0.57`.
- Keputusan: jangan mengulang formula `ext1.5` sebagai satu-satunya arah. Uji keluarga model berbeda.

### 2. Metode alternatif A — ExtraTrees + HistGradientBoosting

Script: `agas/v59/training/train_submission_v59_alternative.py`

Langkah:

1. Pakai `features_cache.pkl` yang dibangun dari data kompetisi lokal.
2. Ambil fitur numerik saja.
3. Pilih 120 fitur per fold dengan separation score manual dari training fold.
4. Latih ExtraTrees dan HistGradientBoosting.
5. Gabungkan prediksi memakai rank blend `65% ExtraTrees + 35% HistGradientBoosting`.

Command:

```bash
python3 agas/v59/training/train_submission_v59_alternative.py --folds 5 --seed 2026 --top-features 120 --out-prefix v59alt
```

Durasi: kurang dari 10 menit.

OOF:

| Model | AUC | KS |
|---|---:|---:|
| ExtraTrees | 0.830321 | 0.499668 |
| HistGradientBoosting | 0.830995 | 0.499361 |
| Rank blend 65/35 | 0.832446 | 0.502018 |

Kesimpulan: model terlalu lemah dibanding v58ext. Tidak dipakai sebagai submission utama.

### 3. Metode alternatif B — XGBoost Lossguide + CatBoost Lossguide

Script: `agas/v59/training/train_submission_v59_boost.py`

Perbedaan dari v57/v58:

- Menggunakan pertumbuhan pohon `lossguide`, bukan depth-wise biasa.
- XGBoost memakai `grow_policy="lossguide"` dan `max_leaves=31`.
- CatBoost memakai `grow_policy="Lossguide"` dan `max_leaves=31`.
- Seleksi 180 fitur dilakukan manual dan fold-safe.
- Prediksi digabung memakai rank blend `55% XGBoost + 45% CatBoost`.

Command:

```bash
python3 agas/v59/training/train_submission_v59_boost.py --folds 5 --seed 2026 --top-features 180 --out-prefix v59boost
```

Durasi: kurang dari 10 menit.

OOF:

| Model | AUC | KS |
|---|---:|---:|
| XGBoost Lossguide | 0.842156 | 0.519982 |
| CatBoost Lossguide | 0.841099 | 0.516187 |
| Rank blend 55/45 | 0.842608 | 0.520637 |

Kesimpulan: lebih kuat dari metode A, tetapi masih di bawah v57/v58ext. Tidak dipakai sebagai submission utama.

### 4. Notebook reproducible

Notebook baru:

- `agas/v59/notebook/notebook_v59_alternative.ipynb`

Isi notebook:

- Batasan kompetisi.
- Jalur reproduksi kedua script alternatif.
- Pembacaan ringkasan OOF.
- Validasi semua submission v59.
- `RUN_TRAINING = True` untuk menjalankan ulang training.

Notebook JSON dan seluruh code cell sudah lolos syntax validation.

### 5. Artefak baru

- `agas/v59/training/train_submission_v59_alternative.py`
- `agas/v59/training/train_submission_v59_boost.py`
- `agas/v59/notebook/notebook_v59_alternative.ipynb`
- `agas/v59/output/summary_v59alt.json`
- `agas/v59/output/summary_v59boost.json`
- `agas/v59/output/submission_v59alt_*.csv`
- `agas/v59/output/submission_v59boost_*.csv`
- `agas/v59/output/train_log_v59alt.txt`
- `agas/v59/output/train_log_v59boost.txt`

Semua submission v59 tervalidasi: 30000 baris, application ID unik dan cocok dengan test, probabilitas berada di `[0,1]`.

### 6. Keputusan

- Tidak ada model alternatif v59 yang layak menggantikan v58ext berdasarkan OOF.
- Submission v59 tidak direkomendasikan sebagai kandidat utama.
- Hasil negatif tetap dicatat agar tidak mengulang metode lemah.
- File lama di luar `agas/` tidak diubah.
- Tidak push ke GitHub dan tidak submit ke Kaggle dari proses ini.

## 2026-10-15 21:06 WIB — Submission v60 konservatif

### Konteks

- v58ext `ext1.5` mendapat public `0.559`.
- Riwayat terbaik tetap `qc_00_ext1.5_v4_fac_full_only.csv` dengan public `0.56171`.
- v59 Lossguide dan ExtraTrees tidak mengalahkan baseline pada OOF.
- Strategi terbaik sekarang: pertahankan ranking submission public terbaik, lalu uji perturbasi kecil dari model lokal.

### Metode

Script: `agas/v60/training/make_submission_v60.py`

Input:

- Anchor: `agas/qc_00_ext1.5_v4_fac_full_only.csv`.
- `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv`.
- `agas/v58ext/output/submission_v58ext_ext1_v4_fac_full_only.csv`.
- `agas/v58ext/output/submission_v58ext_v4_full_only.csv`.
- `agas/v59/output/submission_v59boost_xgb_lossguide.csv`.

Formula utama:

```text
rank((1 - w) * rank(qc_anchor) + w * rank(model_lokal))
```

Bobot kecil `w=0.02` dan `w=0.05` dipakai agar kandidat tidak kehilangan sinyal yang sudah terbukti public. Kandidat 50/50 juga dibuat sebagai opsi berisiko lebih tinggi.

### Hasil validasi

Semua 10 file `submission_v60_*.csv` valid:

- 30000 baris.
- `application_id` unik.
- ID sama dengan test.
- `default_probability` berada di `[0,1]`.

Korelasi rank ke anchor:

- Perturbasi 2%: sekitar `0.99999`.
- Perturbasi 5%: sekitar `0.99994` sampai `0.99999`.
- Blend 50/50 qc dan v58 v4: `0.99874`.
- Blend 50/50 v58 ext1 dan v58 v4: `0.99742`.

### Kandidat

Prioritas:

1. `agas/v60/output/submission_v60_qc_98_v58_ext1_2_v58_v4.csv`
2. `agas/v60/output/submission_v60_qc_98_v58_ext15_2_v58_v4.csv`
3. `agas/v60/output/submission_v60_qc_95_v58_ext1_5_v58_v4.csv`
4. `agas/v60/output/submission_v60_mix_qc_v58_ext1.csv`
5. `agas/v60/output/submission_v60_anchor_qc.csv` sebagai kontrol public `0.56171`.

Tidak ada jaminan kandidat baru mencapai `0.57`. Hanya Kaggle dapat mengukur public score. Jangan menganggap perturbasi kecil otomatis lebih baik daripada anchor.

### Artefak

- `agas/v60/training/make_submission_v60.py`.
- `agas/v60/output/summary_v60.json`.
- `agas/v60/output/submission_v60_*.csv`.

Tidak ada perubahan di luar `agas/`. Tidak push GitHub.

## 2026-10-16 — v61: blok fitur baru, ensemble 10 fold, dan study blend

### 1. Konteks dan target

- Skor public terakhir yang dilaporkan: `0.559` (v58ext `ext1.5`).
- Skor public terbaik tim: `0.56171` (`agas/qc_00_ext1.5_v4_fac_full_only.csv`).
- Target: public `>= 0.57`, dan reporting KS OOF setinggi mungkin (`sekitar 0.55`).
- Batas yang bisa diterima: hanya `agas/` yang berubah, tidak ada push GitHub, tidak ada data eksternal, tanpa AutoML/AutoFE, notebook reproducible wajib.

### 2. Audit dan tool baru

Alat baru yang dibuat khusus supaya iterasi cepat (satu run CV penuh sekitar 1 menit):

- `agas/shared/cv_harness.py` — memuat cache fitur, menghitung factor features per fold, melatih satu LightGBM 5 fold, dan melaporkan AUC/KS. Mendukung `--drop` untuk mengukur kontribusi satu blok fitur lewat prefix.
- `agas/shared/feat_extra.py` — sebelas blok fitur baru, semua diturunkan dari tabel mentah dan stempel waktu keputusan.
- `agas/shared/build_extra.py` — membangun cache per blok di `agas/shared/cache/exblocks/` supaya blok baru bisa ditambah tanpa menghitung ulang tabel besar.
- `agas/shared/factor2.py` — factor model dengan enam sumber tambahan (diuji, lalu ditolak, lihat bagian 4).

### 3. Blok fitur baru (`agas/shared/feat_extra.py`)

| Blok | Prefix | Isi |
|---|---|---|
| `bureau` | `bx_` | jumlah provider dan laporan, usia laporan terbawah (`available_datetime`), lag provider, ketidaksepakatan antar provider, tren 180 hari untuk score/utilization/delinquency/inquiries |
| `repayment` | `rx_` | hari sejak telat terakhir, jumlah dan DPD berbobot half-life 60/180/365, curing rate cicilan yang akhirnya lunas, rate cicilan per Integrate |
| `previous` | `px_` | recency penolakan dan pembatalan, velocity pengajuan 30/90 hari,jendela 90 hari penolakan, keteraturan jarak pengajuan, konsentrasi penolakan |
| `account` | `ax_` | coefficient of variation pemasukan dan arus kas, streak bulan tanpa pemasukan, bulan dengan saldo minimum di bawah angsuran, jumlah bulan dengan saldo negatif |
| `ecommerce` | `ex_` | entropy kategori, jumlah item per order, rasio order terbesar terhadap rata-rata, diskresi versus kebutuhan, `lrel` berbobot half-life 30/90/240 |
| `interactions` | `ix_` | interaksi `interest_rate` dengan DTI, pendapatan, jenis kerja, sewa, riwayat telat, dan `ec_price_rel` |
| `rxtra` | `r2_` | sejarah affordability (amount_due maksimum, tren per loan), rate lunas tepat waktu |
| `bxtra` | `b2_` | kedalaman dan rentang file biro, slope score terhadap waktu, jumlah bulan terlcover |
| `axtra` | `a2_` | CV dispose bulanan, slope saldo dan pemasukan, proporsi pengeluaran melebihi pemasukan |
| `ecommerce2` | `e2_` | rentang aktif, order per bulan, tingkat pengulangan kategori, tren kuantitas |
| `gradenorm` | `gn_` | z-score dalam `interest_rate`, `employment_type`, `region` (diuji, lalu dibuang) |

### 4. Kontribusi per blok (LightGBM tunggal, 5 fold, seed 42, `lr 0.02`, `lambda_l2 20`)

Referensi: `auc 0.845867 / ks 0.525465` dengan semua blok aktif.

| Blok yang dibuang | AUC | KS | Selisih KS dari referensi |
|---|---|---|---|
| `gn_` | 0.845873 | 0.525771 | −0.00031 (blok ini merusak, dibuang) |
| `bx_` | 0.845598 | 0.523770 | +0.00170 |
| `rx_` | 0.845545 | 0.523067 | +0.00240 |
| `ax_` | 0.845596 | 0.524418 | +0.00105 |
| `ex_` | 0.845604 | 0.524543 | +0.00092 |
| `ix_` | 0.845663 | 0.524372 | +0.00109 |
| `px_` | 0.845810 | 0.524552 | +0.00091 |
| `r2_` | 0.845316 | 0.521504 | +0.00379 |
| `b2_` | 0.845499 | 0.522230 | +0.00306 |
| `e2_` | 0.845459 | 0.524243 | +0.00105 |
| `a2_` | 0.845464 | 0.524948 | +0.00034 |

Kontrol dengan semua blok dibuang (`bx_,rx_,px_,ax_,ex_,ix_,r2_,b2_,a2_,e2_,gn_`), rata-rata tiga fold seed (`42,7,13`):

| Konfigurasi | AUC | KS | KS per seed |
|---|---|---|---|
| kontrol, `lr 0.02`, `lambda_l2 20` | 0.845848 | 0.524259 | 0.52364, 0.52421, 0.52493 |
| dengan blok baru, `lr 0.02`, `lambda_l2 20` | 0.845935 | 0.525696 | 0.52568, 0.52577, 0.52564 |
| dengan blok baru, `lr 0.01`, `lambda_l2 5` | 0.846106 | 0.524734 | 0.52385, 0.52562, 0.52474 |
| kontrol, `lr 0.01`, `lambda_l2 5` | 0.846103 | 0.524726 | — |

Pelajaran: pada konfigurasi regularisasi kuat (`lambda_l2 5`, `lr 0.01`) blok baru tidak menambah apa pun karena regularisasi sudah menutup noise-nya. Kontribusinya nyata tapi kecil, sekitar `+0.0014 KS` pada konfigurasi yang dipakai tim sebelumnya.

### 5. Tuning hyperparameter (LightGBM tunggal, rata-rata 3 fold seed)

| Konfigurasi | AUC | KS |
|---|---|---|
| `lr 0.01`, `lambda_l2 5`, `leaves 15`, `ff 0.3`, `bag 0.8` | 0.846106 | 0.524734 |
| `lr 0.01`, `lambda_l2 20` | 0.846065 | 0.524503 |
| `lr 0.02`, `lambda_l2 5` | 0.845863 | 0.525631 |
| `lr 0.02`, `lambda_l2 20` | 0.845848 | 0.524259 |
| `num_leaves 31` | turun | turun |
| `ff 0.2` / `ff 0.5` | turun | turun |
| `min_child_samples 100` | turun | turun |
| `bag 0.6` / `bag 1.0` | turun | turun |
| `max_bin 255` | turun | turun |
| `lr 0.04` | turun | turun |

Jumlah fold (rata-rata 2 fold seed, `lr 0.01`, `lambda_l2 5`):

| Fold | AUC | KS |
|---|---|---|
| 5 | 0.846106 | 0.524734 |
| 10 | 0.846943 | 0.526064 |

10 fold dipakai di run utama karena selisihnya konsisten dan positif.

### 6. Factor model v2 (ditolak)

`agas/shared/factor2.py` menambahkan enam sumber ke factor model: `-log(1+income_like_credits)`, dispose bulanan, bulan dengan saldo minimum di bawah angsuran, selisih kas dengan angsuran dalam log, lateness berbobot half-life 180, dan deviasi harga absolut. Hasil (rata-rata 3 fold seed, `lr 0.02`, `lambda_l2 20`):

- factor v1: AUC 0.845935, KS 0.525696
- factor v2: AUC 0.845899, KS 0.524641

Kesimpulan: sumber tambahan tidak memperbaiki factor model. `factor2.py` disimpan sebagai catatan hasil negatif.

### 7. Run utama v61

Script: `agas/v61/training/train_submission_v61.py`

```bash
python3 agas/v61/training/train_submission_v61.py --folds 10 --fold-seeds 42,7,13 --lgb-seeds 0,1,2 --lr 0.01 --l2 5 --out-prefix v61
```

Durasi: `193.56` menit.

Konfigurasi:

- 10 fold `StratifiedKFold`, tiga fold seed (`42`, `7`, `13`), hasil OOF dirata-ratakan antar fold seed.
- LightGBM pada dua set fitur dan tiga seed per set, CatBoost, XGBoost, LightGBM konfigurasi ekstra, dan spline logistic pada set `lean`.
- 542 kolom total: 314 dari `features.py` plus blok baru. `lean` membuang 108 kolom yang sebelumnya ditandai noise. `fac` membuang blok putaran kedua (`r2_`, `b2_`, `a2_`, `e2_`) sehingga 497 kolom.
- Factor features dihitung ulang di dalam setiap fold dari label fold training saja.

OOF tiap model (setelah dirata-ratakan tiga fold seed):

| Model | AUC | KS |
|---|---:|---:|
| `lgb_full` | 0.846647 | 0.524615 |
| `lgb_lean` | 0.846566 | 0.523214 |
| `lgbx_full` | 0.846351 | 0.524416 |
| `xgb_full` | 0.846485 | 0.525240 |
| `cat_full` | 0.845928 | 0.525236 |
| `linear` (spline) | 0.842667 | 0.522833 |
| `fac_full` | 0.846542 | 0.523080 |

Per fold seed (seed 42 sampai 13), `lgb_full`: 0.845897, 0.846401, 0.846259; setelah dirata-ratkan 0.846647.

### 8. Study blend dan pemilihan rasio

Script: `agas/v61/training/study_blend_v61.py`, `agas/v61/training/make_submission_v61.py`, `agas/v61/training/make_submission_v61_final.py`.

Masalah: KS di 70.000 baris OOF punya galat standar sekitar 0.002 sampai 0.003, jadi memilih satu rasio terbaik dari satu vektor OOF hanyaArtefak noise fitting. Karena itu study ini:

- Menilai 64 kombinasi (4 kaki atas x 2 kaki bawah x 8 rasio).
- Mengulang KS pada enam sampel acak setengah OOF dan melaporkan mean serta nilai terburuk.
- Ekstrapolasi dibatasi hanya pada sumbu `full - fac`, satu-satunya arah yang sebelumnya menghasilkan public terbukti.

Hasil (beberapa baris teratas):

| Blend | Rasio | AUC | KS OOF penuh | KS setengah (mean) | KS setengah (min) |
|---|---:|---:|---:|---:|---:|
| `topmix20 - fac_full` | 0.75 | 0.846780 | 0.527968 | 0.531449 | 0.525629 |
| `topmix20 - fac_full` | 1.00 | 0.846487 | 0.527660 | 0.531372 | 0.526320 |
| `topmix20 - fac_full` | 1.25 | 0.846112 | 0.527842 | 0.531332 | 0.526461 |
| `topmix20 - fac_full` | 1.50 | 0.845662 | 0.527838 | 0.530951 | 0.526216 |
| `topmix20 - fac_full` | 0.50 | 0.846992 | 0.527151 | 0.530866 | 0.525233 |
| `topmix30 - lgb_lean` | 0.50 | 0.846699 | 0.528624 | 0.531424 | 0.525316 |
| `topmix20` (tanpa ekstrapolasi) | — | 0.847177 | 0.525791 | — | — |
| `meta_logit` | — | 0.847157 | 0.525870 | — | — |
| `trees4` | — | 0.846986 | 0.525406 | — | — |
| `fac_full` | — | 0.846542 | 0.523080 | — | — |

Bacaan:

- Arah `full - fac` menaikkan KS di semua rasio 0.25 sampai 2.0 dibanding `topmix20` polos (0.525791). Rasio 0.75 sampai 1.5 membentuk plateaux lebar, jadi pilihan rasio tidak bergantung pada satu puncak OOF.
- AUC turun saat rasio membesar (`ext2.0` AUC 0.844506, `ext3.0` 0.841216). Ini perilaku yang sama seperti file tim yang skornya 0.56171, jadi bukan tanda kegagalan, tapi rasio besar tetap tidak direkomendasikan.
- `topmix30 - lgb_lean` rasio 0.5 punya KS OOF tertinggi di run ini (0.528624), tapi sumbunya bukan `full - fac` yang sudah terbukti public, dan selisihnya 0.0008 di bawah noise. Dipakai sebagai kandidat diversifyasi, bukan kandidat utama.

KS OOF tertinggi yang dicapai run ini: **0.528624**. Target `0.55` tidak tercapai dan tidak realistis: analisis batas di `PROGRESS.md` memperkirakan AUC maksimum teoretis sekitar 0.869 sementara model sekarang sudah di 0.847, dan noise public 0.0015 sampai 0.003 membuat selisih OOF di bawah 0.002 tidak dapat dibedakan dari noise.

### 9. Kandidat final dan validasi

| Kandidat | OOF AUC | OOF KS | Spearman vs anchor |
|---|---:|---:|---:|
| `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` | 0.846699 | 0.528624 | 0.985917 |
| `agas/v61/output/submission_v61_ext0.75_full_fac.csv` | 0.846780 | 0.527968 | 0.986582 |
| `agas/v61/output/submission_v61_consensus.csv` | 0.846775 | 0.527879 | 0.986438 |
| `agas/v61/output/submission_v61_ext1.5_full_fac.csv` | 0.845662 | 0.527838 | 0.981550 |
| `agas/v61/output/submission_v61_ext1.0_full_fac.csv` | 0.846487 | 0.527660 | 0.985179 |
| `agas/v61/output/submission_v61_meta_logit.csv` | 0.847157 | 0.525870 | 0.989369 |
| `agas/v61/output/submission_v61_topmix20.csv` | 0.847177 | 0.525791 | 0.989121 |
| `agas/v61/output/submission_v61_anchor15_consensus.csv` | — | — | 0.999696 |
| `agas/v61/output/submission_v61_anchor5_consensus.csv` | — | — | 0.999966 |
| `agas/v61/output/submission_v61_anchor35_consensus.csv` | — | — | 0.998340 |

`consensus` adalah rata rank dari `ext0.75`, `ext1.0`, `ext1.5` pada sumbu `full - fac`, `alt_ext0.5_tm30_lean`, dan `meta_logit`. Tujuannya menghapus risiko pemilihan satu rasio.

Semua 31 file `submission_v61_*.csv` lolos validasi: 30000 baris, `application_id` unik dan sama dengan `data/test_applications.csv`, `default_probability` berada di `[0,1]` dan tanpa NaN.

### 10. Urutan submit yang disarankan

1. `agas/v61/output/submission_v61_consensus.csv` — OOF KS 0.527879 di keluarga `full - fac`, tidak bergantung pada satu rasio.
2. `agas/v61/output/submission_v61_ext1.5_full_fac.csv` — rasio yang sama dengan file public terbaik tim, untuk menguji apakah base model yang lebih kuat menambah skor.
3. `agas/v61/output/submission_v61_ext0.75_full_fac.csv` — ujung lain dari plateaux, lebih konservatif.
4. `agas/v61/output/submission_v61_meta_logit.csv` — AUC OOF tertinggi tanpa ekstrapolasi; submit sebagai kontrol untuk mengukur apakah arah ekstrapolasi benar-benar membantu di public.
5. `agas/v61/output/submission_v61_anchor15_consensus.csv` — 15 persen consensus baru di atas anchor terbukti, untuk slot harian yang harus aman.

### 11. Artefak baru

- `agas/shared/feat_extra.py`, `agas/shared/build_extra.py`, `agas/shared/cv_harness.py`, `agas/shared/factor2.py`.
- `agas/v61/training/train_submission_v61.py`, `agas/v61/training/make_submission_v61.py`, `agas/v61/training/make_submission_v61_final.py`, `agas/v61/training/study_blend_v61.py`.
- `agas/v61/notebook/notebook_v61.ipynb` (notebook reproducible, JSON dan seluruh code cell lolos syntax validation).
- `agas/v61/output/preds_v61.npz` (semua OOF dan prediksi test), `agas/shared/cache/extra_cache.pkl`, `agas/shared/cache/exblocks/`.
- `agas/v61/output/summary_v61.json`, `agas/v61/output/summary_v61_blends.json`, `agas/v61/output/summary_v61_final.json`, `agas/v61/output/summary_v61_robustness.json`.
- `agas/v61/output/blend_report_v61.csv`, `agas/v61/output/blend_robustness_v61.csv`, `agas/v61/output/train_log_v61.txt`.

### 12. Batasan yang dipatuhi

- Tidak ada file existing di luar `agas/` yang diubah, `features.py` tidak disentuh.
- Tidak ada push ke GitHub dan tidak ada submit ke Kaggle dari proses ini.
- Hanya data lokal `data/`, preprocessing dan feature engineering manual, tanpa AutoML/AutoFE, tanpa data eksternal.
- Tidak ada kebocoran label: seluruh agregat historis dipotong pada `decision_datetime`, dan factor loading dihitung dari label fold training saja.

### 13. Verifikasi notebook

`agas/v61/notebook/notebook_v61.ipynb` sudah dieksekusi penuh dengan `RUN_TRAINING = False` (jalur baca cache) dan seluruh cell lolos: 14 cell, JSON valid, semua code cell lolos kompilasi sintaks, dan tabel validasi di cell terakhir menunjukkan semua file `submission_v61_*.csv` lolos. Notebook yang sudah dijalankan disimpan beserta outputnya sebagai bukti reproduksi.

## 2026-10-16 (lanjutan) — v62: gradien kaya-fitur empat tingkat dan base gabungan

### 1. Pemicu

- User melaporkan `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` mendapat public **0.56312**, naik dari 0.56171.
- Ini adalah bukti public kedua bahwa arah ekstrapolasi bekerja, dan strukturnya sama dengan resep pertama: model dengan fitur lebih kaya didorong menjauh dari model sejenis dengan fitur lebih sedikit.
- Jangkar public_best bergeser dari `qc_00` (0.56171) ke `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` (0.56312).

### 2. Konsekuensi dari bukti baru

Karena mekanismenya sama, langkah alamiah adalah memperpanjang gradien kaya-fitur dan selalu menyisakan setiap tingkatnya. Kalau `topmix30 - lgb_lean` (0.56312) dan `v4 - fac` (0.56171) sama-sama benar, maka lengan pengungkit yang lebih panjang seharusnya memberi lebih banyak ruang untuk menyudutkan perbedaan fitur.

Tangga yang dipakai di v62:

| Tingkat | Jumlah kolom | Isi |
|---|---:|---|
| `full` | 554 | semua fitur, termasuk blok konsistensi baru |
| `fac` | 509 | tanpa blok putaran dua `r2_`, `b2_`, `a2_`, `e2_` |
| `lean` | 446 | tanpa kolom noise yang sudah ditandai tim sebelumnya |
| `faclean` | 401 | tanpa noise dan tanpa blok putaran dua, tingkat terkurus |
| `top` | 200 | terpilih oleh gain, dihitung per fold |
| `topwide` | 350 | terpilih oleh gain,cup lebih longgar |

### 3. Blok fitur putaran tiga (`consistency`, prefix `cx_`)

Tiga belas fitur selisih antar sumber. Logikanya: nasabah yang bersih di bureau tapi tidak bersih di riwayat cicilan internal berperilaku berbeda dari yang dilaporkan bureau, dan celah itu sendiri membawa informasi risiko.

- `cx_bureau_vs_declared_debt`, `cx_bureau_debt_vs_visible_inst`
- `cx_bureau_accounts_vs_our_loans`, `cx_inquiries_vs_our_apps`
- `cx_delinq_gap`, `cx_bureau_delinq_per_internal_late`
- `cx_file_age_gap_days`
- `cx_inst_vs_hist_max`, `cx_inst_vs_hist_mean`, `cx_remaining_vs_inst`
- `cx_recent_app_pressure`, `cx_bureau_known_but_no_apps`

Pengukuran (LightGBM tunggal, rata-rata 3 fold seed `42, 7, 13`):

| Konfigurasi | Dengan `cx_` | Tanpa `cx_` | Selisih KS |
|---|---|---|---|
| `lr 0.02`, `lambda_l2 20` | 0.524769 | 0.524178 | +0.00059 |
| `lr 0.01`, `lambda_l2 5` | 0.524794 | 0.523821 | +0.00097 |

Konsisten positif di kedua konfigurasi, jadi blok ini dipakai.

### 4. Run utama v62

```bash
python3 agas/v62/training/train_submission_v62.py --folds 10 --fold-seeds 42,7,13,21 --lr 0.01 --l2 5 --full-seeds 0,1 --top-n 200 --top-wide 350 --out-prefix v62
```

Durasi: `241.9` menit, empat fold seed (v61 hanya tiga). OOF setiap kaki disimpan per fold seed lewat `oofseed_<key>_<j>` supaya arah ekstrapolasi bisa diuji per seed, bukan hanya pada vektor OOF gabungan.

OOF model dasar, dirata-ratakan empat fold seed:

| Model | AUC | KS | KS per seed |
|---|---:|---:|---|
| `lgb_top` (200 fitur) | **0.847419** | 0.525550 | 0.52674, 0.52815, 0.52436, 0.52382 |
| `lgb_topwide` (350 fitur) | 0.846663 | 0.524101 | 0.52438, 0.52492, 0.52365, 0.52328 |
| `lgb_lean` | 0.846645 | 0.524386 | 0.52475, 0.52524, 0.52322, 0.52354 |
| `lgb_full` | 0.846629 | 0.523196 | 0.52394, 0.52509, 0.52257, 0.52263 |
| `xgb_full` | 0.846639 | 0.524982 | 0.52449, 0.52612, 0.52460, 0.52525 |
| `lgb_fac` | 0.846392 | 0.523101 | 0.52375, 0.52325, 0.52293, 0.52185 |
| `lgb_faclean` | 0.846444 | 0.523102 | 0.52251, 0.52441, 0.52277, 0.52231 |
| `lgbx_full` | 0.846305 | 0.523064 | 0.52284, 0.52430, 0.52250, 0.52120 |
| `cat_full` | 0.846046 | 0.524848 | 0.52416, 0.52498, 0.52128, 0.52404 |
| `linear` | 0.842723 | 0.521687 | 0.51989, 0.52139, 0.52149, 0.52197 |

Temuan utama: **seleksi fitur Based on gain di 200 kolom menghasilkan model terkuat sepanjang masa ini**, AUC 0.847419 dan KS 0.525550, di atas `lgb_full` 0.846629 pada 554 kolom. Feature selection bukan sekadar hiasan di run ini, dia memberi +0.0008 AUC pada model yang sama-sama dilatih dengan pipeline identik.

### 5. Plateau OOF KS

Ini yang paling penting untuk dibaca. v62 memang punya model peringkat lebih baik, tetapi KS tidak mengikuti:

| Run | KS OOF terbaik | AUC OOF terbaik |
|---|---:|---:|
| v57 | 0.526988 | 0.846782 |
| v58ext | 0.527511 | 0.847073 |
| v61 | 0.528624 | 0.847177 |
| v62 | 0.527846 | 0.847419 |
| v61 + v62 gabungan | **0.528717** | 0.847128 |

Selisih 0.0008 antara v61 dan v62 jauh lebih kecil dari galat standar KS OOF di 70.000 baris, yang sekitar 0.002 sampai 0.003. Kesimpulan: KS OOF mentok di sekitar 0.528 dan perbedaan di bawah 0.002 tidak bisa dibedakan dari noise. AUC naik, KS datar.

### 6. Protokol pemilihan arah

v62 menyimpan OOF per fold seed, jadi setiap arah diuji di dalam tiap seed secara terpisah:

- `n_pos` = berapa fold seed yang KS-nya membaik.
- `worst_delta` = selisih terburuk antar seed.
- `robust` = membaik di sedikitnya 3 dari 4 seed.
- Untuk base gabungan v61+v62, per-seed tidak tersedia karena v61 tidak menyimpan OOF per seed, jadi digantikan resampling baris: KS dihitung ulang pada delapan sampel acak setengah OOF, dan mean serta nilai terburuk dilaporkan.

Dua kelemahan metodologis yang ditemukan dan diperbaiki di run ini:

1. Pass pertama pencarian mistakenly memasukkan kaki reduksi ke dalam base blend, sehingga sumbu eksplorasi kabur. Diperbaiki: kaki yang diekstrapolasi selalu dikeluarkan dari base.
2. Base blend harus didefinisikan ulang untuk setiap pasangan base-kaki, bukan sekali untuk semua kaki.

### 7. Hasil pencarian

Aksis paling kuat di v62 adalah `faclean`, tingkat terkurus, dan membaik di keempat seed pada semua rasio 0.25 sampai 1.5. Contoh baris teratas:

| Base | Kaki | Rasio | KS | AUC | worst delta | n_pos |
|---|---|---:|---:|---:|---:|---:|
| `v62_w0.2` | `lgb_faclean` | 0.25 | 0.526537 | 0.846981 | +0.000497 | 4 |
| `v62_w0.1` | `lgb_fac` | 0.25 | 0.525967 | 0.847256 | +0.000332 | 4 |
| `v62_w0.2` | `lgb_faclean` | 0.375 | 0.527188 | 0.846712 | +0.000091 | 4 |
| `v62_w0.3` | `lgb_topwide` | 0.25 | 0.527846 | 0.846457 | −0.000866 | 3 |

Dari 675 kombinasi, hanya 58 yang lolos uji `robust`. Fakta yang perlu dicatat: baris dengan KS tertinggi hampir selalu bukan yang paling robust, yang persis seperti yang diharapkan kalau KS OOF dipakai berlebihan.

Base gabungan v61+v62, dinilai dengan resampling baris:

| Kandidat | KS | half_mean | half_min_delta |
|---|---:|---:|---:|
| `mix_w0.3 - lgb_topwide r0.25` | 0.528347 | 0.528232 | +0.001103 |
| `mix_w0.2 - fac_full r0.5` | 0.528198 | 0.527829 | +0.003307 |
| `mix_w0.2 - lgb_fac r0.5` | 0.527916 | 0.528223 | +0.003403 |

### 8. Dua sumbu sekaligus

Karena kedua arah yang terbukti memakai mekanisme yang sama, kedua sumbu diekstrapolasi bersamaan di atas base gabungan. Ini memberi KS OOF tertinggi sesi ini.

| Kandidat | KS | AUC | half_mean | Spearman vs anchor |
|---|---:|---:|---:|---:|
| `two_w0.3_fac0.25_fac61_0.5` | **0.528717** | 0.846784 | 0.528522 | 0.999653 |
| `two_w0.2_fac0.5_fac61_0.5` | 0.528251 | 0.846363 | 0.528538 | 0.998996 |
| `mixA_topwide_r025` | 0.528347 | 0.846424 | 0.528232 | 0.999744 |
| `mixB_facv61_r05` | 0.528198 | 0.846368 | 0.527829 | 0.998954 |
| `v61_fac_r025` | 0.528048 | 0.846303 | 0.527835 | 0.999476 |
| `v61_facv62_r05` | 0.527993 | 0.846703 | 0.527388 | 0.998242 |
| `consensus` | 0.527285 | 0.846668 | 0.527884 | 0.999564 |

KS OOF terbaik sepanjang sesi: **0.528717**. Peningkatan dari 0.528624 milik v61 hanya 0.00009, jadi secara praktis sama. Yang berubah nyata adalah base model: AUC 0.847419 dan sekarang ada empat fold seed serta feature selection per fold.

### 9. Kandidat final dan urutan submit

Faktor yang dipakai untuk memilih: KS OOF tinggi, arah yang terbukti atau konsisten per seed, dan Spearman terhadap jangkar yang cukup rendah supaya kandidat benar-benar berbeda dari file 0.56312. Kandidat dengan Spearman di atas 0.9995 hampir identik dengan jangkar, jadi mengisinya kembali hanya mengulang skor yang sudah diketahui.

| Prioritas | Kandidat | KS OOF | Spearman vs anchor |
|---|---|---:|---:|
| 1 | `agas/v62/output/submission_v62f_two_w0.2_fac0.5_fac61_0.5.csv` | 0.528251 | 0.998996 |
| 2 | `agas/v62/output/submission_v62f_mixB_facv61_r05.csv` | 0.528198 | 0.998954 |
| 3 | `agas/v62/output/submission_v62f_v61_facv62_r05.csv` | 0.527993 | 0.998242 |
| 4 | `agas/v62/output/submission_v62f_two_w0.3_fac0.25_fac61_0.5.csv` | 0.528717 | 0.999653 |
| 5 | `agas/v62/output/submission_v62f_consensus.csv` | 0.527285 | 0.999564 |
| 6 | `agas/v62/output/submission_v62f_anchor50_consensus.csv` | — | 0.999891 |

Alasan urutannya: tiga pertama punya KS tinggi sekaligus beda cukup jauh dari jangkar, jadi mereka yang benar-benar bisa menguji hipotesis baru. Nomor 4 adalah KS OOF tertinggi sepanjang sesi tetapi praktis identik dengan jangkar, jadi nilai informasinya rendah meskipun angkanya.

Semua 21 file `submission_v62f_*.csv` lolos validasi: 30000 baris, `application_id` unik dan sama dengan `data/test_applications.csv`, probabilitas di rentang [0,1] tanpa NaN.

### 10. Realitas target 0.57

Dengan jangkar sekarang di 0.56312, jarak ke 0.57 adalah 0.0069. Noise public yang tercatat di `PROGRESS.md` adalah 0.0015 sampai 0.003, jadi itu sekitar dua sampai empat kali noise. Gap KS OOF yang dibutuhkan kira-kira 0.02, sementara seluruh usaha feature engineering, tuning, dan blending di dua run terakhir menghasilkan total +0.0017 KS OOF. Rekomendasi jujur: perlakukan 0.57 sebagai batas atas kompetisi yang mungkin tidak terjangkau, dan pakai kuota submit yang tersisa untuk menguji sebanyak mungkin arah berbeda dengan KS OOF di atas rata-rata, bukan untuk mengejar satu angka OOF.

### 11. Artefak baru

- `agas/v62/training/train_submission_v62.py`, `agas/v62/training/search_gradient_v62.py`, `agas/v62/training/search_combined.py`.
- `agas/v62/training/make_submission_v62.py`, `agas/v62/training/make_submission_v62_final.py`.
- Blok `consistency` baru di `agas/shared/feat_extra.py`, cache di `agas/shared/cache/exblocks/consistency.pkl`.
- `agas/v62/output/preds_v62.npz` termasuk `oofseed_<key>_<j>` per fold seed.
- `agas/v62/output/summary_v62.json`, `agas/v62/output/summary_gradient_v62.json`, `agas/v62/output/summary_gradient_combined.json`, `agas/v62/output/summary_v62_final.json`.
- `agas/v62/output/gradient_v62.csv`, `agas/v62/output/gradient_combined.csv`, `agas/v62/output/train_log_v62.txt`.

### 12. Catatan alat

Beberapa `str.replace` berbasis shell gagal diam-diam karena karakter kurung siku pada string perintah dimakan oleh layer pemanggilan shell. Pola ini membuat/edit yang terlihat berhasil padahal tidak mengubah berkas, dan sempat menyesatkan selama beberapa percobaan. Perbaikan: pakai tool edit langsung, atau hindari kurung siku literal di perintah shell. Tidak ada akibat pada hasil numerik mana pun, hanya pada proses penulian.

## 2026-10-17 — Penataan ulang folder `agas/` per versi

### 1. Permintaan

User meminta folder `agas/` dirapikan dengan format versi, di dalamnya ada `output`, `training`, `notebook`, dan seterusnya.

### 2. Struktur akhir

```
agas/
  README.md, progress.md, limit.md
  qc_00_ext1.5_v4_fac_full_only.csv
  shared/
    paths.py, feat_extra.py, build_extra.py, cv_harness.py, factor2.py
    cache/  features_cache.pkl, extra_cache.pkl, exblocks/
  v57/ v58ext/ v59/ v60/ v61/ v62/
    training/  notebook/  output/
```

Pemilihan folder:

- Feature cache dipindah ke `shared/cache/` karena hampir semua versi memakainya, bukan milik satu versi.
- Tooling lintas versi (`feat_extra.py`, `build_extra.py`, `cv_harness.py`, `factor2.py`) masuk `shared/`.
- Jangkar `qc_00_ext1.5_v4_fac_full_only.csv` tetap di root `agas/` karena itu file tim, bukan keluaran eksperimen, dan dirujuk oleh beberapa versi.
- `OtherTries/` dibiarkan apa adanya karena bukan milik eksperimen ini.

Jumlah setelah ditata: v57 satu script, v58ext satu, v59 dua script plus satu notebook, v60 satu script, v61 lima script plus satu notebook, v62 lima script plus satu notebook. Total 112 entri dipindahkan.

### 3. Penyesuaian path

Semua script sebelumnya menghitung root sebagai `HERE.parent`, yang rusak begitu script dipindah satu tingkat lebih dalam. `shared/paths.py` sekarang menjadi satu-satunya sumber path: `ROOT` untuk `features.py` dan `data/`, `CACHE` untuk feature cache, `ANCHOR` untuk jangkar public, `version_output_of(nama)` untuk output versi lain, dan `version_output()` untuk output versi sendiri.

Nilai metrik tidak berubah sama sekali karena semua output dihitung ulang dari cache yang sama dan blend yang sama.

### 4. Kesalahan yang muncul selama penataan

Tiga masalah nyata, semuanya ketahuan lewat verifikasi:

1. Kaki reduksi ikut masuk ke base blend saat pencarian arah v62 pass pertama, sehingga sumbu eksplorasi kabur. Diperbaiki: base dibangun ulang untuk setiap pasangan base-kaki.
2. Header baru memakai `sys.path.insert` tapi beberapa script tidak mengimpor `sys`. Ditambahkan dengan pemeriksaan AST.
3. `paths.version_output()` tanpa argumen mengembalikan folder `shared`, karena `SCRIPT` merujuk ke `paths.py` itu sendiri, bukan pemanggilnya. Diperbaiki dengan mewajibkan argumen script.

Notebook sempat rusak karena penulisan ulang berbasis offset panjang karakter menggeser isi cell setelah penggantian pertama. Kedua notebook ditulis ulang dari awal dengan path baru, lalu dieksekusi penuh untuk memastikan benar.

### 5. Verifikasi setelah penataan

- Semua 20 script lolos pemeriksaan sintaks.
- `make_submission_v60.py`, `study_blend_v61.py`, `search_multiaxis_v61.py`, `search_gradient_v62.py`, `search_combined.py`, `make_submission_v61_final.py`, dan `make_submission_v62_final.py` dijalankan ulang dan menghasilkan angka identik: KS OOF terbaik v61 `0.528624`, v62 `0.528717`.
- `cv_harness.py` dijalankan ulang dan berhasil membaca cache bersama.
- 79 file submission di seluruh versi divalidasi ulang: 30000 baris, ID unik dan sama dengan test, probabilitas di rentang [0, 1], tanpa NaN. Tidak ada yang invalid.
- `git status` menunjukkan tidak ada perubahan di luar `agas/`.
