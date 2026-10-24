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

## 2026-10-17 — v63: feature selection sebagai lever utama, KS OOF 0.529867

### 1. Permintaan dan titik awal

User meminta submission yang lebih kuat dengan aturan yang sama, target KS OOF `0.53` ke atas. Rekor sebelumnya `0.528717` dari v62, jadi yang perlu dicari adalah `+0.0013`.

### 2. Hipotesis awal, dan bahwa hipotesis itu salah

Dugaan awal: `lgb_top` di v62 memakai 200 kolom tapi tetap `feature_fraction` 0.3, jadi tiap pohon hanya melihat 60 dari 200 kolom itu, dan parameter untuk set terseleksi seharusnya lebih lebar.

Sweep dua belas konfigurasi, masing-masing rata-rata tiga fold seed `42`, `7`, `13`, disprove:

| Konfigurasi | AUC | KS | sd antar seed |
|---|---|---|---|
| tanpa seleksi, `ff 0.3` | 0.846102 | 0.524794 | 0.00064 |
| **top 200, `ff 0.3`** | 0.846141 | **0.527077** | **0.00017** |
| top 200, `ff 0.6` | 0.845785 | 0.527346 | 0.00117 |
| top 200, `ff 0.9` | 0.845535 | 0.524625 | 0.00066 |
| top 200, `ff 0.6`, `leaves 31` | 0.845619 | 0.525894 | 0.00063 |
| top 300, `ff 0.5` | 0.846150 | 0.525802 | 0.00036 |
| top 400, `ff 0.4` | 0.846195 | 0.525898 | 0.00037 |
| top 100, `ff 0.9` | 0.845127 | 0.523984 | 0.00054 |
| top 200, `ff 0.6`, `lambda_l2 2` | 0.845873 | 0.527163 | 0.00061 |
| top 200, `ff 0.6`, `lr 0.02` | 0.845547 | 0.525390 | 0.00075 |

Dua kesimpulan:

1. **Seleksi fitur adalah lever-nya.** Menambah 0.0023 KS, konsisten di ketiga seed, dengan sebaran antar seed paling kecil dari semua konfigurasi yang diuji.
2. **Hipotesis `feature_fraction` salah.** Menaikkannya ke 0.6 menaikkan KS 0.0003 tetapi menurunkan AUC 0.0004 dan memperbesar sebaran antar seed tujuh kali. KS OOF elects nilai yang lebih volatil, jadi 0.3 dipertahankan.

Ada puncak KS di sekitar 200 kolom: 100 dan 300 sampai 400 semuanya lebih buruk.

### 3. Run utama v63

```bash
python3 agas/v63/training/train_submission_v63.py --folds 10 --fold-seeds 42,7,13,21 --lr 0.01 --l2 5 --sel200-seeds 0,1 --out-prefix v63
```

Durasi `230.2` menit. Kaki-kaki: `sel100`, `sel200`, `sel200b`, `sel300`, `sel400`, `lgb_full`, `lgb_faclean`, `lgbx_full`, `cat_full`, `xgb_full`, `linear`. Seleksi dihitung per fold dari probe gain pada baris training fold itu saja.

OOF model dasar dirata-ratakan empat fold seed:

| Model | AUC | KS | KS per seed |
|---|---|---|---|
| `sel200` | **0.846981** | **0.526330** | 0.52548, 0.52613, 0.52452, 0.52320 |
| `sel300` | 0.846793 | 0.524864 | 0.52515, 0.52498, 0.52286, 0.52361 |
| `sel200b` | 0.846783 | 0.525837 | 0.52488, 0.52499, 0.52431, 0.52429 |
| `sel400` | 0.846730 | 0.524526 | 0.52423, 0.52556, 0.52338, 0.52385 |
| `xgb_full` | 0.846639 | 0.524982 | 0.52449, 0.52612, 0.52460, 0.52525 |
| `lgb_full` | 0.846502 | 0.523734 | 0.52278, 0.52600, 0.52324, 0.52227 |
| `lgb_faclean` | 0.846444 | 0.523102 | 0.52251, 0.52441, 0.52277, 0.52231 |
| `sel100` | 0.845208 | 0.523120 | 0.52259, 0.52234, 0.52219, 0.52315 |
| `lgbx_full` | 0.846305 | 0.523064 | 0.52284, 0.52430, 0.52250, 0.52120 |
| `cat_full` | 0.846046 | 0.524848 | 0.52416, 0.52498, 0.52128, 0.52404 |
| `linear` | 0.842762 | 0.520747 | 0.51983, 0.51911, 0.52029, 0.52009 |

`sel200` adalah model terkuat sepanjang masa ini untuk KS, dan `sel300` untuk AUC di v63.

### 4. Dua sumbu dan base lintas run

Rekor v62 `0.528717` datang dari dua sumbu ekstrapolasi sekaligus, bukan satu. Pola itu diterapkan lagi di v63, ditambah dua hal baru:

1. **Ketajaman seleksi sebagai sumbu.** Kedua arah yang terbukti di leaderboard mendorong model berfitur lebih kaya menjauh dari model berfitur lebih sedikit. Ketajaman seleksi adalah ukuran lain dari quantity yang sama, jadi `sel200` didorong menjauh dari `sel400` dengan mekanisme identik.
2. **Base lintas run.** v62 dan v63 memakai empat fold seed yang sama, jadi prediksinya bisa dirata-ratakan sebelum diekstrapolasi. Karena kedua run menyimpan OOF per fold seed, pemeriksaan ketahanan bisa dilakukan di dalam tiap seed untuk semua kandidat lintas run, tanpa perlu jatuh ke resampling baris.

Grid 1296 baris: empat base, delapan kaki, sembilan rasio untuk sumbu tunggal, dan semua pasangan kaki dengan tiga nilai per sumbu untuk dua sumbu.

Temuan:

- **Ketahanan arah naik drastis.** 557 dari 1296 baris robust, 43 persen. Di v62 hanya 58 dari 675, 8,6 persen. Artinya jauh lebih banyak arah yang benar-benar konsisten antar seed, bukan hanya bagus di rata-rata.
- **KS OOF terbaik satu titik 0.529767**, consensus lima titik **0.529867**.

### 5. Consensus, ukuran, dan verifikasi

Ukuran consensus diuji dari 2 sampai 12:

| Ukuran | KS OOF | AUC | KS per seed |
|---|---:|---:|---|
| **5** | **0.529867** | 0.847204 | 0.52697, 0.52799, 0.52625, 0.52763 |
| 6 | 0.529827 | 0.847263 | 0.52779, 0.52806, 0.52618, 0.52743 |
| 12 | 0.529620 | 0.847289 | 0.52812, 0.52824, 0.52640, 0.52768 |
| 10 | 0.529596 | 0.847267 | 0.52822, 0.52824, 0.52627, 0.52766 |
| 3 | 0.529426 | 0.847238 | 0.52710, 0.52828, 0.52562, 0.52847 |
| 9 | 0.529375 | 0.847289 | 0.52806, 0.52831, 0.52662, 0.52768 |

Perhatikan dua hal yang berbeda:

- **Consensus 5 punya KS OOF tertinggi**, tetapi rata-rata per seed lebih rendah.
- **Consensus 12 punya KS OOF lebih rendah, tetapi rata-rata per seed tertinggi**, artinya paling stabil. `best_single` juga punya profil per seed paling seragam, unbeaten di keempat seed terhadap base-nya sendiri.

Verifikasi tahan banting (`verify_v63_consensus.py`) membangun ulang consensus dari kode yang sama persis:

- Korelasi rank antara consensus yang dibangun ulang dan file `submission_v63f_consensus5_best.csv` = **1.00000000**. Angka di laporan ini benar-benar berasal dari file submission.
- Selisih consensus terhadap base tanpa ekstrapolasi, per fold seed: `[-0.00012, +0.00132, +0.00018, +0.00217]`. Positif di 3 dari 4 seed, mean `+0.00089`, terburuk `-0.00012`.
- KS per fold seed consensus: `[0.52697, 0.52799, 0.52625, 0.52763]`.

### 6. Kesalahan yang ditemukan dan diperbaiki

Tiga kesalahan nyata muncul di run ini:

1. **Base lintas run tidak pernah dibangun** di pass pertama karena `preds_v62.npz` dicari di `v63/output/`. Akibatnya laporan hanya memuat base v63. Diperbaiki dengan alamat file eksplisit.
2. **`two_axis` mengembalikan vektor, bukan nilai KS**, sehingga perhitungannya gagal diam-diam di percobaan verifikasi pertama.
3. **Verifikasi pertama memakai bobot yang sedikit berbeda** sehingga melaporkan consensus 0.528804 dan rekor v62 0.525100, dan perbandingannya tidak mengukur apa yang diklaimnya. Ketiga skrip sekarang memakai satu jalur kode yang sama, dan hasilnya cocok persis dengan file yang ditulis.

### 7. Nilai KS yang dicapai, dan batasnya

| Run | KS OOF terbaik | AUC OOF terbaik |
|---|---:|---:|
| v61 | 0.528624 | 0.847177 |
| v62 | 0.528717 | 0.847419 |
| **v63** | **0.529867** | 0.847558 |

KS OOF naik `+0.001150` dari rekor sebelumnya.

Batasnya harus disebut terang-terangan: **`0.529867` masih `0.00013` di bawah `0.530000`.** Pada 70.000 baris OOF, galat standar KS sekitar 0.002 sampai 0.003, jadi selisih sebesar `0.00013` tidak dapat dibedakan dari noise. Angka ini membulatkan ke `0.53`, tetapi tidak melampaui `0.53`.

Yang benar-benar membaik dan tidak bisa dibantah noise:

- Base model naik jelas. `sel200` KS 0.526330 dan AUC 0.846981 sebagai model tunggal, dan base gabungan v62+v63 KS 0.526287 dengan AUC 0.847450.
- Arah yang konsisten antar seed naik dari 8,6 persen menjadi 43 persen.
- Kandidat benar-benar berbeda dari jangkar: korelasi terhadap file public 0.56312 turun ke sekitar 0.995, dari 0.9996 milik kandidat v62. Ini informasi baru, bukan pengulangan.

### 8. Kandidat final

| Prioritas | Kandidat | KS OOF | AUC | Korelasi vs jangkar |
|---|---|---:|---:|---:|
| 1 | `submission_v63f_consensus5_best.csv` | 0.529867 | 0.847204 | 0.995281 |
| 2 | `submission_v63f_best_single.csv` | 0.529767 | 0.847304 | 0.994953 |
| 3 | `submission_v63f_consensus12.csv` | 0.529620 | 0.847289 | 0.995430 |
| 4 | `submission_v63d_cand02.csv` s.d. `cand05.csv` | 0.529591 s.d. 0.529267 | — | 0.993 s.d. 0.996 |
| 5 | `submission_v63f_anchor35_consensus.csv` | — | — | 0.999422 |

Dua kandidat pertama adalah pilihan utama. `consensus5_best` punya KS OOF tertinggi. `best_single` punya profil per seed paling seragam, unbeaten di keempat seed terhadap base-nya sendiri, dan korelasi terhadap jangkar paling rendah, jadi paling berbeda dari file yang sudah 0.56312.

Semua 20 file submission v63 lolos validasi: 30000 baris, `application_id` unik dan sama dengan `data/test_applications.csv`, probabilitas di rentang [0, 1] tanpa NaN.

### 9. Artefak

- `agas/v63/training/train_submission_v63.py`, `search_multirun_v63.py`, `make_submission_v63_final.py`, `verify_v63_consensus.py`, `final_v63.py`.
- `agas/v63/sweep_v63.py` dan hasilnya di `agas/shared/harness_output/sweep_v63.json`.
- `agas/v63/output/preds_v63.npz` termasuk `oofseed_<key>_<j>` per fold seed.
- `agas/v63/output/summary_v63.json`, `summary_v63_final.json`, `summary_v63_verify.json`, `summary_v63_multirun.json`.
- `agas/v63/output/gradient_v63.csv`, `gradient_v63_multirun.csv`, `train_log_v63.txt`, `sweep_v63.out`.
- `agas/v63/notebook/notebook_v63.ipynb`, sudah dieksekusi penuh dengan `RUN_TRAINING = False`, 19 cell, nol error.
- `agas/shared/cv_harness.py` sekarang punya opsi `--top-n` untuk seleksi fitur per fold.

### 10. Batasan yang dipatuhi

- Tidak ada file existing di luar `agas/` yang diubah. `features.py` tidak disentuh.
- Tidak ada push ke GitHub dan tidak ada submit ke Kaggle dari proses ini.
- Hanya data lokal `data/`, preprocessing dan feature engineering manual, tanpa AutoML dan AutoFE, tanpa data eksternal.
- Tidak ada kebocoran label: agregat historis dipotong pada `decision_datetime`, seleksi fitur memakai probe gain pada baris training fold saja, dan factor loading dihitung dari label fold training saja.

## 2026-10-17 (lanjutan) — v64: audit korelasi, diagnosis noise, dan perubahan objektif

### 1. Pemicu

User melaporkan submission v63 mendapat public **0.557**, padahal KS OOF-nya `0.529867`, tertinggi sepanjang proyek. User menyimpulkan tidak ada korelasi antara KS dan skor Kaggle.

### 2. Audit data yang tersedia

Empat file kita punya KS OOF dan public score keduanya terukur:

| File | KS OOF | Public | Catatan |
|---|---|---|---|
| v57 `aggressive_145` | 0.526988 | 0.55700 | arah spekulatif, belum pernah terbukti public |
| v58ext `ext1.5` | 0.527511 | 0.55900 | Spearman 0.9994 terhadap `qc_00`,Yet 0.0027 lebih rendah |
| tim `qc_00 ext1.5` | tidak terukur | 0.56171 | arah terbukti, KS OOF-nya tidak pernah diukur |
| v61 `alt_ext0.5_tm30_lean` | 0.528624 | **0.56312** | public terbaik sejauh ini |
| v63 kandidat terbaik | **0.529867** | 0.55700 | KS OOF tertinggi, public terburuk dari empat |

Pearson r pada empat pasangan itu **+0.019**.

### 3. Pair yang menentukan

`v58ext ext1.5` dan `qc_00` memiliki vektor rank dengan Spearman **0.9994**, hampir identik. Skor public mereka berbeda **0.0027**. Dua peringkat yang nyaris sama berjarak 0.0027 di leaderboard. Angka itu adalah ukuran noise split, dan lebih besar dari seluruh gain KS OOF yang berhasil dicapai di v61, v62, dan v63.

### 4. Diagnosis sampling noise

KS adalah maksimum atas ambang dari jumlah kumulatif, jadi noisy bahkan di 70000 baris. Diukur dengan mengambil ulang baris dari satu vektor prediksi yang tetap:

| Ukuran sampel | SD sampling KS |
|---|---|
| 70000 baris (ukuran OOF, bootstrap) | 0.0036 |
| 30000 baris (ukuran public) | **0.0053** |
| 9000 baris (dipakai mesin harvest lama) | 0.0113 |

Tiga 가지 konsekuensi, semuanya diukur:

1. **Seluruh peningkatan v61 ke v63 adalah 0.34 sampling sd.** Peningkatan yang dilaporkan `+0.001243`, sementara satu sampling sd adalah `0.0036`.
2. **Memilih kandidat berdasarkan KS OOF maksimum adalah memilih noise.** Untuk 1300 titik grid yang dicari di v63, argmax memberi kenaikan terukur sekitar `0.012` secara gratis, sepuluh kali lipat dari peningkatan yang dikejar.
3. **Jarak ke 0.57 adalah 1.30 public sd.** Butuh gain+OOF sekitar 0.02, sementara dua run terakhir total menghasilkan 0.0012.

### 5. Konsekuensi untuk desain run ini

Objektif v64 diubah dari KS ke **AUC**. AUC adalah rata-rata halus atas seluruh pasangan, bukan ambang terburuk, jadi sampling noise-nya jauh lebih kecil dan ia menjawab perbaikan model yang sungguhan. KS tetap dilaporkan, tetapi selalu bersama SD sampling-nya, tidak pernah sebagai kunci pengurutan.

Tiga perubahan konkret:

1. **Jumlah fold seed dinaikkan ke enam.** Ini satu-satunya lever yang efeknya monoton di semua eksperimen: 0.845897 pada satu seed, 0.846647 pada tiga, 0.847419 dengan seleksi di atasnya.
2. **Seleksi satu probe, bukan seleksi stabil.** Seleksi stabil diuji lebih dulu dan ditolak, lihat bagian 6.
3. **Kandidat dipilih dengan-spread, bukan argmax.** Builder v64 mengumpulkan semua vektor prediksi dari v61, v62, v63, v64, membuang yang korelasinya di atas 0.998, dan mengurutkan dengan AUC. Dua file yangnearly identik memberi leaderboard draw yang sama, jadi keragaman memberi nilai sementara duplikasi tidak.

### 6. Hasil negatif: seleksi stabil ditolak

Hipotesis: memilih kolom yang bertahan di lima probe gain independen lebih dapat diandalkan daripada satu probe. Diuji pada 10 fold, seed sama, LightGBM sama, hanya cara seleksi yang dibedakan.

| Seleksi | AUC | KS |
|---|---|---|
| satu probe, top 200 | 0.845506 | 0.522678 |
| stabil, 5 probe, top 200 | 0.845148 | 0.522046 |
| tanpa seleksi, 554 kolom | 0.845951 | 0.523256 |

Stabseleksi **turun 0.000358** terhadap satu probe dengan ukuran sama, dan kedua himpunan tumpang tindih 169 dari 200 kolom, 85 persen. Jadi lima probe itu sebagian besar sepakat, dan probe tambahan hanya menambah biaya tanpa menambah informasi. Ditolak, resep satu probe dari v63 dipertahankan.

Dari 554 kolom, sekitar 275 muncul di setidaknya satu probe dan sekitar 165 di 80 persen probe, jadi separuh kolom memang tidak pernah hijau di probe mana pun.

### 7. Perbaikan alat yang terjadi di tengah run

`shared/bootstrap.py` ditambahkan karena versi folder ada dua kedalaman, `agas/<versi>/script.py` dan `agas/<versi>/training/script.py`, sehingga indeks `parents[n]` yang dipatok salah untuk salah satunya. Enam perbaikan berturutan diperlukan dan setiap perbaikan ditemukan lewat verifikasi, bukan asumsi:

1. Bentuk `import shared.bootstrap` gagal karena `agas/` bukan package Python. Diganti pemuatan lewat `importlib`.
2. Helper ditemukan, tapi direktori yang dimasukkan ke `sys.path` adalah induknya, bukan `shared/` tempat `paths.py` berada. Empat perbaikan tambahan setelahnya tetap salah dan tidak ada yang berhasil sampai direvisi bahwa direktori yang dimasukkan adalah `shared/` itu sendiri.
3. Beberapa header tidak mengimpor `sys` di level modul.

Akibatnya 22 dari 31 script sempat rusak. Semuanya dipulihkan dan `verify_all_scripts.py` sekarang menjalankan setiap script untuk memastikan `import paths` benar-benar berhasil. Angka metrik v61 dan v62 diverifikasi ulang setelah perbaikan dan **tidak berubah**.

### 8. Artefak

- `agas/shared/bootstrap.py`, pemuatan `shared/paths.py` dari kedalaman folder mana pun.
- `agas/v64/audit_public_vs_oof.py`, audit empat pasangan OOF dan public.
- `agas/v64/diagnose_selection_noise.py`, pengukuran sampling noise dan biaya argmax.
- `agas/v64/check_stability_selection.py`, uji seleksi stabil yang ditolak.
- `agas/v64/verify_all_scripts.py`, verifikasi semua script.
- `agas/v64/training/train_submission_v64.py`, ensemble utama dengan enam fold seed.
- `agas/v64/make_submission_v64.py`, builder kandidat berbasis keragaman.

### 9. Hasil run utama v64, enam fold seed

Selesai dalam 340.7 menit, 10 fold, seed `42,7,13,21,5,101`. Angka per seed dalam kurung:

| Model | AUC | KS | AUC per seed |
|---|---|---|---|
| `xgb_full` | **0.846749** | 0.525485 | 0.84608, 0.84635, 0.84623, 0.84587, 0.84647, 0.84609 |
| `lgb_full` | 0.846705 | 0.524327 | 0.84603, 0.84622, 0.84595, 0.84574, 0.84623, 0.84623 |
| `sel400` | 0.846607 | 0.524095 | 0.84575, 0.84591, 0.84594, 0.84552, 0.84586, 0.84611 |
| `lgb_faclean` | 0.846528 | 0.524096 | 0.84600, 0.84607, 0.84578, 0.84584, 0.84611, 0.84595 |
| `lgbx_full` | 0.846477 | 0.524037 | 0.84565, 0.84563, 0.84559, 0.84563, 0.84588, 0.84605 |
| `sel200` | 0.846316 | 0.525290 | 0.84554, 0.84569, 0.84568, 0.84530, 0.84535, 0.84530 |
| `cat_full` | 0.846193 | 0.525368 | 0.84504, 0.84515, 0.84495, 0.84512, 0.84525, 0.84495 |
| `linear` | 0.842875 | 0.522065 | 0.84198, 0.84223, 0.84264, 0.84214, 0.84247, 0.84265 |

Dua hal yang perlu dicatat.

**`sel400` mengalahkan `sel200`.** Di v63 `sel200` menang dan `sel400` kalah. Urutannya berubah begitu kunci pengurutan diganti ke AUC. Ini konsekuensi langsung dari diagnosis: KS menghargai prediksi yang lebih ekstrem, dan `sel200` memang lebih ekstrem, sementara `sel400` lebih baik rata-ratanya. Kalau masih memakai KS sebagai kunci, pilihan ini akan terbalik dan salah.

**KS `sel200` turun dari 0.529867 di v63 ke 0.525290 di v64, sementara AUC-nya naik.** Dua seed tambahan membuat prediksi lebih halus, dan prediksi yang lebih halus menaikkan AUC tapi menurunkan KS. Ini bukti kedua bahwa KS dan AUC mengukur hal berbeda, dan KS yang naik justru sepanjang menambah smoothing.

`linear` tetap 0.0038 AUC di bawahGBM dan masuk dedup sebagai sumber keragaman pada percobaan pertama, sebelum lantai AUC menutupnya.

### 10. Pool kandidat akhir

88 vektor setelah blend, 79 lolos lantai AUC, 7 lolos deduplikasi korelasi rank 0.998:

| Slot | Nama | AUC | KS | KS dikurangi 1 sd | Spearman ke anchor |
|---|---|---|---|---|---|
| 1 | `ext0.25\|v62\|lgb_top-v62\|lgb_faclean` | 0.847485 | 0.525884 | 0.522275 | 0.9913 |
| 2 | `avg_sel_lean` | 0.847043 | 0.526490 | 0.522881 | 0.9918 |
| 3 | `v64\|xgb_full` | 0.846749 | 0.525485 | 0.521876 | 0.9946 |
| 4 | `ext0.5\|v63\|sel200-v62\|lgb_fac` | 0.846461 | 0.525958 | 0.522349 | 0.9866 |
| 5 | `ext0.75\|v61\|lgb_full-v61\|lgb_lean` | 0.846210 | 0.523636 | 0.520026 | 0.9929 |
| 6 | `v64\|cat_full` | 0.846193 | 0.525368 | 0.521759 | 0.9905 |
| 7 | `v63\|sel100` | 0.845208 | 0.523120 | 0.519511 | 0.9841 |

Kolom KS dikurangi satu sampling sd sengaja ditampilkan: begitu dikurangi, kandidat 2 tidak lagi yang terbaik, dan urutannya tidak lagi sama dengan urutan KS mentah. Inilah alasan angka KS tidak pernah dipakai sebagai kunci.

Nilai maksimum harapan atas 7 submit diverse adalah **+0.0072 public (1.35 sd)**. Ini hanya batas atas dengan asumsi draw independen. Kandidat terpilih saling berkorelasi 0.984 sampai 0.995, jadi gain sesungguhnya lebih kecil dan tidak dapat dihitung: butuh korelasi noise public itu sendiri, yang butuh beberapa pasangan ranking independen, sedangkan proyek hanya punya satu pasangan.

### 11. Dua bug yang ditemukan lewat verifikasi, bukan lewatenalty yang terlihat

**Slot output bertabrakan.** Tiga kali builder dijalankan, file lama tidak dihapus, dan nomor slot dipakai ulang dengan konten berbeda. Slot 3 pernah berisi `v61|lgb_full` dari run pertama dan `v64|xgb_full` dari run ketiga. Keduanya valid, keduanya mengklaim slot 3, jadi file mana yang terambil bergantung pada waktu pembuatannya. Builder sekarang menghapus prefix-nya sendiri sebelum menulis.

**Lantai AUC tidak ada di deduplikasi pertama.** Deduplikasi murni korelasi rank menyisakan 9 kandidat, tapi ekornya berisi `linear` AUC 0.8427 yang bertahan karena meranking berbeda, sementara blend terkuat saling dianggap duplikat. Itu trade buruk, karena semua kandidat punya sebaran public yang sama sehingga kandidat dengan rata-rata lebih buruk hanya bisa menurunkan nilai maksimum harapan. Lantai AUC 0.0025 di bawah terbaik menutupnya dan pool menyempit ke 7 kandidat yang semuanya AUC di atas 0.8452.

Nilai maksimum dengan asumsi independensi juga sebelumnya dicetak sebagai fakta, sekarang dilabeli batas atas, dan dicetak tiga kali karena loop-nya mengulang angka yang sama.

### 12. Status akhir

Tujuh file submission tervalidasi, 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di rentang [0, 1] tanpa NaN. Seluruh script diverifikasi ulang: 32 fine, 0 broken.

Belum ada submission Kaggle dan belum ada push GitHub. Public score v63 `0.557` dicatat di dokumentasi sebagai hasil, bukan diperbaiki.

## 2026-10-18 — v65: membidik selisih yang tidak bisa ditutup feature engineering

### 1. Pemicu dan koreksi atas kesimpulan sebelumnya

User menambahkan `agas/overview.md` dan `agas/rules.md`, dan melaporkan posisi leaderboard:

- Public score tertinggi saat ini **0.57768**, disusul 0.5753, lalu 0.5717.
- Skor kita **0.56312**, peringkat 29.

Ini mengoreksi satu kesimpulan yang sebelumnya saya tulis sendiri di v64. Di sana saya menulis bahwa 0.57 mungkin tidak terjangkau, berdasarkan argumen bahwa selisih OOF di bawah 0.002 tidak bisa dibedakan dari noise. Argumen itu benar untuk **mengukur** gain, tapi salah untuk **menilai posisi relatif**, karena ia memperlakukan skor public kita sendiri sebagai satu undian noise dan menyimpulkan batasnya. Leaderboard menunjukkan 0.577 ada. Jadi selisih 0.57768 − 0.56312 = **0.0146** adalah jarak antara dua model, dan jarak itu nyata.

Angka noise yang diukur di v64 tetap berlaku sebagai alat ukur: sebaran sampling KS di 30000 baris public sekitar **0.0053**. Jadi 0.0146 itu **2.75 sebaran**. Model di puncak leaderboard benar-benar lebih baik, dan gap-nya bukan noise. Kesimpulan v65: berhenti mengejar 0.002 KS OOF dan cari kualitas model yang lebih besar.

### 2. Audit apa yang belum pernah dicoba

`features.py` menyumbang 311 kolom, sebelas blok tambahan menyumbang sekitar 240, jadi 554 kandidat kolom sudah ada. Empat ide yang belum diuji:

1. **Kedalaman cakupan dan missingness.** Blok yang ada menghitung jumlah baris per tabel, jadi menangkap **ada atau tidak**, tapi tidak seberapa tidak lengkap. Di data biro, "tidak ada file biro" dan "file biro ada tapi skornya kosong" adalah dua sinyal berbeda.
2. **Arah bureaus, bukan hanya levelnya.** Blok bureau punya mean, maks, dan dua kolom tren, semuanya untuk score saja. Utilisasi, inquiry, dan delinquency yang **naik** tidak pernah dienkode.
3. **Beban cicilan terhadap arus kas terukur.** `features.py` punya debt-to-income atas income yang dideklarasikan. Income yang dideklarasikan tidak bisa diverifikasi; arus kas dan cicilan aktif bisa.
4. **Self-training pada test.** Langkah transduktif standar, dan legal karena hanya memakai data yang diberikan panitia.

### 3. Hasil pengujian blok fitur

Tiap blok diuji dengan tiga fold seed, karena satu fold seed bergerak sekitar 0.0006 AUC dari seed saja, ukuran yang sama dengan efek yang seharusnya diceap blok mana pun. Pipeline `lr 0.01`, `lambda_l2 5`, 5 fold, LightGBM, seleksi top-200.

| Konfigurasi | Kolom | AUC | KS |
|---|---|---|---|
| referensi, 554 + 3 blok baru | 625 | 0.845857 ±0.000216 | 0.526100 ±0.000110 |
| buang `dq_` | 602 | 0.845830 ±0.000221 | 0.525910 ±0.000304 |
| buang `bd_` | 604 | **0.846120** ±0.000089 | **0.527299** ±0.000632 |
| buang `af_` | 604 | 0.845914 ±0.000155 | 0.526540 ±0.001433 |

**`burdirection` ditolak.** Membuangnya menaikkan AUC 0.000263 dan KS 0.001199. Alasan yang paling mungkin: blok itu menambahkan 22 kandidat ke probe gain, dan probe gain itu memilih 200 kolom. Kandidat lemah itu tidak pernah terpilih, tapi tetap estatuto kandidat dancrowded lewat kolom yang sebelumnya terpilih.

**`dataquality` diterima**, kontribusinya kecil: +0.000027 AUC dan +0.000190 KS, dengan sebaran antar seed turun dari 0.000304 ke 0.000110.

**`stress` diterima**, nyaris netral: −0.000057 AUC, +0.000440 KS.

### 4. Level seleksi fitur diukur ulang

v63 menemukan 200 kolom sebagai puncak, tapi itu diukur di `lr 0.02 / lambda_l2 20`. Di `lr 0.01 / lambda_l2 5` dengan 604 kandidat jawabannya berbeda, dan yang lebih penting adalah sebaran antar seed, bukan cuma mean.

| Level | AUC | KS | sd KS antar seed |
|---|---|---|---|
| tanpa seleksi | **0.846330** ±0.000169 | 0.525506 | 0.001454 |
| **top 250** | 0.846122 ±0.000129 | **0.526768** | **0.000342** |
| top 400 | 0.846305 ±0.000188 | 0.525219 | 0.000217 |

Dua ratus lima puluh dipilih: mean KS paling tinggi dan sebaran paling kecil. Tanpa seleksi AUC-nya sedikit lebih tinggi, tapi sebaran KS-nya empat kali lebih lebar, dan justru selection noise itulah yang seharusnya dihilangkan oleh seleksi.

### 5. Self-training diuji lalu ditolak

Pseudo-labeling diukur tanpa menipu diri sendiri. Label semu diambil **hanya dari test**, dan skor dihitung pada validation fold yang tidak pernah masuk training di kedua tahap: fold-train → model A → label semu test → fold-train + test semu → model B → skor fold-val.

| Setelan | AUC sebelum | AUC sesudah | KS sebelum | KS sesudah |
|---|---|---|---|---|
| frac 0.20, bobot 0.5 | 0.846005 | 0.845477 | 0.526527 | 0.524218 |
| frac 0.10, bobot 0.2 | 0.846005 | 0.846012 | 0.526527 | 0.526834 |

Setelan kasar jelas merusak. Setelan halus terlihat positif secara mean, tapi delta KS **per seed** adalah `+0.001346`, `-0.001870`, `+0.001448`. Mean positifnya datang dari dua seed yang menutupi satu seed yang lebih besar. Itu pembatalan, bukan peningkatan.

Penyebabnya baru terlihat setelah dicek: **train dan test berinterleave pada rentang tanggal yang sama.** Keduanya 2025-01-01 sampai 2025-12-28, 100000 nasabah berbeda, dan tidak ada nasabah yang tumpang tindih. Keuntungan self-training datang dari pergeseran distribusi antara data berlabel dan data tanpa label, dan di sini tidak ada yang perlu digeser.

Implikasinya untuk run berikutnya: feature yang tersisa adalah feature yang benar-benar baru, dan eksperimen ini menunjukkan tabel ini sudah mengaburkan sebagian besar informasi yang tersedia.: yang benar-benar baru, dan eksperimen ini menunjukkan tabel ini sudah mengaburkan sebagian besar informasi yang tersedia.

### 6. Run utama v65

```bash
python3 agas/v65/training/train_submission_v65.py --folds 10 --fold-seeds 42,7,13,21,5,101 --top-n 250 --out-prefix v65
```

Durasi **86.4 menit**, 10 fold, enam fold seed, 598 kandidat kolom, empat kaki model. Jauh lebih murah dari v64 yang 340.7 menit untuk delapan kaki, karena kaki dikurangi dan tidak ada leg useless.

OOF kaki dasar, dirata-ratakan enam fold seed:

| Kaki | AUC | KS | AUC per seed |
|---|---|---|---|
| `xgb` | **0.847474** | 0.526091 | 0.846844, 0.846717, 0.846294, 0.846647, 0.847144, 0.846927 |
| `sel` | 0.847395 | **0.528338** | 0.846735, 0.847073, 0.846426, 0.846478, 0.846874, 0.846704 |
| `lean` | 0.847303 | 0.524940 | 0.846533, 0.846922, 0.846690, 0.846563, 0.846986, 0.846743 |
| `cat` | 0.846855 | 0.525920 | 0.846001, 0.846275, 0.846247, 0.845669, 0.846230, 0.846253 |

Blend:

| Blend | AUC | KS |
|---|---|---|
| `consensus_ranks` (rata rank 4 kaki) | **0.847708** | 0.527359 |
| `mix_all4` | 0.847682 | 0.527676 |
| `mix_trees` | 0.847667 | 0.527684 |
| `mix_sel_cat` | 0.847640 | 0.527651 |
| `mix_sel_xgb` | 0.847497 | 0.527373 |

AUC 0.847708 adalah yang tertinggi sepanjang proyek. Bandingkan kaki terbaik v64 0.846749: **+0.00096**.

### 7. Dua arah sumbu ekstrapolasi

Dua dari tiga public score di atas 0.559 datang dari satu mekanisme: mendorong model berfitur lebih kaya menjauh dari model berfitur lebih sedikit. `qc_00` 0.56171 memakai arah `kaya - miskin`; v61 0.56312 memakai `terpilih - luas`. Keduanya arah yang sama hanya karena di titik operasi v61 lebih banyak fitur berarti lebih baik.

Di titik operasi v65 gradiennya berbalik: kaki `sel` 250 kolom mengalahkan kaki `lean` 576 kolom. Jadi kedua arah dijalankan, tandanya tidak diasumsikan.

| Arah | Rasio | AUC | KS |
|---|---|---|---|
| `sel - lean` | 0.25 | 0.847263 | **0.528592** |
| `sel - lean` | 0.5 | 0.847071 | 0.528105 |
| `sel - lean` | 0.75 | 0.846815 | 0.527612 |
| `lean - sel` | 0.25 | 0.847127 | 0.525256 |
| `lean - sel` | 0.5 | 0.846896 | 0.524605 |

Arah v61 menang lagi. Dan hasil yang paling menarik di run ini: **KS OOF 0.528592 praktis menyamai KS OOF v61 sebesar 0.528624, sementara AUC-nya 0.847263 melawan 0.846699.** Ini kandidat pertama yang secara bersamaan cocok dengan v61 pada KS dan lebih baik pada AUC, dan v61 adalah file yang memberi 0.56312.

### 8. Pool kandidat

61 vektor setelah blend, 56 lolos lantai AUC, 8 lolos deduplikasi korelasi rank 0.998.

| Slot | Nama | AUC | KS | KS dikurangi 1 sd | Spearman ke anchor |
|---|---|---|---|---|---|
| 1 | `v65\|consensus_ranks` | 0.847708 | 0.527359 | 0.522074 | 0.9918 |
| 2 | `v62\|lgb_top` | 0.847419 | 0.525550 | 0.520265 | 0.9919 |
| 3 | `v63\|sel200` | 0.846981 | 0.526330 | 0.521045 | 0.9904 |
| 4 | `v65\|lean-sel_r0.5` | 0.846896 | 0.524605 | 0.519320 | 0.9900 |
| 5 | `v65\|cat` | 0.846855 | 0.525920 | 0.520635 | 0.9879 |
| 6 | `v65\|sel-lean_r0.75` | 0.846815 | 0.527612 | 0.522327 | 0.9882 |
| 7 | `v64\|xgb_full` | 0.846749 | 0.525485 | 0.520200 | 0.9946 |
| 8 | `v64\|cat_full` | 0.846193 | 0.525368 | 0.520083 | 0.9905 |

Kolom KS dikurangi satu sampling sd sengaja ditampilkan: begitu dikurangi, urutannya tidak lagi sama dengan urutan KS mentah.

Nilai maksimum harapan atas 8 submit diverse adalah **+0.0075 public (1.42 sd)**, hanya batas atas dengan asumsi undian independen. Kandidat terpilih saling berkorelasi 0.988 sampai 0.995, jadi gain sesungguhnya lebih kecil.

### 9. Status

Delapan file submission tervalidasi, 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di [0, 1] tanpa NaN. Jumlah file di disk sama persis dengan jumlah entri di `summary_v65_candidates.json`. Seluruh script diverifikasi ulang: 37 fine, 0 broken. Notebook `agas/v65/notebook/notebook_v65.ipynb` dieksekusi penuh dengan `RUN_TRAINING = False`, 18 cell, nol error.

### 10. Apa yang sebenarnya diketahui, dan apa yang tidak

**Diketahui.** Model v65 lebih baik dari v64 pada AUC, +0.00096 pada kaki terbaik. Blend empat mencapai 0.847708, tertinggi sepanjang proyek. Kandidat sumbunya punya KS OOF yang praktis menyamai v61 dengan AUC yang lebih baik. Seleksi 250 kolom memberi sebaran antar seed paling kecil dari semua konfigurasi yang diuji.

**Tidak diketahui.** Apakah itu masuk leaderboard. Yang diketahui: selisih public antara kita dan puncak leaderboard adalah 0.0146, atau 2.75 sebaran sampling, jadi gap-nya nyata dan model di puncak memang lebih baik. Yang belum ditemukan di sini: apa yang mereka pakai.

**Kemungkinan besar sudah habis.** Eksperimen ini menunjukkan feature engineering manual di dataset ini mendekati batasnya. Dua blok baru menambah 0.00003 AUC. Sembilan blok yang sudah ada mengaburkan sebagian besar informasi yang tersedia. Kalau v65 tidak menutup gap, eksperimen berikutnya sebaiknya mengarahkan energi ke dua hal: lebih banyak fold seed pada model terkuat, dan mencari sumber informasi yang sama sekali belum dipakai.



Rekomendasi jujur: perlakukan 0.57768 sebagai bukti bahwa gap bisa ditutup, bukan sebagai target yang bisa dikejar dengan satu run. Yang dilakukan run ini adalah menaikkan lantai model dan mengembalikan kandidat yang bisa menyamai v61 pada KS dengan AUC lebih baik. Catat public score aktual setiap file, karena itu satu-satunya ukuran yang benar, dan kirim kandidat yang saling berbeda.

## 2026-10-19 — v66: resep v61, dilatih lebih berat

### 1. Pemicu

User melaporkan kandidat v65 hanya mendapat public **0.559**, turun dari **0.56312** milik v61. Permintaan: pakai model yang sama dengan v61, tapi lebih kuat dan lebih baik dilatih.

### 2. Diagnosis: v65 menyimpang dari arsitektur v61

Membandingkan kedua run, selisihnya bukan hyper-parameter. Ada tiga perubahan struktural di v65, dan ketiganya menjauh dari satu-satunya mekanisme yang pernah menghasilkan public tinggi.

**1. v61 tidak memakai feature selection sama sekali.** Setiap kaki dilatih pada himpunan kolom yang didefinisikan tangan: `full`, `lean` (buang 12 kolom noise eksak dan prefix `ec_share_`, `ec_catqty_`, `cs_shift`, `buh_`, `gn_`), dan `fac` (buang prefix putaran dua `r2_`, `b2_`, `a2_`, `e2_`). v62 memperkenalkan seleksi gain per fold, dan v62–v65 semuanya memakainya.

**2. v65 tidak pernah membangun rantai blend v61.** Pemenang v61 adalah:

```text
trees4 = mean(rank(lgb_full), rank(lgb_lean), rank(cat_full), rank(xgb_full))
core   = (2 * trees4 + rank(lgbx_full)) / 3
tm30   = 0.7 * core + 0.3 * rank(linear)
alt    = rank01(tm30 + 0.5 * (tm30 - rank(lgb_lean)))
```

v65 tidak pernah membangun `tm30`. Dia mengekstrapolasi kaki terpilih (250 kolom) menjauhi kaki tidak terpilih (576 kolom), lalu merata-ratakan. Itu objek berbeda, dan arah gambarnya juga berbeda: di v65 kaki 250 kolom mengalahkan kaki 576 kolom, sedangkan di v61 `lgb_lean` yang himpunan kolomnya lebih kecil justru menjadi kaki yang didorong menjauh.

**3. `lgbx` dan `linear` hilang di v65.** Keduanya bagian dari rantai pemenang. `lgbx_full` punya konfigurasi LightGBM yang benar-benar berbeda (`num_leaves 31`, `lambda_l2 ×2.4`, `max_bin 255`, `feature_fraction 0.35`), dan `linear` adalah satu-satunya kaki non-tree. v66 hanya empat kaki dan tidak punya keduanya.

v61 juga merata-ratakan tiga LightGBM seed per kaki; v65 memakai satu.

### 3. Resep diverifikasi lebih dulu

Sebelum training apa pun, `alt_ext0.5_tm30_lean` dibangun ulang dari prediksi v61 yang tersimpan:

```
rebuilt  KS=0.5286237  AUC=0.8466992
target   KS=0.5286237  AUC=0.8466992
```

Cocok sampai tujuh desimal. Jadi rumusnya sudah diketahui benar sebelum ada training baru, dan perbandingan v66 dengan v61 jadi bersih.

### 4. Cacat nyata yang ditemukan saat smoke test

Kaki `linear` gagal dengan `ValueError: Input X contains NaN values and SplineTransformer is configured to error`. Penyebabnya bukan kode v66: dua kolom dari blok v65, yaitu **`af_loan_to_hist_max` dan `af_loan_to_hist_mean`, sepenuhnya NaN**. `median()` dari kolom kosong menghasilkan NaN, lalu `fillna(NaN)` tidak mengisi apa pun.

v61 tidak punya kedua kolom itu, jadi tidak pernah kena. Di v66 keduanya dibuang dari kaki linear lewat filter `X[c].notna().any()`, sekaligus mengembalikan kaki itu ke himpunan kolom yang sama dengan v61. Bug yang sama sudah pernah muncul sebagai bug path di `check_stability_selection.py` dan `make_submission_v64.py`: hal yang salah berakhir di tempat yang tidak diharapkan, dan tidak akan terlihat tanpa eksekusi.

### 5. Run utama

```bash
python3 agas/v66/training/train_submission_v66.py --folds 10 --fold-seeds 42,7,13,21,5,101 --lgb-seeds 0,1 --out-prefix v66
```

Durasi **378.8 menit**, 10 fold, enam fold seed, dua LightGBM seed, tujuh kaki.

Kenapa enam fold seed dan dua LightGBM seed, bukan lima LightGBM seed: jumlah fold seed adalah satu-satunya lever yang efeknya monoton di seluruh proyek (0.845897 satu seed, 0.846647 tiga seed, 0.847419 dengan seleksi di atasnya). Lima LightGBM seed dikali enam fold seed akan memakan sekitar 15 jam; kombinasi enam dan dua masuk dalam 6,3 jam.

OOF kaki, dibandingkan v61 yang memakai tiga fold seed dan tiga LightGBM seed:

| Kaki | AUC v66 | AUC v61 | Δ AUC | KS v66 | KS v61 |
|---|---|---|---|---|---|
| `xgb_full` | **0.846756** | 0.846485 | +0.000271 | 0.525536 | 0.525240 |
| `lgb_lean` | **0.846721** | 0.846566 | +0.000155 | **0.525311** | 0.524615 |
| `lgb_full` | 0.846682 | 0.846647 | +0.000035 | 0.524974 | 0.524615 |
| `fac_full` | 0.846662 | 0.846542 | +0.000120 | 0.524205 | 0.523080 |
| `lgbx_full` | 0.846550 | 0.846351 | +0.000199 | 0.524089 | 0.524416 |
| `cat_full` | 0.846311 | 0.845928 | +0.000383 | 0.525234 | 0.525236 |
| `linear` | 0.842771 | 0.842667 | +0.000104 | 0.522712 | 0.522833 |

Tujuh kaki naik, tanpa kecuali. `xgb_full` naik paling besar (+0.000271), `cat_full` paling besar di relatif (+0.000383). Kenaan kecil dan konsisten dengan pengurangan varians partisi, bukan peningkatan kapasitas model: tidak ada parameter yang diubah.

### 6. Rantai blend, v66 lawan v61

| Rantai | KS v66 | AUC v66 | KS v61 | AUC v61 |
|---|---|---|---|---|
| `trees4` | 0.525693 | 0.847142 | 0.525406 | 0.846986 |
| `core` | 0.525719 | 0.847072 | 0.525263 | 0.846930 |
| `tm20` | **0.526970** | **0.847312** | 0.525791 | 0.847177 |
| `tm30` | **0.528476** | 0.847223 | 0.526567 | 0.847088 |

Resep SBS winner, kedua arah ekstruoplasi:

| Resep | KS v66 | AUC v66 | KS v61 | AUC v61 |
|---|---|---|---|---|
| `ext0.5\|tm30\|lgb_lean` | **0.528713** | 0.846840 | 0.528624 | 0.846699 |
| `ext0.75\|tm30\|fac_full` | **0.529078** | 0.846403 | 0.527689 | 0.846166 |
| `ext0.25\|tm30\|lgb_lean` | 0.528338 | 0.847085 | 0.527972 | 0.846947 |
| `ext0.5\|tm30\|fac_full` | 0.528889 | 0.846797 | 0.527099 | 0.846600 |

Resep v61 yang persis, yaitu `ext0.5|tm30|lgb_lean`, di v66 memberi KS 0.528713 melawan 0.528624. Naik 0.000089, kecil, tapi positif dan ini perbandingan apples-to-apples: model identik, partisi berbeda.

### 7. Kandidat

35 vektor, 32 lolos lantai AUC, 13 lolos deduplikasi korelasi rank 0.9995.

| Slot | Nama | AUC | KS | KS dikurangi 1 sd | Spearman ke v61 |
|---|---|---|---|---|---|
| 1 | `tm20` | **0.847312** | 0.526970 | 0.523370 | 0.9979 |
| 2 | `plateau\|tm20\|fac_full` | 0.847200 | 0.528463 | 0.524863 | 0.9988 |
| 3 | `trees4` | 0.847142 | 0.525693 | 0.522093 | 0.9949 |
| 4 | `ext0.5\|tm30\|lgb_lean` | 0.846840 | **0.528713** | **0.525113** | 0.9995 |
| 5 | `ext1.0\|tm20\|fac_full` | 0.846793 | **0.529873** | 0.526273 | 0.9986 |
| 6 | `leg\|xgb_full` | 0.846756 | 0.525536 | 0.521936 | 0.9945 |
| 7 | `leg\|lgb_lean` | 0.846721 | 0.525311 | 0.521711 | 0.9928 |
| 11 | `ext1.5\|tm20\|lgb_lean` | 0.846174 | 0.528163 | 0.524563 | 0.9983 |

Slot 5 punya KS OOF tertinggi di pool, 0.529873, tapi slot 4 punya KS dikurangi satu sampling sd tertinggi, 0.525113. Keduanya di atas 0.525 setelah pengurangan noise. Perbedaannya illustrates why KS mentah tidak pernah dipakai sebagai kunci.

Ambang deduplikasi dilonggarkan ke 0.9995, dan `ext0.5|tm30|lgb_lean` dipaksa masuk. Pada ambang 0.995 kandidat utama justru terbuang karena peringkatnya di bawah dua yang hampir identik pada AUC — ambang yang terlalu ketat menghapus kandidat yang justru ingin dikirim.

### 8. Status

Tiga belas file tervalidasi, 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di [0, 1] tanpa NaN. Jumlah file di disk sama persis dengan jumlah entri di `summary_v66_candidates.json`. Seluruh script diverifikasi ulang: 39 fine, 0 broken. Notebook `agas/v66/notebook/notebook_v66.ipynb` dieksekusi penuh dengan `RUN_TRAINING = False`, 16 cell, nol error.

### 9. Apa yang diketahui, dan apa yang tidak

**Diketahui.** Resep v61 bisa direproduksi tepat, jadi perbandingan v66 dengan v61 bersih. Tujuh kaki naik tanpa kecuali. Resep winner di v66 memberi KS 0.528713 melawan 0.528624, dan ada kandidat dengan KS dikurangi noise tertinggi di pool.

**Tidak diketahui.** Apakah itu masuk leaderboard. Yang jelas dari data proyek ini: OOF KS dan public score hampir tidak berkorelasi. Rekor KS OOF adalah v63 0.529867 dan public-nya 0.557; v61 0.528624 memberi 0.56312. Jadi kemajuan OOF di run ini **bukan** bukti kemajuan public.

**Kalau v66 masih di bawah 0.56312**, kesimpulannya bukan "lebih banyak fold seed tidak berguna", tapi "v61 adalah titik lokal yang sulit dikalahkan dengan perubahan arsitektur". Dan itu yang sebenarnya terjadi di v65: perubahan arsitektur, bukan tambahan training. v66 mengembalikannya ke arsitektur v61 persis dan hanya menambah training, jadi perbandingan akhirnya adil.

Belum ada submission Kaggle dan belum ada push GitHub.

## 2026-10-20 — v67: stacking pipeline, 0.56424 sebagai jangkar baru

### Pemicu

User melaporkan dua public score dari v66:
- `submission_v66_02_plateau_tm20_fac_full.csv`: **0.55877**
- `submission_v66_04_ext0.5_tm30_lgb_lean.csv`: **0.56424**

Jangkar public bergeser lagi ke `v66_04`. Arah `lgb_lean` sebagai kaki yang didorong menjauh terbukti lebih baik. Saran eksternal: stacking beberapa pipeline dan banyak model dalam satu submission.

### Metode

v67 memakai semua prediksi out-of-fold dari v61 sampai v66 sebagai fitur meta: 49 kolom, termasuk `tm20` dan `tm30` turunan v66. Meta-model: `LogisticRegression(C=0.5)` dan LightGBM dangkal, dilatih pada rank fitur. Estimasi jujur dengan 5-fold CV.

### Hasil

| Kandidat | AUC CV | KS CV |
|---|---|---|
| `meta_logit` | **0.848116** | 0.528510 |
| `consensus_meta_anchor` | 0.847753 | 0.529476 |
| `blend_anchor_meta_w0.50` | 0.847600 | 0.529414 |
| `blend_anchor_meta_w0.25` | 0.847267 | 0.528859 |
| `meta_lgb` | 0.847078 | 0.525840 |
| `anchor` (v66_04) | 0.846840 | 0.528713 |

### Status

Tujuh file tervalidasi, notebook 3 cell tanpa error. Tidak ada push GitHub. Public score v67 yang diketahui hanya `01_anchor` = **0.56424**; kandidat meta belum dilaporkan.

### Perbedaan submission v67

- `01_anchor`: vektor public terbaik sebelumnya (`v66_04`, ext0.5 tm30 lgb_lean), dipakai sebagai kontrol.
- `02/03/04_blend_anchor_meta_w0.10/0.25/0.50`: campuran anchor dan meta_logit dengan bobot meta 10%, 25%, 50%.
- `05_consensus_meta_anchor`: rata-rata rank meta_logit dan anchor (50/50).
- `06_meta_lgb`: meta-model LightGBM dangkal pada 49 fitur rank.
- `07_meta_logit`: meta-model logistik pada 49 fitur rank, AUC CV tertinggi.


## 2026-10-20 — v68: resep v61/v66 dengan sepuluh fold seed

### Pemicu

Public terbaik sekarang `submission_v66_04_ext0.5_tm30_lgb_lean.csv` dengan **0.56424**. User meminta model yang lebih kuat lagi dari v66 dengan cara yang sama seperti v66 dan v61, bukan arsitektur baru dan bukan stacking.

### Metode

v68 memakai tujuh kaki, himpunan kolom, parameter, faktor per fold, dan rantai blend yang identik dengan v61/v66. Yang ditambah hanya partisi: empat fold seed baru, `3, 17, 11, 23`, sehingga totalnya sepuluh.

Enam partisi v66 tidak dilatih ulang. `preds_v66.npz` menyimpan rata-rata enam seed, jadi rata-rata sepuluh seed adalah

```text
(6 * rata-rata_v66 + jumlah_prediksi_seed_baru) / 10
```

dan skrip menjumlahkan empat seed baru langsung. Ini ekuivalen matematis dengan melatih sepuluh seed dari nol, tetapi memakai 271 menit, bukan sekitar 632 menit. Partial disimpan per seed, jadi run bisa dilanjutkan tanpa mengulang seed yang selesai.

### Run

```bash
python3 agas/v68/training/train_submission_v68.py --folds 10 --fold-seeds 3,17,11,23 --lgb-seeds 0,1 --base-prefix v66 --base-seed-count 6 --out-prefix v68
python3 agas/v68/make_submission_v68.py
```

Durasi **271.5 menit** untuk empat seed baru. Peluncuran pertama dihentikan sebelum seed pertama selesai karena versi awal penulis partial menyimpan array OOF di kunci test; run dijalankan ulang dengan kunci OOF dan test terpisah. Tujuh kaki naik di AUC dan KS dibanding v66, tanpa kecuali.

| Kaki | AUC v68 | AUC v66 | Δ AUC | KS v68 | KS v66 | Δ KS |
|---|---|---|---|---|---|---|
| `fac_full` | 0.846760 | 0.846662 | +0.000098 | 0.524287 | 0.524205 | +0.000081 |
| `lgb_full` | 0.846834 | 0.846682 | +0.000153 | 0.525618 | 0.524974 | +0.000644 |
| `lgb_lean` | **0.846857** | 0.846721 | +0.000136 | **0.526881** | 0.525311 | **+0.001570** |
| `lgbx_full` | 0.846645 | 0.846550 | +0.000094 | 0.524779 | 0.524089 | +0.000690 |
| `cat_full` | 0.846523 | 0.846311 | +0.000212 | 0.525501 | 0.525234 | +0.000267 |
| `xgb_full` | 0.846891 | 0.846756 | +0.000135 | 0.525990 | 0.525536 | +0.000454 |
| `linear` | 0.842910 | 0.842771 | +0.000139 | 0.524114 | 0.522712 | +0.001401 |

### Rantai blend

| Rantai | KS v68 | KS v66 | AUC v68 | AUC v66 |
|---|---|---|---|---|
| `trees4` | 0.526196 | 0.525693 | 0.847283 | 0.847142 |
| `core` | 0.525385 | 0.525719 | 0.847189 | 0.847072 |
| `tm20` | **0.527124** | 0.526970 | **0.847427** | 0.847312 |
| `tm30` | 0.527661 | 0.528476 | 0.847334 | 0.847223 |
| `ext0.5\|tm30\|lgb_lean` | 0.528390 | 0.528713 | 0.846956 | 0.846840 |
| `ext0.5\|tm30\|fac_full` | 0.529355 | 0.528889 | 0.846928 | 0.846797 |
| `ext0.75\|tm20\|lgb_lean` | 0.528832 | 0.528286 | 0.847164 | 0.847053 |

Pola yang penting: semua kaki naik, tetapi `tm30` dan resep winner `ext0.5|tm30|lgb_lean` sedikit turun di KS OOF. AUC-nya naik. Ini bukan kontradiksi: `lgb_lean` membaik lebih cepat daripada `tm30`, sehingga jarak yang diekstrapolasi mengecil. Pada data ini, perbaikan kaki tidak otomatis memperbaiki KS blend.

### Kandidat

63 vektor, 61 lolos lantai AUC, 14 lolos deduplikasi korelasi rank 0.9995. `ext0.5|tm30|lgb_lean` dipaksa masuk.

| Slot | Nama | AUC | KS | KS dikurangi 1 sd | Spearman ke v66_04 |
|---|---|---|---|---|---|
| 1 | `tm20` | **0.847427** | 0.527124 | 0.523524 | 0.998256 |
| 2 | `plateau\|tm20\|fac_full` | 0.847328 | 0.528539 | 0.524939 | 0.999154 |
| 3 | `trees4` | 0.847283 | 0.526196 | 0.522596 | 0.995121 |
| 4 | `ext0.5\|tm30\|lgb_lean` | 0.846956 | 0.528390 | 0.524790 | **0.999997** |
| 5 | `ext1\|tm20\|fac_full` | 0.846950 | **0.529867** | **0.526267** | 0.998987 |

`v68_04` adalah versi sepuluh seed dari vektor yang mendapat public 0.56424. Korelasi rank terhadap `v66_04` adalah 0.999997, jadi public-nya kemungkinan hampir sama; ini kontrol yang aman, bukan arah baru. Kandidat yang paling berbeda dan punya KS OOF tertinggi adalah `v68_05_ext1_tm20_fac_full`.

### Status

Empat belas file tervalidasi: 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di [0, 1] tanpa NaN. Notebook `agas/v68/notebook/notebook_v68.ipynb` dieksekusi penuh dengan `RUN_TRAINING = False`, 13 cell, nol error. Perubahan v68 di-commit sebagai `0710fa7`; tidak ada push GitHub. Public `v68_04` = **0.56411**.

### Apa yang diketahui, dan apa yang tidak

**Diketahui.** Semua tujuh kaki membaik di AUC dan KS ketika partisi ditambah dari enam ke sepuluh. Ini konsisten dengan pengurangan varians, bukan perubahan kapasitas. Kandidat `v68_04` hampir identik dengan `v66_04`; kandidat `v68_05` adalah uji arah yang lebih berbeda dengan KS OOF tertinggi.

**Tidak diketahui.** Apakah kandidat v68 lain naik. `v68_04` sendiri sudah diukur: public **0.56411**, yaitu 0.00013 di bawah `v66_04`. Karena rank-nya berkorelasi 0.999997 dengan `v66_04`, hasil ini menunjukkan empat fold seed tambahan tidak memberi gain public yang terukur. Uji yang lebih informatif adalah `v68_05`, tetapi arah itu belum punya hasil public.

### Public feedback v67/v68

- `agas/v67/output/submission_v67_01_anchor.csv`: **0.56424**. Ini kontrol dan hasilnya sama dengan `v66_04`, jadi stacking v67 belum terbukti menambah nilai public.
- `agas/v68/output/submission_v68_04_ext0.5_tm30_lgb_lean.csv`: **0.56411**. Sepuluh fold seed tidak memperbaiki public secara terukur dibanding enam seed.
- Kandidat v67 selain anchor dan kandidat v68 selain slot 04 belum dilaporkan public-nya.


## 2026-10-21 — v69: factor features, `repayseq`, dan 15 fold

### Pemicu

Public `v67_01` tetap **0.56424** dan `v68_04` **0.56411**. Menambah fold seed dari enam ke sepuluh tidak mengubah vektor winner secara terukur (Spearman 0.999997). Arah berikutnya adalah fitur baru, fold lebih banyak, dan variasi `lgb_lean` yang lebih kaku.

### Temuan utama: factor features tidak pernah dilatih

v61-v68 membangun `factor_z`, `factor_sd`, dan `fs_*` per fold, tetapi tidak memasukkannya ke kolom training. Di v61, `fac_names` dibentuk di `agas/v61/training/train_submission_v61.py:157` lalu tidak dipakai; `full`, `lean`, dan `fac_cols` semuanya dibangun tanpa factor. v57/v58 memakai factor features; v61-v68 tidak. Ini menjelaskan kenapa gain factor yang terlihat di v57/v58 tidak muncul lagi.

v69 memasukkan enam kolom factor fold-safe ke kaki lean. Pada 5 fold seed 42, `base` tanpa factor **0.845398 / KS 0.523272**, `base` dengan factor **0.846035 / KS 0.524932**: **+0.000637 AUC / +0.001660 KS** pada himpunan fitur yang sama.

### Fitur baru

`agas/v69/training/feat_v69.py` membangun 122 fitur, semua dipotong pada `decision_datetime`:

- `cf_*` (`block_cashflow`, 52): level, CV, slope, drawdown, buffer, dan bulan negatif.
- `rq_*` (`block_repayseq`, 36): roll/cure/onset/worsen/improve, streak, delay, paid ratio, transisi pinjaman terakhir.
- `pv_*` (`block_prevseq`, 34): urutan aplikasi sebelumnya, approval, dan jarak hari.

`build_features_v69.py` menulis `v69_blocks.pkl` (100000, 123) dan `features_v69.pkl` (100000, 723). Cache berisi 720 fitur model + `application_id` + `customer_id` + `target`; 30.000 baris thin-file tidak punya riwayat dan tetap ikut dengan NaN. Bug `cured_before` diperbaiki sebelum build: status cure hanya dihitung dari transisi `late -> ontime`, bukan dari status awal.

### Screening

Semua baris 5 fold, fold seed 42, LGB seed 0, `jobs=4`:

| Run | AUC | KS | Lean | Factor |
|---|---:|---:|---:|---|
| `base_nofactor` | 0.845398 | 0.523272 | 490 | tidak |
| `base_factor` | 0.846035 | 0.524932 | 490 | ya |
| `all_blocks_factor` | 0.846204 | 0.525296 | 612 | ya |
| `cashflow_factor` | 0.845928 | 0.524118 | 542 | ya |
| `repayseq_factor` | 0.846140 | 0.524248 | 526 | ya |
| `prevseq_factor` | 0.845989 | 0.524229 | 524 | ya |
| `repayseq_extra_factor` | **0.846747** | 0.524847 | 526 | ya |
| `repayseq_stiff7_factor` | 0.846368 | **0.527178** | 526 | ya |

Factor adalah gain paling jelas; `repayseq` menambah sedikit; `cashflow` dan `prevseq` negatif. Hanya `repayseq` + factor yang dibawa ke run utama.

Sweep konfigurasi tanpa factor pada fitur lama juga dijalankan. Fold seed 42: `base` 0.845531, `l2_20` 0.845629, `stiff7` 0.845599, `wide31` 0.845208, `extra` 0.846127. Fold seed 7: `base` 0.845737, `l2_20` 0.845667, `stiff7` 0.845782, `wide31` 0.845445, `extra` 0.846422 (hanya LGB seed 0). Sweep dihentikan setelah dua fold seed untuk membebaskan CPU; tidak ada `preds_v69sweep.npz` atau `summary_v69sweep.json`.

### Run utama

```bash
python3 agas/v69/training/train_lean_v69.py \
  --features agas/v69/output/features_v69.pkl \
  --variants extra,stiff7 --folds 15 \
  --fold-seeds 42,7,13,21,5,101,3,17,11,23,77,55 \
  --lgb-seeds 0 --drop-prefixes cf_,pv_ --out-prefix v69
```

- 15 fold, dua varian, LGB seed 0, 532 kolom training (526 lean + 6 factor).
- Enam fold seed selesai (`42,7,13,21,5,101`) dalam **133.6 menit**. Proses dihentikan saat seed 3 mulai, bukan karena error, agar CPU bebas. Enam seed sisanya tidak dijalankan.
- `aggregate_partial_v69.py` merata-ratakan partial tanpa menunggu 12 seed. Partial menyimpan daftar 12 seed, jadi perintah yang sama bisa dilanjutkan; script akan melewati enam seed yang selesai.
- Log: `agas/v69/output/run_v69_main.out` dan `train_log_v69.txt`. Partial: `preds_v69_partial.npz`. Final: `preds_v69.npz`, `summary_v69.json` (`partial: true`).

### Hasil

| Seed | `extra` AUC | `extra` KS | `stiff7` AUC | `stiff7` KS |
|---:|---:|---:|---:|---:|
| 42 | 0.847640 | 0.528350 | 0.847116 | 0.527321 |
| 7 | 0.847866 | 0.528276 | 0.847382 | 0.527609 |
| 13 | 0.847836 | 0.527342 | 0.847418 | 0.526527 |
| 21 | 0.847715 | 0.526961 | 0.847366 | 0.528883 |
| 5 | 0.847618 | 0.528057 | 0.847190 | 0.529847 |
| 101 | 0.847814 | 0.527167 | 0.847442 | 0.528638 |
| **Bagged** | **0.848132** | **0.527736** | **0.847653** | **0.527933** |

`v68_lgb_lean` adalah 0.846857 / 0.526881. Bagged `extra` memberi **+0.001275 AUC / +0.000855 KS**, tetapi v68 memakai 10 fold dan 10 seed sedangkan v69 memakai 15 fold dan 6 seed. Yang bisa diklaim: enam seed v69 konsisten di atas v68, bukan bahwa 15 fold lebih baik dari 10 fold.

### Kandidat

`make_submission_v69.py` mengganti `lgb_lean` pada rantai v68, menambah ekstrapolasi, meta logistic, dan blend anchor. Sebelas file lolos deduplikasi rank 0.9995 dan lantai AUC. Kandidat teratas:

| Slot | File | AUC | KS | Spearman vs v66 |
|---:|---|---:|---:|---:|
| 01 | `submission_v69_01_leg_extra.csv` | **0.848132** | 0.527736 | 0.991086 |
| 02 | `submission_v69_02_stack_logit.csv` | 0.847997 | 0.527692 | 0.996453 |
| 03 | `submission_v69_03_stack_anchor_w0.1.csv` | 0.847913 | 0.528224 | 0.998454 |
| 04 | `submission_v69_04_trees4_extra.csv` | 0.847765 | 0.526741 | 0.995292 |
| 05 | `submission_v69_05_tm20_extra.csv` | 0.847674 | 0.527580 | 0.998347 |
| 06 | `submission_v69_06_leg_stiff7.csv` | 0.847653 | 0.527933 | 0.991371 |
| 10 | `submission_v69_10_ext0.5_tm30_extra.csv` | 0.846292 | 0.528254 | 0.999465 |

`leg|extra` adalah arah paling berbeda dengan AUC tertinggi. `ext0.5|tm30|extra` adalah resep public yang sudah terbukti, tetapi kini sangat dekat dengan anchor (0.999465) dan AUC-nya lebih rendah; ekstrapolasi melemahkan lean yang lebih kuat.

### Status

- Sebelas CSV tervalidasi: 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di [0, 1] tanpa NaN.
- Notebook `agas/v69/notebook/notebook_v69.ipynb` dieksekusi penuh dengan `RUN_TRAINING = False`, 13 cell, nol error.
- Tidak ada submission Kaggle dan tidak ada push GitHub; perubahan v69 di-commit bersama artefak utamanya.
- Public score semua kandidat v69 belum diketahui.

### Apa yang diketahui, dan apa yang tidak

**Diketahui.** Factor features yang sebelumnya tidak terpakai memberi gain AUC dan KS pada screening yang sama. `repayseq` adalah satu-satunya blok baru yang lolos; `cashflow` dan `prevseq` tidak. Enam seed v69 konsisten di atas v68, dan `extra` konsisten di atas `stiff7` di AUC.

**Tidak diketahui.** Apakah kenaikan OOF ini masuk public. Public `v68_04` hanya 0.00013 di bawah anchor, jadi perubahan 0.001 masih di dalam noise. `stack_logit` pernah gagal di public v67; `leg|extra` adalah arah baru yang belum pernah diuji. Enam seed bukan 12, dan 15 fold tidak otomatis lebih baik dari 10 fold.

### Artefak

- `agas/v69/training/feat_v69.py`, `build_features_v69.py`, `cv_v69.py`, `train_lean_v69.py`, `aggregate_partial_v69.py`
- `agas/v69/make_submission_v69.py`
- `agas/v69/notebook/notebook_v69.ipynb`, 13 cell, nol error
- `agas/v69/output/features_v69.pkl`, `v69_blocks.pkl`, `preds_v69.npz`, `summary_v69.json`, `summary_v69_candidates.json`
- `agas/v69/output/submission_v69_01_leg_extra.csv` sampai `submission_v69_11_ext0.75_tm30_stiff7.csv`
- `agas/v69/output/summary_v69_screening.json`, `candidates_v69.out`, `run_v69_main.out`

### Langkah berikutnya

1. Submit `submission_v69_01_leg_extra.csv` dan `submission_v69_04_trees4_extra.csv` sebagai dua arah baru.
2. Submit `submission_v69_02_stack_logit.csv` untuk menguji meta logistic dengan lean v69.
3. Submit `submission_v69_10_ext0.5_tm30_extra.csv` sebagai kontrol arah public 0.56424.
4. Catat public score di ledger.

## 2026-10-23 — v70: blok lintas sumber, 15 fold, 10 fold seed

### Titik awal

`v69_10` (`ext0.5|tm30|extra`) hanya mendapat **0.55989**; `v66_04` tetap **0.56424**. Pelajarannya: jangan mengganti pivot `lgb_lean` v66. v70 karena itu memakai ulang seluruh leg v66 dari `preds_v66.npz` dan leg v69 dari `preds_v69.npz`, lalu hanya menambah fitur dan training untuk leg baru. Tidak ada model lama yang dilatih ulang, sehingga arah public yang sudah terbukti tidak ikut berubah.

### Probe sumber dan grade

| Sumber | AUC 5-fold |
|---|---:|
| app + profile | 0.8308 |
| e-commerce | 0.7635 |
| repayment | 0.7462 |
| bureau | 0.6869 |
| previous applications | 0.6309 |
| account behavior | 0.6207 |
| semua 612 kolom | **0.8459** |
| rank blend 6 sumber | 0.8138 |

Rank blend per sumber jauh di bawah model gabungan, jadi informasi baru harus datang dari interaksi, bukan dari menambah model per sumber.

Grade probe: global 0.845949, per-grade 0.840175, hybrid `w=0.25` 0.845985 (+0.000036). Tidak dipakai.

### Blok fitur baru

`agas/v70/training/feat_v70.py` membangun empat blok:

| Prefix | Isi | Kolom |
|---|---|---:|
| `ms_` | panel bulanan repayment (due/late/unpaid) × cash flow (net, saldo, low-balance days) | 67 |
| `ls_` | status pinjaman: late/dpd per pinjaman, aktif vs lunas, rasio bayar | 40 |
| `ed_` | kedalaman e-commerce: keranjang, kategori, tren harga, konsentrasi | 51 |
| `ss_` | panel 3 arah: late × net cash flow × e-commerce spend | 42 |

Screening memakai 5 fold, fold seed 42/7/13, satu LGB seed, config `extra`, factor aktif:

| Run | Blok | n_lean | AUC | KS |
|---|---|---:|---:|---:|
| `s0_base` | v69 lean + factor | 526 | 0.847013 | 0.526286 |
| `s1_ms` | + `ms_` | 593 | 0.847135 | 0.527648 |
| `s2_ls` | + `ls_` | 566 | 0.847065 | 0.526527 |
| `s3_ed` | + `ed_` | 577 | 0.846887 | 0.526514 |
| `s4_all` | + `ms_` + `ls_` + `ed_` | 684 | **0.847219** | **0.527724** |
| `s5_ss` | + `ss_` | 568 | 0.846972 | 0.526633 |
| `s6_allss` | + all + `ss_` | 726 | 0.847023 | 0.526422 |

`ms_` satu-satunya blok yang konsisten positif (+0.000122 AUC, +0.001362 KS di ketiga seed). `ls_` netral, `ed_` negatif sendirian, tetapi `+all` sedikit di atas `+ms` (+0.000084 AUC) dengan sd lebih kecil. `ss_` negatif dan dibuang. Feature set final: `+all`, 684 kolom lean (690 dengan factor).

### Training

15 fold, fold seed `42,7,13,21,5,101,3,17,11,23`, LGB seed 0, dua varian:

| Fold seed | `extra` AUC | `extra` KS | `stiff7` AUC | `stiff7` KS |
|---:|---:|---:|---:|---:|
| 42 | 0.847821 | 0.529059 | 0.847238 | 0.527718 |
| 7 | 0.847880 | 0.528070 | 0.847631 | 0.528471 |
| 13 | 0.847907 | 0.527917 | 0.847593 | 0.527085 |
| 21 | 0.847941 | 0.526791 | 0.847500 | 0.526376 |
| 5 | 0.847984 | 0.528610 | 0.847247 | 0.529096 |
| 101 | 0.847899 | 0.528304 | 0.847393 | 0.526442 |
| 3 | 0.848124 | 0.527680 | 0.847844 | 0.528610 |
| 17 | 0.847923 | 0.527526 | 0.847252 | 0.526168 |
| 11 | 0.848083 | 0.527341 | 0.847466 | 0.526110 |
| 23 | 0.847945 | 0.527423 | 0.847547 | 0.529113 |
| **Bagged** | **0.848369** | **0.528503** | **0.847843** | **0.527944** |

Dibanding `v69_extra` (0.848132 / 0.527736), bagged `v70_extra` adalah **+0.000237 AUC / +0.000767 KS**. Semua 10 seed `extra` di atas 0.8478; `stiff7` tetap di bawah `extra` tetapi menambah diversity.

### Kandidat

`make_submission_v70.py` membangun 195 vektor, 143 lolos lantai AUC 0.0025, 21 lolos deduplikasi Spearman 0.9995:

| Slot | File | AUC | KS | Spearman vs v66_04 |
|---:|---|---:|---:|---:|
| 01 | `submission_v70_01_leg_v70_extra.csv` | **0.848369** | 0.528503 | 0.989875 |
| 02 | `submission_v70_02_leg_x69_extra.csv` | 0.848132 | 0.527736 | 0.991086 |
| 03 | `submission_v70_03_tm30w7_v70_extra.csv` | 0.847852 | 0.528663 | 0.998486 |
| 04 | `submission_v70_04_leg_v70_stiff7.csv` | 0.847843 | 0.527944 | 0.990402 |
| 05 | `submission_v70_05_leg_x69_stiff7.csv` | 0.847653 | 0.527933 | 0.991371 |
| 06 | `submission_v70_06_plateau_tm30w7_v70_extra_lgb_lean.csv` | 0.847623 | **0.530232** | 0.998421 |
| 07 | `submission_v70_07_ext0.5_tm30v70_lgb_lean.csv` | 0.847249 | 0.529397 | 0.999670 |
| 08 | `submission_v70_08_tm30v66.csv` | 0.847223 | 0.528476 | 0.999216 |
| 09 | `submission_v70_09_ext0.25_tm30w7_v70_stiff7_x69_extra.csv` | 0.847164 | 0.528747 | 0.999052 |
| 10 | `submission_v70_10_trees4.csv` | 0.847142 | 0.525693 | 0.995111 |

`plateau|tm30w7|v70_extra|lgb_lean` (slot 06) pada dasarnya adalah `ext0.5` dengan bobot 0.7 pada leg terkuat; rata-rata rasio 0.25/0.5/0.75 tidak menaikkan AUC dibanding `ext0.5` (0.847623 vs 0.847623) dan hanya menambah +0.000115 KS. Jadi plateau di sini bukan taruhan pada rasio agresif.

### Rekomendasi submit

1. **Slot 06** — KS OOF tertinggi, AUC 0.847623, Spearman 0.9984 terhadap anchor; keluarga ekstrapolasi yang sama, dengan core yang lebih kuat.
2. **Slot 07** — paling dekat dengan resep 0.56424 (Spearman 0.99967); pilihan konservatif.
3. **Slot 01** — AUC tertinggi (0.848369) tanpa ekstrapolasi; kontrol untuk menguji apakah model mentah yang lebih kuat juga menang di public.

Kalau hanya satu: **slot 06**. Kalau dua: **06 + 07**.

### Status

21 CSV tervalidasi: 30000 baris, ID unik dan cocok dengan `data/test_applications.csv`, probabilitas di [0, 1], tanpa NaN. Notebook `agas/v70/notebook/notebook_v70.ipynb` dieksekusi penuh dengan `RUN_TRAINING = False`, nol error. Tidak ada submission Kaggle dan tidak ada push GitHub.

## Public score ledger

| File | Public | Status |
|---|---:|---|
| `agas/v57/output/submission_v57_best.csv` | 0.55700 | varian `aggressive_145` |
| `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` | 0.55900 | replikasi `qc_00` |
| `agas/qc_00_ext1.5_v4_fac_full_only.csv` | 0.56171 | file tim |
| `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` | 0.56312 | rekor sebelumnya |
| `agas/v63/output/submission_v63f_consensus5_best.csv` | 0.55700 | kandidat prioritas 1 |
| `agas/v65/output/submission_v65_01_v65_consensus_ranks.csv` | 0.55900 | kandidat prioritas 1 |
| `agas/v66/output/submission_v66_02_plateau_tm20_fac_full.csv` | 0.55877 | plateau `tm20` |
| `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv` | **0.56424** | public terbaik |
| `agas/v67/output/submission_v67_01_anchor.csv` | **0.56424** | kontrol |
| `agas/v68/output/submission_v68_04_ext0.5_tm30_lgb_lean.csv` | 0.56411 | 10 fold seed |
| `agas/v69/output/submission_v69_10_ext0.5_tm30_extra.csv` | 0.55989 | pivot `extra` |
| `agas/v70/output/submission_v70_06_plateau_tm30w7_v70_extra_lgb_lean.csv` | 0.56040 | OOF KS tertinggi, public turun |
| `agas/v70/output/submission_v70_07_ext0.5_tm30v70_lgb_lean.csv` | 0.56419 | praktis seri dengan anchor |
| `agas/v70/output/submission_v70_01_leg_v70_extra.csv` | 0.55801 | OOF AUC tertinggi, public turun |

Kandidat v67 selain `01_anchor`, kandidat v68 selain `04`, dan kandidat v70 selain tiga di atas belum disubmit; public score-nya belum diketahui. Pelajaran dari v70: pada level ini OOF AUC/KS dan public tidak monoton; selisih OOF <0.001 adalah noise. Anchor `v66_04` tetap 0.56424.

## 2026-10-24 — v71: historical panel, rate residual, dan MLP diversity

### Titik awal

Public v70 sudah masuk: `v70_06` 0.56040, `v70_07` 0.56419, `v70_01` 0.55801.
Tidak ada yang melewati anchor `v66_04` 0.56424. OOF KS tertinggi (v70_06)
justru public terburuk kedua; OOF AUC tertinggi (v70_01) public paling rendah.
Kesimpulan yang dipakai: selisih OOF <0.001 tidak prediktif, dan feature
engineering keluarga v66–v70 sudah jenuh.

### Historical panel

`build_panel_v71.py` membangun panel 174.918 baris: 100.000 aplikasi sekarang
dan 74.918 pinjaman historis approved yang window 12 bulannya selesai sebelum
`decision_datetime`. Positive rate historis 16.50%. Ada 375 fitur `pn_` yang
semua dihitung pada cutoff masing-masing baris:

- terms pinjaman historis, previous applications, repayment, bureau, account
  behavior, dan e-commerce sebelum cut;
- umur/tenure di-backdate; income adalah snapshot sekarang dan dicatat sebagai
  keterbatasan;
- label adalah proksi `max_days_past_due >= 90` pada 12 angsuran pertama, bukan
  timestamp harian DPD 90. Ini aproksimasi, bukan label resmi.

Model panel: LightGBM 5-fold **GroupKFold by customer**, 2 konfigurasi × 3 seed.
Hasil historical OOF AUC **0.708283**, KS **0.296240**. Skor panel untuk target
aplikasi sekarang: AUC **0.756185**, KS **0.372977**.

Screening current model memakai 5 fold × 3 fold seed, config `extra`, factor
aktif:

| Run | n_lean | AUC | KS |
|---|---:|---:|---:|
| v70 baseline | 684 | **0.847219** | 0.527724 |
| panel score only | 771 | 0.847202 | 0.526964 |
| panel score + 68 `pn_` + `rf_` | 850 | 0.847199 | 0.526876 |

Panel **tidak dipakai** di model final. Selisihnya −0.000017 AUC, jauh di dalam
noise; menambah 68 fitur historis juga tidak menolong. Ini negative result yang
disimpan sebagai artefak, bukan fitur paksa.

### Auxiliary rate

Model multiclass memprediksi tier rate (16/20/24/28/32%) dari fitur selain rate
dan installment. Rate ternyata dapat diprediksi hampir sempurna dari fitur yang
sudah ada: AUC `rate >= 24%` = **1.0**, residual median absolut ~**2.7e-6**,
ekor maksimum ~0.066. Artinya rate adalah fungsi deterministik dari profil yang
sudah terukur, bukan sinyal baru. `rf_*` dibuang dari model final.

### MLP diversity

Feature set tetap 684 kolom lean v70 + factor fold-safe untuk tree. MLP memakai
771 kolom numerik (tanpa panel/rate), median imputation + clipping 0.1/99.9 +
standardisasi **di dalam fold**, arsitektur 256→64, BatchNorm, dropout, early
stopping. Final 15 fold × 3 fold seed × 3 seed:

| Leg | AUC | KS |
|---|---:|---:|
| `v70_extra` (LGBM) | 0.848369 | 0.528503 |
| MLP 256/64 | **0.848433** | **0.529473** |
| rank blend 60% MLP + 40% v70 | **0.850082** | **0.531973** |

Korelasi rank MLP vs v70 hanya ~0.973, sedangkan antar-LGBM ~0.99. Ini sumber
kenaikan: bukan MLP sendirian, tetapi error yang tidak sama. MLP wide 512/128
juga dilatih sebagai leg opsional; OOF-nya lebih rendah dari MLP 256/64, jadi
hanya boleh masuk dengan bobot kecil.

### Final blend dan kandidat

Final training 15 fold, tiga fold seed:

| Leg | AUC | KS | Catatan |
|---|---:|---:|---|
| MLP 256/64, 3 seed | 0.848433 | 0.529473 | leg utama |
| MLP wide 512/128, 1 seed | 0.846901 | 0.526979 | lebih lemah, hanya leg kecil |
| CatBoost | 0.846704 | 0.526520 | sd AUC 0.000109 |

Weight search pada OOF memilih `core71`:

```
0.60 × MLP-mix  +  0.10 × CatBoost  +  0.30 × v70
MLP-mix = 0.70 × MLP 256/64 + 0.30 × MLP wide
```

Jadi bobot efektif: MLP 256/64 0.42, MLP wide 0.18, CatBoost 0.10, v70 0.30.
AUC **0.850147**, KS **0.532834**. Kandidat KS-constrained `core71_ks`
memakai MLP-mix 0.55 (wide hanya 0.055), CatBoost 0.15, v70 0.30; AUC
0.850112, KS **0.532884**. Bedanya dengan core praktis nol, tetapi file tetap
disimpan sebagai kandidat KS.

Public-proxy bootstrap 600 × 30.000 subset stratifikasi:

| Kandidat | dKS | P(KS > anchor) | dAUC | P(AUC > anchor) |
|---|---:|---:|---:|---:|
| `core71` | +0.004244 | 0.965 | +0.003330 | 1.000 |
| `core71_ks` | +0.004243 | 0.967 | +0.003311 | 1.000 |
| `stack_lr` | +0.003755 | 0.952 | +0.003209 | 1.000 |
| `anchor0.25\|core71` | +0.001786 | 0.940 | +0.001505 | 1.000 |

Sensitivitas prevalence (10%, 13.58%, 20%, 25%) memberi `dKS` sekitar
+0.0040 sampai +0.0042 dan P(KS lebih baik) 0.93–0.997. Keunggulan core tidak
bergantung pada satu asumsi prevalence public.

Kandidat final yang disimpan (14 CSV, semuanya tervalidasi 30.000 baris, ID
unik/cocok test, [0,1], tanpa NaN):

| Slot | Nama | AUC | KS | P(KS) | File |
|---:|---|---:|---:|---:|---|
| 01 | `core71` | **0.850147** | 0.532834 | 0.965 | `submission_v71_01_core71.csv` |
| 02 | `core71_ks` | 0.850112 | **0.532884** | 0.967 | `submission_v71_02_core71_ks.csv` |
| 03 | `stack_lr` | 0.850049 | 0.531650 | 0.952 | `submission_v71_03_stack_lr.csv` |
| 04 | `anchor0.50\|mlp_base` | 0.849669 | 0.530838 | 0.945 | `submission_v71_04_anchor0.50_mlp_base.csv` |
| 07 | `anchor0.50\|core71` | 0.849395 | 0.530802 | 0.962 | `submission_v71_07_anchor0.50_core71.csv` |
| 11 | `anchor0.25\|core71` | 0.848340 | 0.530017 | 0.940 | `submission_v71_11_anchor0.25_core71.csv` |
| 14 | `anchor_v66_04` | 0.846840 | 0.528713 | — | `submission_v71_14_anchor_v66_04.csv` |

Sisanya adalah kontrol `tm30`/`ext` dan leg mentah; semuanya ada di
`summary_v71_candidates.json`.

### Ekspektasi public dan batas

Dengan pergeseran OOF yang sama seperti anchor, `core71` berada di sekitar
**0.568** public. Angka 0.577 berarti +0.009 di atas expected, sekitar 1.6 sd
public (sd historis ~0.0053). Jadi 0.577 masih mungkin secara varians, tetapi
bukan hasil yang bisa diklaim sebelum submit. Rekomendasi urutan submit:
**01 `core71`** (AUC tertinggi), **02 `core71_ks`** (KS OOF tertinggi), lalu
**11 `anchor0.25|core71`** (konservatif, rho 0.99919 terhadap anchor).

Tidak ada XGBoost final: kandidat yang ada sudah cukup kuat dan waktu lebih
baik dipakai untuk dokumentasi serta validasi. Tidak ada submit Kaggle dan
tidak ada push GitHub. Notebook `notebook_v71.ipynb` dieksekusi penuh dengan
`RUN_TRAINING=False`, 22 sel, nol error.
