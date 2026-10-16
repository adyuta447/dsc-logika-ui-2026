# README - Eksperimen Submission `agas/`

Dokumen ini menjelaskan langkah kerja untuk menghasilkan kandidat submission baru tanpa mengubah file lama di repo.

## 0) Struktur folder

Semua artefak dikelompokkan per versi. Setiap versi punya tiga folder dengan peran yang sama: `training/` berisi script, `notebook/` berisi notebook reproducible, `output/` berisi submission, prediksi, ringkasan, dan log.

```
agas/
  README.md                  dokumentasi metode per versi
  progress.md                jurnal kronologis semua eksperimen
  limit.md                   aturan kompetisi, tidak boleh push GitHub
  qc_00_ext1.5_v4_fac_full_only.csv
                             jangkar public terbaik sebelum v61, 0.56171
  shared/
    paths.py                 resolusi path untuk semua versi
    feat_extra.py            sebelas blok fitur baru
    build_extra.py           pembangun cache fitur per blok
    cv_harness.py            harness CV cepat untuk iterasi fitur
    factor2.py               factor model v2, diuji lalu ditolak
    cache/
      features_cache.pkl     frame fitur dari features.py
      extra_cache.pkl        gabungan blok fitur baru
      exblocks/              cache per blok
  v57/ v58ext/ v59/ v60/ v61/ v62/
    training/                script yang menghasilkan kandidat versi ini
    notebook/                notebook reproducible versi ini
    output/                  submission_*.csv, preds_*.npz, summary_*.json, log
```

Aturan path yang berlaku:

- `shared/paths.py` mengekspor `ROOT`, `AGAS`, `SHARED`, `CACHE`, `ANCHOR`, `version_output_of(nama)`, dan `version_output()`. Setiap script mengambil lokasi foldernya sendiri dari `Path(__file__)`, jadi tidak ada nama folder versi yang ditulis manual di dalam script.
- Script v62 membaca `preds_v61.npz` milik v61 secara eksplisit lewat `paths.version_output_of("v61")`, jadi ketergantungan lintas versi terlihat.
- Feature cache tinggal di `shared/cache/` karena dipakai hampir semua versi, sementara submission tidak pernah dipakai lintas versi kecuali sebagai jangkar.
- Script dijalankan dari root repo: `python3 agas/v62/training/train_submission_v62.py`.

## 1) Batasan yang dipatuhi

- Tidak push apa pun ke GitHub.
- Tidak mengubah file existing di luar `agas/`.
- Semua artefak baru diletakkan di `agas/`.
- Hanya memakai data kompetisi lokal di folder `data/`.

## 2) Metode yang dipakai

Script utama: `agas/v57/training/train_submission_v57.py`

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
python3 agas/v57/training/train_submission_v57.py --folds 5 --fold-seed 42 --lr 0.01 --lgb-seeds 0,1,2 --out-prefix v57
```

## 4) Struktur output

Semua hasil ada di `agas/shared/cache/` (sebelumnya `agas/output/`):

- `features_cache.pkl` - cache fitur gabungan train+test
- `agas/v57/output/train_log_v57.txt` - log training detail
- `agas/v57/output/summary_v57.json` - ringkasan metrik semua blend
- `submission_v57_*.csv` - kandidat submission
- `agas/v57/output/submission_v57_best.csv` - kandidat terbaik otomatis

## 5) Validasi file submission

Semua file `submission_v57_*.csv` sudah lolos cek:

- jumlah baris = 30000
- `application_id` unik
- set `application_id` sama persis dengan `data/test_applications.csv`
- `default_probability` dalam rentang `[0,1]`

## 6) Saran urutan submit ke Kaggle (run v57, SUDAH DIEVALUASI — lihat bagian 7)

1. ~~`agas/v57/output/submission_v57_best.csv` (varian `aggressive_145`)~~ — disubmit, hasil public **0.557**, lebih buruk dari baseline tim.
2. `agas/v57/output/submission_v57_top_mix.csv`
3. `agas/v57/output/submission_v57_meta_logit.csv`

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
python3 agas/v58ext/training/train_submission_v58_ext.py --fold-seeds 42,7,13 --folds 5 --lr 0.01 --lgb-seeds 0,1,2 --ext-ratios 1.0,1.5,2.0 --out-prefix v58ext
```

Perbedaan vs resep asli: 3 LightGBM seed per varian (bukan 1), dijalankan di 3 fold seed seperti resep asli, untuk ensemble dasar yang sedikit lebih stabil. Tidak mengubah `features.py` — kolom v4-only dibuang manual lewat daftar kolom di script ini saja.

### Hasil OOF (`agas/v58ext/output/summary_v58ext.json`)

| Varian | AUC | KS |
|---|---|---|
| `v4_full_only` | 0.847073 | 0.526265 |
| `fac_full_only` | 0.846160 | 0.525122 |
| `ext1.0` | 0.846846 | 0.527486 |
| **`ext1.5`** | 0.846298 | **0.527511** |
| `ext2.0` | 0.845463 | 0.526260 |

Arah `v4 - fac` positif konsisten di semua 3 fold seed (bukan kebetulan satu split). Rasio `1.5` tetap rasio terbaik di OOF, sejalan dengan temuan tim sebelumnya.

Validasi arah: korelasi Spearman antara `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` dan file lama tim `qc_00_ext1.5_v4_fac_full_only.csv` = **0.9994** — konfirmasi ini replikasi arah yang sama dengan model dasar dilatih ulang, bukan blend spekulatif baru.

### File output

- `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv` — **kandidat utama berikutnya**
- `agas/v58ext/output/submission_v58ext_ext1_v4_fac_full_only.csv` — cadangan konservatif
- `agas/v58ext/output/submission_v58ext_ext2_v4_fac_full_only.csv` — cadangan agresif
- `agas/v58ext/output/submission_v58ext_v4_full_only.csv`, `..._fac_full_only.csv` — dua kaki mentah, untuk diagnosis kalau ekstrapolasi meleset

Semua lolos validasi format (30000 baris, id unik dan cocok test, probabilitas di `[0,1]`).

### Saran submit berikutnya

1. `agas/v58ext/output/submission_v58ext_ext1.5_v4_fac_full_only.csv`
2. `agas/v58ext/output/submission_v58ext_ext1_v4_fac_full_only.csv` (kalau 1.5 ternyata overfit noise public lagi)

Jangan submit varian blend spekulatif baru (seperti `aggressive_145`) tanpa validasi arah ke public dulu.

---

## 8) Model alternatif v59 setelah v58ext mendapat public 0.559

Submission `v58ext_ext1.5_v4_fac_full_only.csv` mendapat public `0.559`, sehingga dua keluarga model baru diuji. Keduanya memakai data lokal, feature cache, seleksi fitur manual fold-safe, dan tidak memakai AutoML/AutoFE.

### A. ExtraTrees + HistGradientBoosting

Script:

```text
agas/v59/training/train_submission_v59_alternative.py
```

Command:

```bash
python3 agas/v59/training/train_submission_v59_alternative.py --folds 5 --seed 2026 --top-features 120 --out-prefix v59alt
```

OOF:

| Model | AUC | KS |
|---|---:|---:|
| ExtraTrees | 0.830321 | 0.499668 |
| HistGradientBoosting | 0.830995 | 0.499361 |
| Rank blend 65/35 | 0.832446 | 0.502018 |

Model ini terlalu lemah untuk menggantikan baseline.

### B. XGBoost Lossguide + CatBoost Lossguide

Script:

```text
agas/v59/training/train_submission_v59_boost.py
```

Command:

```bash
python3 agas/v59/training/train_submission_v59_boost.py --folds 5 --seed 2026 --top-features 180 --out-prefix v59boost
```

Perubahan utama: pertumbuhan pohon `lossguide`, bukan depth-wise biasa. Seleksi 180 fitur dilakukan di dalam setiap training fold.

OOF:

| Model | AUC | KS |
|---|---:|---:|
| XGBoost Lossguide | 0.842156 | 0.519982 |
| CatBoost Lossguide | 0.841099 | 0.516187 |
| Rank blend 55/45 | 0.842608 | 0.520637 |

Model ini lebih baik dari metode A, tetapi tetap di bawah v57/v58ext.

### Notebook reproducible

Notebook:

```text
agas/v59/notebook/notebook_v59_alternative.ipynb
```

Set `RUN_TRAINING = True` untuk menjalankan kedua script. Notebook membaca summary, menampilkan metrik, dan memvalidasi submission v59. JSON notebook dan semua code cell sudah lolos syntax validation.

### Keputusan v59

- Tidak memilih v59 sebagai submission utama.
- Tidak menghapus submission v59; artefak disimpan untuk audit dan eksperimen blend masa depan.
- Jangan menyamakan OOF KS dengan public score.
- Semua perubahan tetap hanya di `agas/`.

---

## 9) v60: konservatif di sekitar submission public terbaik

v58ext `ext1.5` mendapat public `0.559`. Model v59 juga tidak lebih baik pada OOF. Karena itu v60 mempertahankan ranking file `qc_00_ext1.5_v4_fac_full_only.csv`, yang sudah mendapat public `0.56171`, lalu menambahkan perturbasi kecil dari model lokal.

Script:

```text
agas/v60/training/make_submission_v60.py
```

Formula:

```text
rank((1 - w) * rank(qc_anchor) + w * rank(model_lokal))
```

Bobot perturbasi: `w=0.02` dan `w=0.05`. Blend 50/50 juga dibuat sebagai kandidat berisiko lebih tinggi.

Input model:

- v58ext `ext1.5`
- v58ext `ext1.0`
- v58 `v4_full_only`
- v59 XGBoost Lossguide

Kandidat utama:

1. `agas/v60/output/submission_v60_qc_98_v58_ext1_2_v58_v4.csv`
2. `agas/v60/output/submission_v60_qc_98_v58_ext15_2_v58_v4.csv`
3. `agas/v60/output/submission_v60_qc_95_v58_ext1_5_v58_v4.csv`
4. `agas/v60/output/submission_v60_mix_qc_v58_ext1.csv`
5. `agas/v60/output/submission_v60_anchor_qc.csv` sebagai kontrol.

Semua 10 file v60 valid dengan 30000 baris, ID test cocok, ID unik, dan probabilitas `[0,1]`. Spearman kandidat perturbasi 2% terhadap anchor sekitar `0.99999`; perturbasi 5% sekitar `0.99994` sampai `0.99999`.

Catatan: v60 adalah strategi paling aman berdasarkan bukti public yang tersedia, tetapi tidak menjamin skor `0.57`. Submit anchor sebagai kontrol jika ingin mengukur apakah perturbasi benar-benar membantu. Jangan memakai blend 50/50 sebelum varian kecil memberi bukti public.

---

## 10) v61: blok fitur baru, ensemble 10 fold, study blend (16 Okt 2026)

Run ini menambah sebelas blok fitur di luar `features.py`, lalu melatih ensemble
10 fold x 3 fold seed, lalu memilih rasio ekstrapolasi dengan uji ketahanan
bukan hanya OOF penuh.

### Cara menjalankan

```bash
# 1. cache fitur baru (per blok, agar blok berikutnya murah)
python3 agas/shared/build_extra.py bureau,repayment,previous,account,ecommerce,interactions,gradenorm,rxtra,bxtra,axtra,ecommerce2

# 2. ensemble utama (kira-kira 3 jam 15 menit di 10 core)
python3 agas/v61/training/train_submission_v61.py --folds 10 --fold-seeds 42,7,13 --lgb-seeds 0,1,2 --lr 0.01 --l2 5 --out-prefix v61

# 3. study blend dan file kandidat
python3 agas/v61/training/study_blend_v61.py
python3 agas/v61/training/make_submission_v61_final.py
```

`RUN_TRAINING = False` di `agas/v61/notebook/notebook_v61.ipynb` membuat notebook hanya
membaca cache sehingga bisa diverifikasi tanpa menghitung ulang.

### Blok fitur baru

`full` 542 kolom, `lean` 434 kolom (buang kolom noise), `fac` 497 kolom
(buang blok putaran kedua `r2_`, `b2_`, `a2_`, `e2_`).

| Blok | Prefix | Fokus |
|---|---|---|
| `bureau` | `bx_` | jumlah provider, usia laporan terbawah, lag provider, ketidaksepakatan antar provider, tren 180 hari |
| `repayment` | `rx_` | recency telat, late dan DPD berbobot half-life, curing rate cicilan, rate per loan |
| `previous` | `px_` | recency penolakan, velocity pengajuan, jendela penolakan 90 hari, keteraturan jarak |
| `account` | `ax_` | CV pemasukan dan arus kas, streak tanpa pemasukan, bulan saldo di bawah angsuran |
| `ecommerce` | `ex_` | entropy kategori, item per order, diskresi versus kebutuhan, `lrel` berbobot |
| `interactions` | `ix_` | `interest_rate` terhadap DTI, pendapatan, jenis kerja, sewa, riwayat telat |
| `rxtra` | `r2_` | sejarah affordability, rate lunas tepat waktu |
| `bxtra` | `b2_` | kedalaman file biro, slope score, bulan terlcover |
| `axtra` | `a2_` | CV dispose bulanan, slope saldo, pengeluaran melebihi pemasukan |
| `ecommerce2` | `e2_` | rentang aktif, order per bulan, pengulangan kategori |
| `gradenorm` | `gn_` | z-score dalam grade dan kelompok; **diuji dan dibuang**, merusak KS |

### Hasil OOF model dasar (rata-rata tiga fold seed)

| Model | AUC | KS |
|---|---:|---:|
| `lgb_full` | 0.846647 | 0.524615 |
| `lgb_lean` | 0.846566 | 0.523214 |
| `lgbx_full` | 0.846351 | 0.524416 |
| `xgb_full` | 0.846485 | 0.525240 |
| `cat_full` | 0.845928 | 0.525236 |
| `linear` | 0.842667 | 0.522833 |
| `fac_full` | 0.846542 | 0.523080 |

### Hasil blend

| Blend | Rasio | AUC | KS OOF |
|---|---:|---:|---:|
| `topmix30 - lgb_lean` | 0.50 | 0.846699 | **0.528624** |
| `ext topmix20 - fac_full` | 0.75 | 0.846780 | 0.527968 |
| `consensus` | — | 0.846775 | 0.527879 |
| `ext topmix20 - fac_full` | 1.50 | 0.845662 | 0.527838 |
| `ext topmix20 - fac_full` | 1.00 | 0.846487 | 0.527660 |
| `topmix20` | — | 0.847177 | 0.525791 |
| `meta_logit` | — | 0.847157 | 0.525870 |

KS OOF tertinggi run ini **0.528624**. Nilai `0.55` tidak tercapai; analisis
batas di `PROGRESS.md` memperkirakan AUC maksimum teoretis sekitar 0.869
sedangkan model sekarang 0.847, dan noise public 0.0015 sampai 0.003 membuat
selisih OOF di bawah 0.002 tidak dapat dibedakan dari noise.

### Kenapa rasio dipilih dengan cara begini

KS di 70.000 baris OOF punya galat standar sekitar 0.002 sampai 0.003.
`study_blend_v61.py` karena itu menilai 64 kombinasi pada OOF penuh dan pada
enam sampel acak setengah OOF. Keluarga `full - fac` membentuk plateaux lebar
antara rasio 0.75 sampai 1.5, jadi kandidat utama memakai rata rank beberapa
rasio (`consensus`) alih-alih satu rasio terpilih.

### Kandidat yang direkomendasikan

1. `agas/v61/output/submission_v61_consensus.csv`
2. `agas/v61/output/submission_v61_ext1.5_full_fac.csv`
3. `agas/v61/output/submission_v61_ext0.75_full_fac.csv`
4. `agas/v61/output/submission_v61_meta_logit.csv` (kontrol, AUC tertinggi)
5. `agas/v61/output/submission_v61_anchor15_consensus.csv` (opsional paling aman)

Semua file `submission_v61_*.csv` lolos validasi: 30000 baris, ID unik dan
sama dengan test, probabilitas di `[0,1]`.

---

## 11) v62: gradien kaya-fitur, feature selection, dan base gabungan

Dipicu oleh hasil public baru: `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` mendapat **0.56312**, naik dari 0.56171. Arah itu memakai mekanisme yang sama dengan resep pertama, yaitu mendorong model berfitur lebih kaya menjauh dari model sejenis berfitur lebih sedikit. v62 memperpanjang gradien itu dan menyisakan setiap tingkatnya.

### Tangga kaya-fitur

| Tingkat | Kolom | Isi |
|---|---:|---|
| `full` | 554 | semua fitur, termasuk blok `cx_` baru |
| `fac` | 509 | tanpa blok putaran dua `r2_`, `b2_`, `a2_`, `e2_` |
| `lean` | 446 | tanpa kolom noise |
| `faclean` | 401 | tanpa noise dan tanpa blok putaran dua, tingkat terkurus |
| `top` | 200 | terpilih gain, per fold |
| `topwide` | 350 | terpilih gain, lebih longgar |

### Blok fitur `consistency` (putaran tiga)

Selisih antar sumber: utang bureau versus utang yang terlihat, jumlah akun bureau versus jumlah kredit yang kita ketahui, delinquency bureau versus riwayat cicilan internal, usia file bureau versus usia riwayat pengajuan, dan angsuran baru versus angsuran terbesar yang pernah dipegang. Fitur turunan dari tabel mentah dan stempel waktu keputusan saja.

Nilai rata-rata tiga fold seed: `+0.00059 KS` pada `lr 0.02 / lambda_l2 20` dan `+0.00097 KS` pada `lr 0.01 / lambda_l2 5`. Konsisten positif, jadi dipakai.

### Cara menjalankan

```bash
# blok fitur baru
python3 agas/shared/build_extra.py consistency
python3 agas/shared/build_extra.py bureau,repayment,previous,account,ecommerce,interactions,gradenorm,rxtra,bxtra,axtra,ecommerce2,consistency

# ensemble utama, sekitar 4 jam
python3 agas/v62/training/train_submission_v62.py --folds 10 --fold-seeds 42,7,13,21 --lr 0.01 --l2 5 --full-seeds 0,1 --top-n 200 --top-wide 350 --out-prefix v62

# pencarian arah dan kandidat
python3 agas/v62/training/search_gradient_v62.py
python3 agas/v62/training/search_combined.py
python3 agas/v62/training/make_submission_v62_final.py
```

### OOF model dasar (rata-rata empat fold seed)

| Model | AUC | KS |
|---|---:|---:|
| `lgb_top` (200 fitur) | **0.847419** | 0.525550 |
| `lgb_topwide` (350 fitur) | 0.846663 | 0.524101 |
| `lgb_lean` | 0.846645 | 0.524386 |
| `lgb_full` | 0.846629 | 0.523196 |
| `xgb_full` | 0.846639 | 0.524982 |
| `lgb_fac` | 0.846392 | 0.523101 |
| `lgb_faclean` | 0.846444 | 0.523102 |
| `lgbx_full` | 0.846305 | 0.523064 |
| `cat_full` | 0.846046 | 0.524848 |
| `linear` | 0.842723 | 0.521687 |

Temuan: feature selection berbasis gain di 200 kolom memberi model terkuat sepanjang masa ini, AUC 0.847419 melawan 0.846629 untuk 554 kolom pada pipeline yang sama.

### Plateau KS OOF

| Run | KS OOF terbaik | AUC OOF terbaik |
|---|---:|---:|
| v57 | 0.526988 | 0.846782 |
| v58ext | 0.527511 | 0.847073 |
| v61 | 0.528624 | 0.847177 |
| v62 | 0.527846 | 0.847419 |
| v61 + v62 | **0.528717** | 0.847128 |

KS OOF mentok di sekitar 0.528. Selisih di bawah 0.002 tidak dapat dibedakan dari noise, karena galat standar KS di 70.000 baris OOF sekitar 0.002 sampai 0.003.

### Protokol pemilihan arah

v62 menyimpan OOF per fold seed, jadi setiap arah diuji di tiap seed. Kandidat hanya direkomendasikan kalau KS membaik di sedikitnya tiga dari empat seed. Untuk base gabungan v61+v62, per-seed tidak tersedia sehingga dipakai resampling baris: KS dihitung ulang pada delapan sampel acak setengah OOF.

Dua koreksi metodologis di run ini:

1. Kaki reduksi harus dikeluarkan dari base blend, kalau tidak sumbu eksplorasi kabur.
2. Base blend harus dibangun ulang untuk setiap pasangan base-kaki.

Dari 675 kombinasi hanya 58 yang lolos uji ketahanan, dan baris dengan KS tertinggi hampir selalu bukan yang paling ketahanan. Itu persis yang diharapkan kalau OOF dipakai berlebihan.

### Kandidat final

KS OOF tertinggi sesi ini `0.528717` pada dua sumbu sekaligus di atas base gabungan. Peningkatan dari v61 hanya 0.00009, jadi praktis sama; yang berubah nyata adalah kualitas base model.

Urutan submit yang disarankan, dengan jangkar `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` (public 0.56312):

1. `agas/v62/output/submission_v62f_two_w0.2_fac0.5_fac61_0.5.csv` KS 0.528251, Spearman 0.998996
2. `agas/v62/output/submission_v62f_mixB_facv61_r05.csv` KS 0.528198, Spearman 0.998954
3. `agas/v62/output/submission_v62f_v61_facv62_r05.csv` KS 0.527993, Spearman 0.998242
4. `agas/v62/output/submission_v62f_two_w0.3_fac0.25_fac61_0.5.csv` KS 0.528717, Spearman 0.999653
5. `agas/v62/output/submission_v62f_consensus.csv` KS 0.527285, Spearman 0.999564
6. `agas/v62/output/submission_v62f_anchor50_consensus.csv` Spearman 0.999891

Tiga pertama dipilih karena KS-nya tinggi sekaligus cukup berbeda dari jangkar. Nomor 4 punya KS tertinggi tapi praktis identik dengan jangkar, jadi nilai informasinya rendah.

Semua 21 file `submission_v62f_*.csv` lolos validasi: 30000 baris, ID unik dan sama dengan test, probabilitas di [0,1].
