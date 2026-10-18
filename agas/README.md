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

---

## 12) v63: feature selection sebagai lever, KS OOF 0.529867

Target run ini KS OOF di atas 0.5299, naik dari rekor 0.528717 milik v62.

### Cara menjalankan

```bash
# sweep level seleksi dan hyperparameter, sekitar 75 menit
python3 agas/v63/sweep_v63.py

# ensemble utama, sekitar 3 jam 50 menit
python3 agas/v63/training/train_submission_v63.py --folds 10 --fold-seeds 42,7,13,21 --lr 0.01 --l2 5 --sel200-seeds 0,1 --out-prefix v63

# pencarian arah lintas run, verifikasi, dan kandidat final
python3 agas/v63/training/search_multirun_v63.py
python3 agas/v63/training/verify_v63_consensus.py
python3 agas/v63/training/final_v63.py
```

### Temuan sweep

Dua belas konfigurasi, rata-rata tiga fold seed `42`, `7`, `13`.

| Konfigurasi | AUC | KS | sd antar seed |
|---|---|---|---|
| tanpa seleksi | 0.846102 | 0.524794 | 0.00064 |
| **top 200, `feature_fraction` 0.3** | 0.846141 | **0.527077** | **0.00017** |
| top 200, `feature_fraction` 0.6 | 0.845785 | 0.527346 | 0.00117 |
| top 300 | 0.846150 | 0.525802 | 0.00036 |
| top 400 | 0.846195 | 0.525898 | 0.00037 |
| top 100 | 0.845127 | 0.523984 | 0.00054 |

Seleksi fitur menambah 0.0023 KS secara konsisten dan punya sebaran antar seed terkecil. Hipotesis bahwa `feature_fraction` perlu dinaikkan untuk set terseleksi ternyata salah: menaikkan ke 0.6 menaikkan KS 0.0003 tetapi menurunkan AUC dan memperbesar volatilitas tujuh kali.

### OOF model dasar (rata-rata empat fold seed)

| Model | AUC | KS |
|---|---|---|
| `sel200` | 0.846981 | 0.526330 |
| `sel200b` | 0.846783 | 0.525837 |
| `sel300` | 0.846793 | 0.524864 |
| `sel400` | 0.846730 | 0.524526 |
| `xgb_full` | 0.846639 | 0.524982 |
| `lgb_full` | 0.846502 | 0.523734 |
| `lgb_faclean` | 0.846444 | 0.523102 |
| `sel100` | 0.845208 | 0.523120 |
| `lgbx_full` | 0.846305 | 0.523064 |
| `cat_full` | 0.846046 | 0.524848 |
| `linear` | 0.842762 | 0.520747 |

### Dua sumbu dan base lintas run

Rekor v62 datang dari dua sumbu ekstrapolasi sekaligus. Pola yang sama diterapkan lagi, ditambah:

- **Ketajaman seleksi sebagai sumbu baru.** Kedua arah yang terbukti di leaderboard mendorong model berfitur lebih kaya menjauh dari model berfitur lebih sedikit. Ketajaman seleksi adalah ukuran lain dari quantity yang sama, jadi `sel200` didorong menjauh dari `sel400`.
- **Base lintas run.** v62 dan v63 memakai empat fold seed yang sama sehingga prediksinya bisa dirata-ratakan sebelum diekstrapolasi, dan karena keduanya menyimpan OOF per fold seed, ketahanan dapat diuji di dalam tiap seed tanpa resampling baris.

Dari 1296 baris grid, 557 atau 43 persen robust, naik dari 8,6 persen di v62.

### Consensus dan verifikasi

| Ukuran | KS OOF | AUC | KS per seed |
|---|---:|---:|---|
| **5** | **0.529867** | 0.847204 | 0.52697, 0.52799, 0.52625, 0.52763 |
| 6 | 0.529827 | 0.847263 | 0.52779, 0.52806, 0.52618, 0.52743 |
| 12 | 0.529620 | 0.847289 | 0.52812, 0.52824, 0.52640, 0.52768 |
| 9 | 0.529375 | 0.847289 | 0.52806, 0.52831, 0.52662, 0.52768 |

`verify_v63_consensus.py` membangun ulang consensus dari kode yang sama dan menghasilkan korelasi rank **1.00000000** terhadap file yang ditulis, sehingga angka laporan berasal dari file submission. Selisih consensus terhadap base tanpa ekstrapolasi positif di 3 dari 4 fold seed, mean +0.00089.

### Batas yang jujur

`0.529867` masih `0.00013` di bawah `0.530000`. Pada 70.000 baris OOF galat standar KS sekitar 0.002 sampai 0.003, jadi selisih sebesar itu tidak dapat dibedakan dari noise. Angka ini membulatkan ke 0.53 tetapi tidak melampaui 0.53.

Yang membaik dan bukan noise: base model naik jelas, arah yang konsisten antar seed naik dari 8,6 persen ke 43 persen, dan korelasi kandidat terhadap file public 0.56312 turun ke sekitar 0.995 dari 0.9996, jadi kandidat ini benar-benar menguji hipotesis baru.

### Kandidat final

| Prioritas | Kandidat | KS OOF | Korelasi vs jangkar |
|---|---|---:|---:|
| 1 | `submission_v63f_consensus5_best.csv` | 0.529867 | 0.995281 |
| 2 | `submission_v63f_best_single.csv` | 0.529767 | 0.994953 |
| 3 | `submission_v63f_consensus12.csv` | 0.529620 | 0.995430 |
| 4 | `submission_v63d_cand02.csv` s.d. `cand05.csv` | 0.529591 s.d. 0.529267 | 0.993 s.d. 0.996 |
| 5 | `submission_v63f_anchor35_consensus.csv` | — | 0.999422 |

Jangkar tetap `submission_v61_alt_ext0.5_tm30_lean.csv` dengan public 0.56312. Semua 20 file submission v63 lolos validasi: 30000 baris, ID unik dan sama dengan test, probabilitas di [0,1].

---

## 13) v64: audit korelasi, diagnosis noise, dan perubahan objektif

 dipicu hasil public v63 yang hanya 0.557 padahal KS OOF-nya 0.529867, tertinggi
sepanjang proyek.

### Apa yang sebenarnya terjadi

Empat file punya kedua angka terukur:

| File | KS OOF | Public |
|---|---|---|
| v57 `aggressive_145` | 0.526988 | 0.55700 |
| v58ext `ext1.5` | 0.527511 | 0.55900 |
| v61 `alt_ext0.5_tm30_lean` | 0.528624 | **0.56312** |
| v63 kandidat terbaik | **0.529867** | 0.55700 |

Pearson r pada empat pasangan itu +0.019. Pair yang menentukan: `v58ext ext1.5`
dan `qc_00` memiliki vektor rank dengan Spearman 0.9994, hampir identik, tapi
skor public-nya berbeda 0.0027.

### Kenapa KS OOF tidak bisa dipakai sebagai kriteria

KS adalah maksimum atas ambang dari jumlah kumulatif, jadi noisy. Diukur dengan
mengambil ulang baris dari satu vektor prediksi yang tetap:

| Ukuran sampel | SD sampling KS |
|---|---|
| 70000 baris (ukuran OOF) | 0.0036 |
| 30000 baris (ukuran public) | **0.0053** |
| 9000 baris (mesin harvest lama) | 0.0113 |

Akibatnya:

- Peningkatan v61 ke v63 sebesar 0.001243 adalah **0.34 sampling sd**.
- Memilih kandidat dari 1300 titik grid dengan KS OOF maksimum memberi kenaikan
  terukur sekitar **0.012 secara gratis**, sepuluh kali dari yang dikejar.
- Jarak ke 0.57 adalah **1.30 public sd**.

### Objektif baru

v64 memakai AUC, bukan KS maksimum. AUC adalah rata-rata halus atas seluruh
pasangan, bukan ambang terburuk, sehingga ia menjawab perbaikan model sungguhan.
KS tetap dilaporkan dengan SD sampling-nya, tidak pernah sebagai kunci pengurutan.

### Hasil negatif yang dicatat

Seleksi stabil diuji dan ditolak sebelum run utama:

| Seleksi, 10 fold, seed sama | AUC | KS |
|---|---|---|
| satu probe, top 200 | 0.845506 | 0.522678 |
| stabil, 5 probe, top 200 | 0.845148 | 0.522046 |
| tanpa seleksi, 554 kolom | 0.845951 | 0.523256 |

Seleksi stabil turun 0.000358, dan kedua himpunan tumpang tindih 169 dari 200
kolom, jadi probe tambahan sebagian besar sepakat tanpa menambah informasi.

### Cara menjalankan

```bash
# diagnosis, sekitar 5 menit
python3 agas/v64/audit_public_vs_oof.py
python3 agas/v64/diagnose_selection_noise.py

# uji seleksi stabil yang ditolak, sekitar 13 menit
python3 agas/v64/check_stability_selection.py

# ensemble utama, enam fold seed
python3 agas/v64/training/train_submission_v64.py --folds 10 --fold-seeds 42,7,13,21,5,101 --lr 0.01 --l2 5 --top-n 200 --out-prefix v64

# kandidat berbasis keragaman
python3 agas/v64/make_submission_v64.py
```

### Perbaikan alat

`shared/bootstrap.py` ditambahkan karena versi folder ada dua kedalaman, sehingga
indeks `parents[n]` yang dipatok salah untuk salah satunya. Enam perbaikan
berturutan diperlukan, dan setiap tahap ditemukan lewat verifikasi. Script yang
rusak dipulihkan, lalu `verify_all_scripts.py` menjalankan seluruh script untuk
memastikan `import paths` benar-benar berhasil. Angka metrik v61 dan v62
diverifikasi ulang setelah perbaikan dan tidak berubah.

### Prinsip pemilihan kandidat

Builder v64 mengumpulkan semua vektor prediksi dari v61, v62, v63, v64,
membuang yang korelasinya di atas 0.998, lalu mengurutkan dengan AUC. Dua file
yang nyaris identik memberi leaderboard draw yang sama, jadi keragaman memberi
nilai sementara duplikasi tidak.

### Hasil run utama, enam fold seed

Selesai dalam 340.7 menit, 10 fold, seed `42,7,13,21,5,101`:

| Model | AUC | KS |
|---|---|---|
| `xgb_full` | **0.846749** | 0.525485 |
| `lgb_full` | 0.846705 | 0.524327 |
| `sel400` | 0.846607 | 0.524095 |
| `lgb_faclean` | 0.846528 | 0.524096 |
| `lgbx_full` | 0.846477 | 0.524037 |
| `sel200` | 0.846316 | 0.525290 |
| `cat_full` | 0.846193 | 0.525368 |
| `linear` | 0.842875 | 0.522065 |

Dua hal yang perlu dicatat.

`sel400` mengalahkan `sel200`, sedangkan di v63 urutannya terbalik. Kunci pengurutan diganti ke AUC, dan KS menghargai prediksi yang lebih ekstrem, sementara `sel400` lebih baik rata-ratanya. Kalau masih memakai KS sebagai kunci, pilihan ini akan terbalik dan salah.

KS `sel200` turun dari 0.529867 di v63 ke 0.525290 di v64 sementara AUC-nya naik. Dua seed tambahan membuat prediksi lebih halus, dan prediksi yang lebih halus menaikkan AUC tapi menurunkan KS. Bukti kedua bahwa KS dan AUC mengukur hal berbeda.

### Pool kandidat akhir

88 vektor setelah blend, 79 lolos lantai AUC, 7 lolos deduplikasi korelasi rank 0.998:

| Slot | Nama | AUC | KS dikurangi 1 sd | Spearman ke anchor |
|---|---|---|---|---|
| 1 | `ext0.25|v62|lgb_top-v62|lgb_faclean` | 0.847485 | 0.522275 | 0.9913 |
| 2 | `avg_sel_lean` | 0.847043 | 0.522881 | 0.9918 |
| 3 | `v64|xgb_full` | 0.846749 | 0.521876 | 0.9946 |
| 4 | `ext0.5|v63|sel200-v62|lgb_fac` | 0.846461 | 0.522349 | 0.9866 |
| 5 | `ext0.75|v61|lgb_full-v61|lgb_lean` | 0.846210 | 0.520026 | 0.9929 |
| 6 | `v64|cat_full` | 0.846193 | 0.521759 | 0.9905 |
| 7 | `v63|sel100` | 0.845208 | 0.519511 | 0.9841 |

Kolom KS dikurangi satu sampling sd sengaja ditampilkan. Begitu dikurangi, kandidat 2 tidak lagi yang terbaik dan urutannya tidak lagi sama dengan urutan KS mentah. Itulah alasan angka KS tidak pernah dipakai sebagai kunci.

Nilai maksimum harapan atas 7 submit diverse adalah **+0.0072 public (1.35 sd)**, hanya batas atas dengan asumsi draw independen. Kandidat terpilih saling berkorelasi 0.984 sampai 0.995, jadi gain sesungguhnya lebih kecil dan tidak dapat dihitung: butuh korelasi noise public itu sendiri, yang butuh beberapa pasangan ranking independen, sedangkan proyek hanya punya satu pasangan.

### Dua bug yang ditemukan lewat verifikasi

**Slot output bertabrakan.** Tiga kali builder dijalankan tanpa menghapus file lama, nomor slot dipakai ulang dengan konten berbeda. Slot 3 pernah berisi `v61|lgb_full` dari run pertama dan `v64|xgb_full` dari run ketiga, keduanya valid dan keduanya mengklaim slot 3. Builder sekarang menghapus prefix-nya sendiri sebelum menulis.

**Lantai AUC tidak ada di deduplikasi pertama.** Deduplikasi murni korelasi rank menyisakan 9 kandidat, tapi ekornya berisi `linear` AUC 0.8427 yang bertahan karena meranking berbeda, sementara blend terkuat saling dianggap duplikat. Itu trade buruk, karena semua kandidat punya sebaran public yang sama sehingga kandidat dengan rata-rata lebih buruk hanya bisa menurunkan nilai maksimum harapan. Lantai AUC 0.0025 di bawah terbaik menutupnya.

### Status

Tujuh file submission tervalidasi: 30000 baris, `application_id` unik dan cocok dengan `data/test_applications.csv`, probabilitas di rentang [0, 1] tanpa NaN. Seluruh script diverifikasi ulang, 32 fine, 0 broken. Notebook `notebook/notebook_v64.ipynb` ada dengan `RUN_TRAINING = False`. Belum ada submission Kaggle dan belum ada push GitHub.

---

## 14) v65: membidik selisih yang tidak bisa ditutup feature engineering

### Pemicu, dan koreksi atas kesimpulan v64

Leaderboard dibaca user: puncak **0.57768**, lalu 0.5753, lalu 0.5717. Kita di **0.56312**, peringkat 29.

Ini mengoreksi kesimpulan v64. Di sana saya menulis bahwa 0.57 mungkin tidak terjangkau, karena selisih OOF di bawah 0.002 tidak bisa dibedakan dari noise. Argumen itu benar untuk mengukur gain, tapi salah untuk menilai posisi relatif, karena ia memperlakukan skor public kita sendiri sebagai satu undian noise lalu menyimpulkan batasnya. Leaderboard menunjukkan 0.577 ada.

Selisih 0.57768 − 0.56312 = **0.0146**. Sebaran sampling KS di 30000 baris public, yang diukur di v64, adalah **0.0053**. Jadi gap itu **2.75 sebaran**: model di puncak benar-benar lebih baik. Run ini berhenti mengejar 0.002 KS OOF dan mengarahkan energi ke kualitas model yang lebih besar.

### Blok fitur baru, dua diterima satu ditolak

Tiap blok diuji dengan tiga fold seed, karena satu fold seed bergerak sekitar 0.0006 AUC, ukuran yang sama dengan efek yang seharusnya diceap blok mana pun.

| Blok | Isi | Hasil |
|---|---|---|
| `dataquality` (`dq_`) | kedalaman cakupan, fraksi missingness per tabel, pola cakupan bersama | **diterima**, +0.000027 AUC, +0.000190 KS, sebaran antar seed turun 0.000304 ke 0.000110 |
| `stress` (`af_`) | beban cicilan terhadap arus kas terukur dan cicilan aktif, bukan income dideklarasikan | **diterima**, nyaris netral |
| `burdirection` (`bd_`) | arah bureaus, utilisasi, inquiry, delinquency; volatilitas momentum | **ditolak**, membuangnya menaikkan AUC 0.000263 dan KS 0.001199 |

`burdirection` ditolak karena menambah 22 kandidat lemah ke probe gain yang hanya memilih 200 kolom. Kandidat lemah tidak pernah terpilih, tapi tetap menyingkirkan kolom yang sebelumnya terpilih.

### Level seleksi diukur ulang

v63 menemukan 200 kolom sebagai puncak, diukur di `lr 0.02 / lambda_l2 20`. Di `lr 0.01 / lambda_l2 5` jawabannya berbeda, dan yang lebih penting adalah sebaran antar seed.

| Level | AUC | KS | sd KS antar seed |
|---|---|---|---|
| tanpa seleksi | 0.846330 | 0.525506 | 0.001454 |
| **top 250** | 0.846122 | **0.526768** | **0.000342** |
| top 400 | 0.846305 | 0.525219 | 0.000217 |

### Self-training ditolak, dan alasannya

Pseudo-labeling diukur tanpa menipu diri sendiri: label semu hanya dari test, skor dihitung pada validation fold yang tidak pernah masuk training di kedua tahap.

| Setelan | AUC sebelum | sesudah | KS sebelum | sesudah |
|---|---|---|---|---|
| frac 0.20, bobot 0.5 | 0.846005 | 0.845477 | 0.526527 | 0.524218 |
| frac 0.10, bobot 0.2 | 0.846005 | 0.846012 | 0.526527 | 0.526834 |

Setelan halus terlihat positif secara mean, tapi delta KS **per seed** adalah `+0.001346`, `-0.001870`, `+0.001448`. Mean positifnya datang dari dua seed yang menutupi satu seed yang lebih besar.

Penyebabnya terlihat setelah dicek: **train dan test berinterleave pada rentang tanggal yang sama**, keduanya 2025-01-01 sampai 2025-12-28, 100000 nasabah berbeda, tanpa tumpang tindih. Keuntungan self-training datang dari pergeseran distribusi, dan di sini tidak ada yang perlu digeser.

### Cara menjalankan

```bash
# bangun blok fitur baru (harus memuat semua blok, cache ditimpa)
python3 agas/shared/build_extra.py bureau,repayment,previous,account,ecommerce,interactions,gradenorm,rxtra,bxtra,axtra,ecommerce2,consistency,dataquality,stress

# pengukuran cepat per blok, tiga fold seed
python3 agas/v65/training/cv3.py s250 folds=5 lr=0.01 l2=5 seeds=0,1 top_n=250
python3 agas/v65/training/cv3.py s0   folds=5 lr=0.01 l2=5 seeds=0,1

# uji self-training
python3 agas/v65/training/study_pseudo_v65.py 0.10 0.2

# run utama, 86 menit
python3 agas/v65/training/train_submission_v65.py --folds 10 --fold-seeds 42,7,13,21,5,101 --top-n 250 --out-prefix v65

# kandidat
python3 agas/v65/make_submission_v65.py
```

### Hasil run utama

10 fold, enam fold seed, 598 kandidat kolom, empat kaki, 86.4 menit.

| Kaki | AUC | KS |
|---|---|---|
| `xgb` | **0.847474** | 0.526091 |
| `sel` (250 kolom) | 0.847395 | **0.528338** |
| `lean` (semua kolom) | 0.847303 | 0.524940 |
| `cat` | 0.846855 | 0.525920 |
| `consensus_ranks` | **0.847708** | 0.527359 |
| `mix_all4` | 0.847682 | 0.527676 |

AUC 0.847708 adalah yang tertinggi sepanjang proyek, +0.00096 di atas kaki terbaik v64.

### Dua arah sumbu ekstrapolasi, bukan satu

`qc_00` 0.56171 memakai arah `kaya - miskin`; v61 0.56312 memakai `terpilih - luas`. Keduanya arah yang sama hanya karena di titik operasi v61 lebih banyak fitur berarti lebih baik. Di v65 gradiennya berbalik, kaki 250 kolom mengalahkan kaki 576 kolom, jadi kedua arah dijalankan dan tandanya tidak diasumsikan.

| Arah | Rasio | AUC | KS |
|---|---|---|---|
| `sel - lean` | 0.25 | 0.847263 | **0.528592** |
| `sel - lean` | 0.5 | 0.847071 | 0.528105 |
| `lean - sel` | 0.25 | 0.847127 | 0.525256 |

Arah v61 menang lagi. Hasil paling menarik: **KS OOF 0.528592 praktis menyamai KS OOF v61 0.528624, dengan AUC 0.847263 melawan 0.846699.** Kandidat pertama yang cocok dengan v61 pada KS sekaligus lebih baik pada AUC, dan v61 adalah file yang memberi 0.56312.

### Kandidat final

| Slot | Nama | AUC | KS dikurangi 1 sd | Spearman ke anchor |
|---|---|---|---|---|
| 1 | `v65\|consensus_ranks` | 0.847708 | 0.522074 | 0.9918 |
| 2 | `v62\|lgb_top` | 0.847419 | 0.520265 | 0.9919 |
| 3 | `v63\|sel200` | 0.846981 | 0.521045 | 0.9904 |
| 4 | `v65\|lean-sel_r0.5` | 0.846896 | 0.519320 | 0.9900 |
| 5 | `v65\|cat` | 0.846855 | 0.520635 | 0.9879 |
| 6 | `v65\|sel-lean_r0.75` | 0.846815 | 0.522327 | 0.9882 |
| 7 | `v64\|xgb_full` | 0.846749 | 0.520200 | 0.9946 |
| 8 | `v64\|cat_full` | 0.846193 | 0.520083 | 0.9905 |

### Status

Delapan file tervalidasi, 37 script fine, notebook 18 cell tanpa error. Belum ada submission Kaggle dan belum ada push GitHub. Public score file mana pun di run ini belum diketahui, dan itulah satu-satunya ukuran yang benar.
