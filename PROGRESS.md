# DSC LOGIKA UI 2026, tim "kali ini beneran deh" (DSC0152)

Kaggle: data-science-competition-logika-ui-2026, metrik KS. Deadline 30 Okt 2026 23:59 WIB.
Nilai akhir = 45% public + 45% private + 10% reproducibility notebook. Maks 3 submit per hari, reset 07:00 WIB.

## Status 17 Okt 00:45 WIB

- Laptop restart 16 Okt 20:59: scheduler dan `replan.sh` mati, scratchpad planner di /private/tmp (prediksi model 11 tahap training `div/*.npz`, `ks_cache.pkl`) ikut terhapus. File antrean `variants/p22_*.csv` masih ada.
- `scheduled_unified.sh` jalan lagi dengan antrean p22 (42 file) di slot `experiments/slots.py` mulai 17 Okt 07:01 (p22_00 dulu, file 18:01 kemarin yang gagal karena kuota). Kalau submit gagal (kuota habis karena submit manual Agas, atau tidak ada internet) file yang sama dicoba lagi di slot berikutnya, tidak dibuang.
- Re-plan otomatis tidak jalan lagi sampai prediksi `div/` dilatih ulang (berhari hari). Urutan p22 tetap pilihan terbaik dari semua skor yang sudah terlihat.
- Cek 17 Okt: urutan baris, hash application_id/customer_id, tren waktu: AUC 0.496 sampai 0.501, tidak ada bocoran.
- 17 Okt 01:10: file v61 Agas (public 0.56312) plus v57 dan v58ext diunduh dari Kaggle (`experiments/download_sub.py <ref> <out>`, CLI tidak punya perintah ini) ke `agas/output/dl/`.
  v61 tidak bisa dibuat ulang dari model kita (spearman terbaik 0.9934 dengan ext0.5 v4 linear30 minus lean). `experiments/v61_plan.py`: file tanpa OOF diberi pseudo OOF
  (proyeksi ke komponen notebook kita plus noise sebesar sisanya), 4000 potongan public simulasi dikondisikan ke 14 skor public. Hasil stabil di 3 seed noise:
  campuran 70% v61 + 30% file kita paling unggul (peluang di atas 0.565 sekitar 10% per file, antrean p22 maks 5.6%).
  Antrean sekarang: 17 Okt 07:01 `v61_00_mix0.7_v61_p19_00`, 12:01 `v61_01_mix0.85_v61_qc00`, 18:01 `v61_02_mix0.5_v61_p19_00`, 18 Okt 07:01 `v61_03_mix0.5_v61_p17_00`,
  lalu p22. Peluang minimal satu dari empat di atas 0.565: sekitar 25%.
- 17 Okt 15:50: v61_00 dapat 0.56299, v61_04 (70% v61 + p17_00) 0.56253, v63f Agas (manual 11:35, kuota hari ini habis) 0.55712. Ketiganya masuk ke `v61_plan.py`.
  Peluang per file turun ke maks 5% (mix0.5 v61+p19_00). Antrean mulai 18 Okt 07:01 (slot 18:01 hari ini gagal karena kuota, file disimpan):
  `v61_02_mix0.5_v61_p19_00`, `v61_01_mix0.85_v61_qc00`, `v61_06_mix0.85_v61_p19_00`, `v61_07_mix0.85_v61_v4full`, `v61_08_ext0.25_v61_p21_00`, lalu v61_05, v61_03, p22.
  Peluang minimal satu dari empat pertama di atas 0.565: sekitar 16 sampai 19%.
- 17 Okt 23:50: kandidat campuran v61 dengan v63f Agas ditambahkan ke `v61_plan.py`. 70% v61 + 30% v63f peluangnya 8.7% (rata 2 seed), tertinggi; 85/15 7.5%.
  Ekstrapolasi menjauh dari v63f justru turun (2% ke bawah). Antrean 18 Okt: 07:01 `v61_09_mix0.7_v61_v63f`, 12:01 `v61_02_mix0.5_v61_p19_00`, 18:01 `v61_10_mix0.85_v61_v63f`,
  lalu v61_01, v61_06, v61_07, v61_08, v61_05, v61_03, p22. Peluang minimal satu dari empat pertama di atas 0.565: sekitar 20 sampai 23%.
- 18 Okt 00:15: Agas push `agas/v6x/output/preds_v6x.npz` (OOF dan test asli, urutan baris sama dengan data/*.csv). Komponennya masuk ke `v61_plan.py` sebagai kandidat
  dan basis proyeksi: v61 dan v63f sekarang terproyeksi dengan korelasi 0.99999 (sebelumnya 0.993). Hasilnya angka 8.7% campuran v63f di atas ternyata dari noise pseudo OOF;
  dengan OOF asli campuran itu cuma 1.7 sampai 1.9%. Peluang terbaik sekarang sekitar 5% per file, rata 3 seed. Antrean baru mulai 18 Okt 07:01:
  `v61_11_mix0.7_v61_av64linear` (4.9%), `v61_12_mix0.5_v61_qc00` (4.7%), `p22_08` (3.8%), `v61_03` (4.2%), `v61_13_ext0.5_v61_v63f` (4.6%), `v61_14_mix0.7_v61_av64cat` (4.1%),
  lalu sisa v61 dan p22. Peluang minimal satu dari empat pertama di atas 0.565: sekitar 15 sampai 19%.
- 18 Okt 11:55: v61_11 dapat 0.56086 (perkiraan 0.5615). Dua submit manual lain pagi ini (07:20 `submission_ready.csv` 0.48980, 07:48 `submission_v2.csv` 0.44707)
  menghabiskan kuota, jadi slot 12:01 dan 18:01 gagal. Antrean mulai 19 Okt 07:01: `v61_03_mix0.5_v61_p17_00` (4.5%), `v61_14_mix0.7_v61_av64cat` (4.2%),
  `v61_12_mix0.5_v61_qc00` (4.5%), `v61_06_mix0.85_v61_p19_00`, `p22_07`, `p22_08`, `v61_02`, lalu sisanya.
- 18 Okt 22:10: seed ke-4 v4 (`run_v4_s21`, selesai 14 Okt) tidak menambah: AUC 0.84772 vs 0.84774, test spearman 0.999998, jadi training seed baru (sekitar 7 jam per seed) tidak dijalankan.
  `experiments/team_stack.py` menulis `variants/team_stack.csv` (LR di atas 64 komponen kita + Agas, AUC CV 0.84806), kandidat final untuk private.
  Masuk antrean slot ke-5 (20 Okt 12:01) supaya sudah pernah disubmit dan bisa dipilih sebagai final.

## Status terakhir (14 Okt 01:15 WIB)

- Rank 18 dari 83, public terbaik **0.56171** (13 Okt 18:43, ekstrapolasi 1.5 v4 dikurangi fac, full_only). Rank 13: 0.56341, rank 10: 0.56440, rank 8: 0.56572, rank 1: 0.57059.
- Antrean 50 submit otomatis (`scheduled_unified.sh`, file `variants/pNN_*.csv`, sekarang p05) mulai 14 Okt 10:01. Jam submit 14 Okt: 10:01, 15:01, 20:01; hari lain 07:01, 12:01, 18:01; 30 Okt hanya 07:01 dan 12:01, 30 Okt 18:01 cadangan.
- `replan.sh` menyusun ulang antrean otomatis setiap ada skor public baru atau training selesai (`experiments/harvest_kf.py`: 10000 potongan public simulasi, dikondisikan ke semua skor public yang sudah terlihat, file dipilih demi peluang tembus 0.565).
  p05 (339 kandidat): peluang 0.565 54%, 0.568 10%. Batas akan naik karena tim lain juga submit.
- Pencarian keberuntungan (`experiments/luck_search.py`, `experiments/luck_check.py`): dari 1,2 juta potongan simulasi, 1086 cocok dengan semua skor public kita. Isinya tidak berbeda dari potongan acak (pergeseran fitur maks 0,1 sd, proporsi default sama), jadi keberuntungan 0.56171 murni soal baris mana yang masuk public. File 0.56171 itu lebih jelek di semua 70000 baris train (AUC 0.84654 vs 0.84774), pilihan final tetap bag v4 top_mix.
  Antrean p05 dinilai ulang di 1086 potongan itu: peluang 0.565 55%, sama dengan planner; file 10:01 punya peluang tunggal tertinggi (7%).
- Training: tahap 4 selesai 14 Okt 00:52. Tahap 5 (38 model) dan v4 fold seed 21 jalan, lalu tahap 6 dan 7.
- Notebook final: verifikasi penuh selesai 14 Okt 00:40, output **bit-identical** dengan bag v4 top_mix yang disubmit. Notebook dan CSV deliverable di folder utama sudah diganti dengan hasil ini (versi lama di `old_deliverables/`). Bobot `run_final/models` (1,4 GB) belum diupload ke Drive.
- Uji 13 Okt tanpa sinyal: pola waktu antar tabel (LR maks 4.2), arus kas vs cicilan (LR maks 2.8). Setiap nasabah hanya muncul sekali (train 70000, test 30000, tidak ada irisan).

- Public terbaik: **0.55865** (run_fac top_mix, 12 Okt), peringkat 20 (12 Okt 23:15, rank 8: 0.56440). Yang lolos final: **top 8** (rank 8 sekarang 0.56149).
- Model terbaik di CV: notebook 10 fold + factor model, AUC **0.84649** (top_mix).
- Submit 12 Okt 07:01: satu file top_mix. Pakai `run_seeds` (2 seed) kalau sudah selesai dan CV lebih baik (AUC > 0.84649, KS >= 0.5237), selain itu `run_fac`. Slot 2 dan 3 TIDAK otomatis, menunggu keputusan setelah skor slot 1 dilihat.
  Kandidat: `run_full_fac` top_mix (refit 100% + factor), linear30 dari run factor terbaik.
- 12 Okt: slot 3 tidak dipakai (keputusan tim).
- Sedang jalan: seed 13 dan 21 (`run_seed_queue.sh`), uji setelan parameter (`experiments/fz6.py`), uji model khusus nasabah baru (`experiments/newcust.py`).
  Rencana sore: gabungkan semua yang lolos uji jadi satu run final (bagging 4 seed, varian trees_only dan top_mix).
  `bag.py` merata ratakan blend dari beberapa seed, CV dihitung dari rata rata OOF. File: `variants/bag_top_mix_N.csv`.
- Run 2 seed CatBoost dan XGBoost (`run_seeds/`) selesai: top_mix 0.84648 / KS 0.52510, tidak lebih baik dari 1 seed (0.84649 / 0.52568). Slot 1 pagi ini tetap `run_fac` top_mix.

## Riwayat submit

| Tanggal | File | CV | Public |
|---|---|---|---|
| 10 Okt | lgb 5 fold v1 | KS 0.505 | 0.54417 |
| 10 Okt | lgb + harga relatif + bureau gabungan | KS 0.521 | 0.55679 |
| 10 Okt | blend lgb 3 seed + catboost | KS 0.522 | 0.55717 |
| 11 Okt | blend v3+v4+cat | AUC 0.8451 | 0.55752 |
| 11 Okt | blend cat heavy | AUC 0.8447 | **0.55784** |
| 11 Okt | + spline logistic 20% | AUC 0.8450 | 0.55658 |
| 12 Okt | run_fac top_mix (10 fold + factor) | AUC 0.84649 | **0.55865** |
| 12 Okt | bag seed 42+7 linear30 | AUC 0.84663 | 0.55826 |
| 12 Okt | bag v4 seed 42+7+13 top_mix | AUC 0.84774 | 0.55860 |
| 13 Okt | ekstrapolasi 1.5 v4 dikurangi fac, trees_only | | 0.55810 |
| 13 Okt | LightGBM fitur lengkap (v4, 3 fold seed, lr 0.02) | AUC 0.84729 | **0.55931** |
| 13 Okt | ekstrapolasi 1.5 v4 dikurangi fac, full_only (slot 18:01, terkirim 18:43 karena laptop tidur) | | **0.56171** |

## CV per versi notebook

| Versi | Folder | main AUC | top_mix AUC | top_mix KS |
|---|---|---|---|---|
| 5 fold | root | 0.84507 | 0.84518 | 0.52588 |
| 10 fold | `run10/` | 0.84579 | 0.84580 | 0.52556 |
| 10 fold + factor | `run_fac/` | 0.84635 | **0.84649** | 0.52568 |
| 10 fold + factor, 2 seed cat/xgb | `run_seeds/` | 0.84639 | 0.84648 | 0.52510 |
| 10 fold + factor, fold seed 7 | `run_s7/` | 0.84637 | 0.84664 | 0.52723 |
| bag fold seed 42 + 7 | `variants/bag_*_2.csv` | | 0.84670 | 0.52642 |
| 10 fold + factor, fold seed 13 | `run_s13/` | | 0.84666 | 0.52507 |
| bag fold seed 42 + 7 + 13 | `variants/bag_*_3.csv` | | 0.84678 | 0.52645 |
| **v4: + musim + porsi gaji + saldo awal + jarak pengajuan/belanja** | `run_v4_s42/` | 0.84726 | 0.84748 | 0.52752 |
| v4 fold seed 7 | `run_v4_s7/` | | 0.84760 | 0.52884 |
| bag v4 seed 42 + 7 | `variants/bag_run_v4_s42_*_2.csv` | | 0.84769 | 0.52826 |
| v4 fold seed 13 | `run_v4_s13/` | | 0.84754 | 0.52725 |
| **bag v4 seed 42 + 7 + 13** | `variants/bag_run_v4_s42_*_3.csv` | | **0.84774** | 0.52786 |
| v4 learning rate 0.005 | `run_v4_lr005/` | 0.84715 | 0.84741 | 0.52697 |
| bag v4 3 seed + lr 0.005 | `variants/bag_run_v4_s42_*_4.csv` | | 0.84769 | 0.52771 |
| refit 100% data + factor | `run_full_fac/` | tidak bisa di CV | | |

## Strategi baru (12 Okt malam)

- Noise potongan public antar model kita ternyata 0.0015 sampai 0.003 (bukan 0.0005). Itu sebabnya bag v4 tidak naik di public walau CV naik.
- Public: submit varian kuat yang beragam (`variants/harvest_*.csv`, 12 file siap), skor public terbaik yang dihitung. Perkiraan dengan 13 submit: rank 15 sekitar 71%, rank 13 41%, rank 10 16%, rank 8 7%.
- Private: pilih bag v4 top_mix sebagai final (CV terbaik), plus varian kedua kalau boleh pilih dua.
- Perlu ditanyakan ke panitia: komponen public 45% itu skor public terbaik atau skor public dari submission yang dipilih, dan boleh pilih berapa submission final.

## Temuan penting

- Sinyal terkuat: harga satuan belanja ecommerce relatif terhadap median kategori (`ec_price_rel`).
- Suku bunga adalah grade risiko 5 level (16% sampai 32%), default rate 5.5% sampai 50%.
- Factor model (11 Okt): 12 sumber digabung jadi satu skor risiko, tiap nasabah dibobot sesuai jumlah datanya. Loading dihitung per fold dari label train fold saja. LightGBM 0.84437 jadi 0.84524.
- Train dan test identik (adversarial AUC 0.503). Noise public besar: model yang sama bisa beda sekitar 0.01 tergantung potongan.

## Sudah dicoba, tidak membantu

400+ fitur tambahan, MLP, GAM, stacking, pseudo label, seleksi fitur, denoising harga, normalisasi per kelompok,
pisah data 6 bulan di factor model (0.84502), tambah 6 sumber di factor model (0.84521), pembersihan noise cicilan,
sub factor per kelompok (0.84517), shrinkage semua sumber (0.84523),
optimasi bobot blend (validasi setengah data: +0.00002 dan +0.0001, bobot manual sudah hampir optimal),
bobot kelas untuk KS (AUC turun ke 0.840), empirical logit sumber biner (0.84505), kalibrasi ulang per kelompok (turun), bobot blend dioptimalkan untuk KS (validasi setengah data: -0.00016 dan +0.00027),
batasan monoton (0.84522), num_leaves 31 (0.84498), min_child_samples 100 (0.84527), feature_fraction 0.5 (0.84503),
model khusus nasabah baru (AUC di kelompok baru 0.8277 vs global 0.8293; dirata rata 0.84520 vs 0.84523).
18 Okt: 44 fitur ecommerce baru (`experiments/ec_deep.py`: pembatalan per metode bayar, harga barang batal vs jadi, jam malam, tanggal gajian, isi keranjang,
harga per kategori x jumlah dan kategori x bulan). Saringan AUC kondisional terhadap OOF v4 top_mix memunculkan `ed_lrel_cancel_gap` (z 6.9), jam belanja dan e-wallet,
tapi di model penuh (`addtest.py`, 2 seed) 9 fitur teratas +0.00018 dan -0.00015, `ed_lrel_cancel_gap` sendiri +0.00013 dan -0.00040: tidak membantu.
Gabungan pipeline kita + Agas (stacking LR di dalam fold): AUC OOF 0.84806 vs 0.84774 kita saja, KS 0.52848 vs 0.52786; prior public tertinggi tapi peluang 0.565 cuma 0.6 sampai 3%.
18 Okt: 30 fitur baru dari tabel lain (`experiments/other_deep.py`, saringan `experiments/cond_screen.py`): skor bureau distandarkan per provider, tren utilisasi,
selisih suku bunga sekarang vs pinjaman terakhir, pinjaman vs pengajuan terakhir, drawdown saldo, pembayaran sebagian. 8 teratas di model penuh -0.00003 dan -0.00037,
dua terkuat +0.00007 dan -0.00029: tidak membantu. Fakta data: term_months selalu 12, cicilan = anuitas dibulatkan ke atas (selisih 0 sampai 1 rupiah),
saldo rekening konsisten persis, semua laporan bureau tersedia sebelum keputusan, tidak ada pembayaran lebih awal.
`data/ecommerce_transactions.csv` sempat hilang dari laptop (sejak sekitar 15 Okt 21:07), diunduh ulang 18 Okt 00:30 dan dicek identik (jumlah order per nasabah 100% sama dengan features.parquet).

## Analisis batas skor (12 Okt pagi)

- Varian fold seed hampir identik di data test (spearman 0.99996), jadi tidak menambah keragaman skor public. Yang berbeda hanya varian bobot blend (linear30 0.9998, top_only 0.9995) dan versi tanpa factor (0.996).

- Noise harga dan skor bureau berdistribusi normal murni (skew 0.00, kurtosis 0.01), jadi rata-rata per nasabah sudah penaksir optimal.
- Ambang KS ada di p 0.139, sama dengan base rate 0.136. Kalibrasi antar kelompok sudah baik.
- AUC per desil ketidakpastian risiko: 0.86 untuk nasabah yang datanya lengkap, 0.83 untuk yang datanya sedikit. Batas teoritis kalau risiko terukur sempurna sekitar 0.869 (sekarang 0.8465). Sisa celah berasal dari jumlah data per nasabah, bukan dari model.
- Semua 72 kolom di 7 tabel sudah dipakai. Hubungan antar tabel (hutang di luar, akun di luar, telat di bureau vs internal) dan proses bureau untuk pengajuan sekarang tidak berisi sinyal.
- Musim: default rate per bulan 12.0% (Mei) sampai 14.7% (Feb). Kuat untuk kalibrasi (LR 45), tapi koreksi bulan di luar sampel hanya AUC +0.00012.
  Sebagai fitur `dec_doy` di LightGBM: 0.84524 jadi 0.84544. Interaksi musim dengan region, pekerjaan, rumah, umur, tujuan, grade: tidak ada.
  Volume pengajuan per hari atau minggu: tidak ada.

Patokan factor v1: 0.84524 (LightGBM lean, 5 fold, 2 seed).

## Struktur repo

- `features.py`: pipeline fitur, termasuk `factor_inputs` dan `factor_features`.
- `make_nb.py`: generator notebook. `python make_nb.py <output.ipynb> <N_FOLDS> <TREE_SEEDS> <FOLD_SEEDS> <LR>`, contoh final: `10 1 42,7,13 0.01`.
- `DSC0152_kali ini beneran deh_Notebook.ipynb`: notebook deliverable (masih versi 5 fold tanpa factor, diganti setelah pilihan final).
- `run10/`, `run_fac/`, `run_seeds/`: notebook 10 fold yang sudah dijalankan beserta outputnya.
- `experiments/`: skrip eksperimen dan log hasilnya.
- `scheduled_submit*.sh`: skrip submit terjadwal.

Data kompetisi, model, dan file CSV tidak ikut di repo (lihat `.gitignore`). Untuk menjalankan, taruh data di `data/`
lalu jalankan notebook.

## Log submit otomatis

- 2026-10-13 07:02 queue_00_ext1.5_v4_fac_trees_only: public 0.55810, rank 
- 2026-10-13 12:02 qb_00_new_lgb_fullset: public 0.55931, rank 21
- 2026-10-13 18:43 qc_00_ext1.5_v4_fac_full_only: public 0.56171, rank 18
- 2026-10-14 10:02 p07_00_ext1.0_full_lgb_extra31_facfull_lgb_extra31: public 0.55994, rank 20
- 2026-10-14 17:41 p10_00_ext2.0_plus_ec_lgb_extra_lgb_extra: public 0.55763, rank 22
- 2026-10-15 07:15 p12_00_ext2.0_plus_prof_lgb_mc800_lgb_mc800: public 0.55789, rank 27
- 2026-10-15 12:02 p17_00_mix2_ext1.0_v4_fac_xgb_full_ext1.5_v4full_faclean: public 0.56083, rank 28
- 2026-10-16 07:10 p19_00_ext1.0_full_lgb_extra_lgb_extra: public 0.56014, rank 31
- 2026-10-16 12:02 p21_00_ext1.5_full_lgb_shallow_facfull_lgb_shallow: public 0.55779, rank 25
- 2026-10-17 07:02 v61_00_mix0.7_v61_p19_00: public 0.56299, rank 26
- 2026-10-17 12:36 v61_04_mix0.7_v61_p17_00: public 0.56253, rank 26
- 2026-10-18 07:02 v61_11_mix0.7_v61_av64linear: public 0.56086, rank 31
