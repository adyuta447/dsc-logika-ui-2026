# DSC LOGIKA UI 2026, tim "kali ini beneran deh" (DSC0152)

Kaggle: data-science-competition-logika-ui-2026, metrik KS. Deadline 30 Okt 2026 23:59 WIB.
Nilai akhir = 45% public + 45% private + 10% reproducibility notebook. Maks 3 submit per hari, reset 07:00 WIB.

## Status terakhir (12 Okt 07:05 WIB)

- Public terbaik: **0.55865** (run_fac top_mix, 12 Okt), peringkat 11 dari 62 tim. Yang lolos final: **top 8** (rank 8 sekarang 0.56149).
- Model terbaik di CV: notebook 10 fold + factor model, AUC **0.84649** (top_mix).
- Submit 12 Okt 07:01: satu file top_mix. Pakai `run_seeds` (2 seed) kalau sudah selesai dan CV lebih baik (AUC > 0.84649, KS >= 0.5237), selain itu `run_fac`. Slot 2 dan 3 TIDAK otomatis, menunggu keputusan setelah skor slot 1 dilihat.
  Kandidat: `run_full_fac` top_mix (refit 100% + factor), linear30 dari run factor terbaik.
- Sedang jalan: varian fold seed 7 (`run_s7/`, 10 fold + factor), selesai sekitar 09:45 WIB, untuk slot hari berikutnya.
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

## CV per versi notebook

| Versi | Folder | main AUC | top_mix AUC | top_mix KS |
|---|---|---|---|---|
| 5 fold | root | 0.84507 | 0.84518 | 0.52588 |
| 10 fold | `run10/` | 0.84579 | 0.84580 | 0.52556 |
| 10 fold + factor | `run_fac/` | 0.84635 | **0.84649** | 0.52568 |
| 10 fold + factor, 2 seed cat/xgb | `run_seeds/` | 0.84639 | 0.84648 | 0.52510 |
| refit 100% data + factor | `run_full_fac/` | tidak bisa di CV | | |

## Temuan penting

- Sinyal terkuat: harga satuan belanja ecommerce relatif terhadap median kategori (`ec_price_rel`).
- Suku bunga adalah grade risiko 5 level (16% sampai 32%), default rate 5.5% sampai 50%.
- Factor model (11 Okt): 12 sumber digabung jadi satu skor risiko, tiap nasabah dibobot sesuai jumlah datanya. Loading dihitung per fold dari label train fold saja. LightGBM 0.84437 jadi 0.84524.
- Train dan test identik (adversarial AUC 0.503). Noise public besar: model yang sama bisa beda sekitar 0.01 tergantung potongan.

## Sudah dicoba, tidak membantu

400+ fitur tambahan, MLP, GAM, stacking, pseudo label, seleksi fitur, denoising harga, normalisasi per kelompok,
pisah data 6 bulan di factor model (0.84502), tambah 6 sumber di factor model (0.84521), pembersihan noise cicilan,
sub factor per kelompok (0.84517), shrinkage semua sumber (0.84523),
optimasi bobot blend (validasi setengah data: +0.00002 dan +0.0001, bobot manual sudah hampir optimal). Patokan factor v1: 0.84524 (LightGBM lean, 5 fold, 2 seed).

## Struktur repo

- `features.py`: pipeline fitur, termasuk `factor_inputs` dan `factor_features`.
- `make_nb.py`: generator notebook. `python make_nb.py <output.ipynb> <N_FOLDS> <TREE_SEEDS> <SEED>`. SEED mengatur pembagian fold, 42 = versi utama.
- `DSC0152_kali ini beneran deh_Notebook.ipynb`: notebook deliverable (masih versi 5 fold tanpa factor, diganti setelah pilihan final).
- `run10/`, `run_fac/`, `run_seeds/`: notebook 10 fold yang sudah dijalankan beserta outputnya.
- `experiments/`: skrip eksperimen dan log hasilnya.
- `scheduled_submit*.sh`: skrip submit terjadwal.

Data kompetisi, model, dan file CSV tidak ikut di repo (lihat `.gitignore`). Untuk menjalankan, taruh data di `data/`
lalu jalankan notebook.
