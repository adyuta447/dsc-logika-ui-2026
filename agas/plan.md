# Rencana kerja `agas/` — v71 dan setelahnya

Dokumen ini adalah rencana eksekusi aktif. Rencana berikutnya setelah v71
ditulis tepat di bawah rencana v71, di bagian **“Rencana berikutnya setelah
v71”**. Aturan kompetisi di `agas/limit.md`, `agas/overview.md`, dan
`agas/rules.md` tetap mengikat.

## Status public yang melatarbelakangi

| Kandidat | OOF AUC | OOF KS | Public | Catatan |
|---|---:|---:|---:|---|
| `v66_04` | 0.846840 | 0.528713 | **0.56424** | anchor terbaik |
| `v69_10` | 0.846292 | 0.528254 | 0.55989 | pivot `extra` gagal |
| `v70_06` | 0.847623 | 0.530232 | 0.56040 | OOF KS tertinggi, public turun |
| `v70_07` | 0.847249 | 0.529397 | 0.56419 | praktis sama dengan anchor |
| `v70_01` | 0.848369 | 0.528503 | 0.55801 | AUC tertinggi, public turun |

Kesimpulan: keluarga fitur/model v66–v70 sudah jenuh. Perbedaan OOF di bawah
~0.001 tidak memprediksi public. Target 0.577 adalah ~2.5 sd public di atas
0.56424 dan sebagian besar kemungkinan merupakan ekor noise leaderboard. Karena
itu v71 fokus pada **informasi baru yang benar-benar berbeda**, bukan menambah
blok fitur redundan atau mengubah rasio blend.

## Prinsip yang tidak boleh dilanggar

- Python saja; hanya data kompetisi lokal di `data/`.
- Tanpa external data, AutoML, atau AutoFE.
- Tanpa leakage: semua fitur memakai cutoff `decision_datetime`; data historis
  hanya memakai outcome yang window 12 bulannya sudah selesai.
- Semua artefak baru di `agas/v71/{training,notebook,output}`; `features.py`
  tidak diubah.
- Notebook reproducible, dijalankan penuh dengan `RUN_TRAINING=False`.
- Tidak push GitHub dan tidak submit Kaggle tanpa instruksi eksplisit.
- Update `agas/progress.md`, `agas/README.md`, dan Obsidian `coding-vault` note
  `Data Science Competition Logika UI.md`.

# Rencana v71 — historical panel + auxiliary risk + public proxy

## Diagnosis yang harus diselesaikan

1. **OOF vs public**: buat public-proxy bootstrap dari OOF (subset stratifikasi
   seukuran public) untuk mengukur mean, sd, dan probabilitas kandidat
   mengalahkan anchor. Ini menentukan apakah selisih 0.001–0.004 memang noise.
2. **Informasi baru**: 74.918 pinjaman lama punya window 12 bulan lengkap dan
   label proksi 90+ DPD (positif 16.50%). Ini adalah training set kedua yang
   belum pernah dipakai.
3. **Auxiliary rate**: rate tier adalah grade risiko lender (default 5.49%–
   50.29%). Residual rate terhadap fitur risiko adalah sinyal yang belum
   dieksploitasi.
4. **Seleksi kandidat**: ganti argmax OOF KS dengan AUC + public-proxy yang
   robust; pertahankan kedekatan dengan anchor sebagai faktor risiko.

## Fase 0 — audit dan public proxy (cepat)

- Audit event-time: tidak ada data setelah `decision_datetime`; cek negatif
  `days_ago` untuk semua sumber.
- Adversarial validation train vs test (sudah: AUC 0.501; dicatat ulang).
- Bootstrap 1.000 subset OOF; hitung KS tiap leg/kandidat; laporkan
  `P(candidate > anchor)`.
- Tentukan kandidat diagnostik yang paling informatif, bukan yang OOF KS-nya
  tertinggi.

## Fase 1 — historical panel (lever utama)

### 1A. Dataset panel

- Ambil `previous_applications` berstatus `approved` dan punya `loan_id`.
- Label: `max_days_past_due >= 90` pada 12 bulan pertama sejak due date pertama.
- Hanya pakai pinjaman yang window 12 bulannya selesai sebelum
  `decision_datetime` aplikasi sekarang (74.918 baris; 16.50% positif).
- Fitur dihitung pada `decision_datetime` pinjaman historis, bukan pada
  aplikasi sekarang:
  - terms pinjaman historis: requested/approved amount, rasio, rate, tenor,
    installment, decision lag, jam/hari;
  - previous applications sebelum cut: jumlah, approval/rejection, amount, rate,
    tren rate, recency, gap, active count, rasio amount/income;
  - repayment history sebelum cut: late30/late90, DPD, unpaid, paid ratio,
    payment delay, tren, outcome per loan, active loan;
  - bureau observation milik `application_id` historis: score, provider
    disagreement, utilization, inquiries, delinquency;
  - account behavior sebelum cut: inflow/outflow, net, saldo, low-balance days,
    tren 1/3/6/12 bulan;
  - e-commerce sebelum cut: spend/order window 30/90/180/365, basket, kategori,
    payment method, cancel, tren. Agregasi harian per customer dulu agar hemat
    memori.
- Profil customer adalah snapshot saat ini. Untuk baris historis, umur dan
  tenure di-backdate; income dipakai dengan catatan keterbatasan ini di
  dokumentasi. Tidak ada target aplikasi sekarang yang dipakai.

### 1B. Model panel

- LightGBM binary, 5-fold **GroupKFold by customer**.
- Evaluasi AUC/KS pada customer yang tidak pernah dilihat model panel.
- Prediksi panel untuk baris sekarang dibuat dengan cross-fitting per customer
  (model untuk customer group g tidak dilatih pada label historis customer g).
- Prediksi test memakai model panel yang dilatih pada seluruh baris historis.
- Tambahkan `panel_score`, `panel_rank`, dan fitur event-time terpilih ke cache
  fitur v71.

### 1C. Go/no-go

- Panel model harus punya AUC held-out yang masuk akal (≥0.75) dan prediksinya
  harus menambah **≥0.0005 AUC** ke model sekarang pada 5 fold × 3 seed.
- Jika gagal, panel tetap didokumentasikan sebagai negative result; fitur
  event-time individual boleh diuji terpisah.

## Fase 2 — auxiliary rate dan fitur murah

- Latih model auxiliary target-independent untuk memprediksi rate tier dari
  fitur selain rate (out-of-fold). Tambahkan:
  - `rf_pred_rate`, `rf_rate_residual`, `rf_rate_surprise`,
  - probabilitas tier aktual, entropy prediksi.
- Tambahkan payment-timing trend dan bureau-provider disagreement yang belum
  ada.
- Screening 5 fold × 3 seed; hanya diterima jika mean AUC naik ≥0.0005 dan
  konsisten di ketiga seed.

## Fase 3 — training v71 dan seleksi

- Feature cache: `features_v70.pkl` + blok panel + blok auxiliary.
- Screen 5 fold × 3 seed untuk setiap blok; buang yang tidak lolos.
- Final run: 15 fold × minimal 3 fold seed × 2 konfigurasi LightGBM
  (`extra` dan satu varian regularisasi), memakai factor features fold-safe
  seperti v66–v70.
- Kandidat dibuat dari rantai `tm30`/`ext0.5` yang sudah terbukti, dengan leg
  v71 sebagai core tambahan; sertakan leg mentah dan satu kandidat konservatif
  yang sangat dekat dengan anchor.
- Seleksi memakai AUC + public-proxy bootstrap, bukan OOF KS mentah.
- Validasi CSV: 30.000 baris, ID unik dan cocok test, probabilitas [0,1], tanpa
  NaN.

## Fase 4 — dokumentasi dan artefak

- `agas/v71/training/`: build panel, build fitur, screening, training, kandidat.
- `agas/v71/output/`: cache, preds, summary, submission, log.
- `agas/v71/notebook/notebook_v71.ipynb`: dieksekusi penuh dengan
  `RUN_TRAINING=False`, nol error.
- Update `agas/progress.md`, `agas/README.md`, dan Obsidian note.
- Commit lokal; tanpa push dan tanpa submit.

## Ekspektasi dan batas

- Target realistis v71: **0.566–0.569** dari kenaikan MLP; panel dan rate
  residual sudah terbukti tidak menambah sinyal.
- 0.577 tidak dijamin; kemungkinan besar sebagian adalah noise public. v71
  berusaha memaksimalkan expected score dan menyediakan kandidat yang tahan
  public-proxy, bukan mengejar argmax noise.
- Jika panel/rate gagal, negative result tetap bernilai: keluarga v66–v70
  adalah plafon feature engineering ini, dan langkah berikutnya harus model
  diversity/sequence seperti pada rencana v72 di bawah.

---

# Rencana berikutnya setelah v71 — v72: MLP-first, lalu sequence

## Temuan v71 yang mengubah arah

- Historical panel **gagal** menambah model sekarang: panel score saja
  0.847202 vs baseline 0.847219; panel + 68 fitur + rate residual 0.847199.
  Panel tetap disimpan sebagai artefak dan negative result, bukan fitur final.
- Auxiliary rate **gagal**: rate dapat diprediksi hampir sempurna dari fitur
  yang sudah ada (AUC rate>=24% = 1.0), residual median ~2.7e-6. `rf_*` dibuang.
- **MLP berhasil**: MLP 15 fold × 3 fold seed × 3 seed mencapai AUC 0.848433 /
  KS 0.529473; core MLP + CatBoost + v70 mencapai AUC **0.850147** / KS
  **0.532834** (MLP wide hanya 18% dari total). Ini kenaikan OOF yang cukup
  besar dibanding plafon LightGBM 0.848369.
- Public-proxy bootstrap menunjukkan keunggulan core71 tidak bergantung pada
  prevalence: pada prevalence 10–25% `dKS` tetap ~+0.004 dan P(KS lebih baik)
  0.93–0.997.
- Artinya lever berikutnya bukan feature engineering tambahan, tetapi
  **arsitektur model dan dekorrelasi error**.

## Fase 0 — validasi public v71 (wajib sebelum ekspansi)

- Submit maksimal tiga kandidat v71: `core71` (AUC tertinggi), `core71_ks`
  (KS OOF tertinggi dengan AUC dalam 0.0005 dari core), dan kandidat
  konservatif `anchor0.25|core71`.
- Catat public di `agas/progress.md` dan Obsidian. Jangan menambah kandidat
  baru sebelum hasil ini masuk.
- Gate:
  - jika `core71` < 0.56424, kembali ke anchor dan perlakukan MLP sebagai
    eksperimen yang tidak transfer ke public;
  - jika `core71` 0.564–0.567, lanjutkan Fase 1 tetapi pertahankan anchor;
  - jika `core71` > 0.567, jadikan MLP core sebagai pivot baru.

## Fase 1 — perkuat MLP (lever utama, biaya rendah)

Semua screening memakai 5 fold × 3 fold seed, lalu final 15 fold × minimal
3 fold seed × 3 seed. Jangan memilih berdasarkan KS OOF mentah.

1. **Arsitektur**: sweep `128/32`, `256/64`, `512/128`, `768/256`; dropout
   0.20–0.45; `SiLU` vs `ReLU`; weight decay 1e-4–1e-3; early stopping.
2. **Preprocessing**: standardisasi yang sekarang vs rank-gauss
   (`QuantileTransformer(output_distribution="normal")`); clipping 0.1/99.9
   vs 1/99; missing-count feature.
3. **Feature selection untuk MLP**: importance LightGBM 5-fold, lalu uji
   top-200/400/600/771. MLP mungkin lebih baik dengan fitur lebih sedikit.
4. **Factor features**: uji apakah 6 target-encoding fold-safe menambah MLP.
5. **Ensemble**: minimal tiga arsitektur × tiga seed; rata-rata rank, bukan
   probabilitas mentah. Target OOF AUC ≥0.851 dan KS ≥0.5325.

## Fase 2 — diversity pohon dan stack

- XGBoost final 15 fold × 3 fold seed; CatBoost varian depth 4/6/8; satu
  LightGBM `dart` atau objektif ranking/focal sebagai leg kontrol.
- Stack hanya dengan model sederhana (logistic regression pada rank leg, atau
  LightGBM meta depth ≤3 dengan regularisasi kuat). Meta LGB nonlinear yang
  dicoba di v71 overfit (0.84850 vs core 0.85015), jadi stack tidak otomatis.
- Seleksi tetap memakai AUC + public-proxy bootstrap; tambahkan sensitivitas
  prevalence (10%, 13.6%, 20%, 25%) agar tidak bergantung pada satu asumsi.

## Fase 3 — sequence model (structural break berikutnya)

Dibangun hanya jika Fase 0–2 selesai dan masih ada ruang.

- Sequence bulanan 12–24 bulan per customer: repayment (due/late/unpaid/DPD),
  account (inflow/outflow/net/saldo/low-balance), e-commerce (spend/order/
  cancel/kategori). Semua dipotong pada `decision_datetime`.
- Model kecil GRU/Transformer (1–2 layer, hidden 64–128), 5-fold GroupKFold by
  customer, OOF cross-fit. Evaluasi AUC/KS dan korelasi terhadap MLP/LGBM.
- Label sequence hanya memakai outcome yang sudah terjadi sebelum cutoff;
  tidak ada target aplikasi sekarang.
- Jika sequence leg menambah ≥0.0005 AUC pada blend 5-fold, finalkan; jika
  tidak, dokumentasikan sebagai negative result dan berhenti.

## Fase 4 — panel versi survival (opsional, prioritas rendah)

Panel v71 memakai proksi “12 angsuran pertama”. Versi berikutnya hanya layak
jika waktu DPD 90 bisa direkonstruksi dari due date + 90 hari dan status
pembayaran; pinjaman yang belum mencapai 90 DPD dalam 12 bulan harus
diperlakukan sebagai censored, bukan negatif. Tanpa perbaikan timing ini,
panel tidak diulang.

## Fase 5 — seleksi dan dokumentasi

- Kandidat wajib: `core72` (MLP ensemble + tree), `anchor` (v66_04), dan satu
  blend konservatif. Maksimal 3 submit per iterasi.
- Semua CSV: 30.000 baris, ID unik/cocok test, [0,1], tanpa NaN.
- Notebook v72 dieksekusi penuh `RUN_TRAINING=False`, nol error.
- Update `agas/progress.md`, `agas/README.md`, Obsidian; commit lokal tanpa
  push dan tanpa submit tanpa instruksi.

## Ekspektasi

- Target realistis v72: **0.567–0.570** bila MLP benar-benar transfer ke
  public. 0.577 tetap ekor noise: perlu kenaikan OOF ~0.003–0.004 lagi atau
  public variance yang menguntungkan.
- Keputusan berhenti: jika dua iterasi berturut-turut (v71 dan v72) gagal
  melewati 0.56424, akui plafon data dan pertahankan anchor, bukan menambah
  kandidat yang makin jauh dari anchor.
