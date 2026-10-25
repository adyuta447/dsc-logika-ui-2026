# Rencana kerja `agas/` — v72 dan setelahnya

Dokumen ini adalah rencana eksekusi aktif. Rencana v72 sudah dijalankan dan
hasilnya ditulis di bawah rencana v71; rencana v73 untuk tahap berikutnya
ditulis tepat di bawah rencana v72. Aturan kompetisi di `agas/limit.md`,
`agas/overview.md`, dan `agas/rules.md` tetap mengikat.

## Status public yang melatarbelakangi

| Kandidat | OOF AUC | OOF KS | Public | Catatan |
|---|---:|---:|---:|---|
| `v66_04` | 0.846840 | 0.528713 | **0.56424** | anchor terbaik |
| `v68_04` | 0.847427 | — | 0.56411 | fold-seed ulang, rho 0.999997 |
| `v69_10` | 0.846292 | 0.528254 | 0.55989 | pivot `extra` gagal |
| `v70_06` | 0.847623 | 0.530232 | 0.56040 | OOF KS tertinggi, public turun |
| `v70_07` | 0.847249 | 0.529397 | 0.56419 | praktis sama dengan anchor |
| `v70_01` | 0.848369 | 0.528503 | 0.55801 | AUC tertinggi, public turun |
| `v71 core71` | 0.850147 | 0.532834 | 0.56111 | MLP-heavy, OOF gain tidak transfer |
| `v72 seq05` | 0.846968 | 0.528844 | 0.55968 | sequence probe, ditolak |

Kesimpulan: keluarga fitur/model v66–v72 sudah jenuh. Perbedaan OOF di bawah
~0.001 tidak memprediksi public; kandidat yang jauh dari rank anchor justru
turun. Target 0.577 adalah ~2.5 sd public di atas 0.56424 dan sebagian besar
kemungkinan merupakan ekor noise leaderboard. Karena itu v72 menguji
**representasi sequence** sebagai informasi baru, bukan menambah bobot MLP.
v72 juga gagal: tidak ada kandidat yang melewati anchor, sehingga v73 menjadi
tahap freeze dan audit final.

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
# Rencana v72 — sequence representation (SUDAH DIEKSEKUSI)

## Status

- Public v71 `core71` = **0.56111**, turun 0.00313 dari anchor `v66_04`
  (0.56424), meskipun OOF KS-nya +0.00412. Ekspansi MLP dihentikan.
- v72 dijalankan 2026-10-24 dan **gagal lolos gate**. Public
  `submission_v72_02_anchor_seq05.csv` = **0.55968** (2026-10-25), turun
  **−0.00456** dari anchor. OOF gain +0.00013 KS tidak transfer. Tidak ada
  kandidat v72 yang dipromosikan di atas anchor; v72 ditutup.
- Artefak lengkap ada di `agas/v72/{training,notebook,output}` dan di-ringkas
  di `agas/progress.md`, `agas/README.md`, serta Obsidian.

## Hipotesis

Model v66–v71 memadatkan setiap sumber menjadi agregat statis. Agregat itu
kehilangan **urutan dan transisi**: income shock bulan ini, keterlambatan bulan
berikutnya, perubahan pola belanja, dan status pinjaman aktif. Sequence 24 bulan
yang hanya memakai event sebelum `decision_datetime` seharusnya menangkap
informasi yang tidak dimiliki agregat.

## Fase 0 — data sequence (selesai)

- `agas/v72/training/build_sequences_v72.py` membangun tensor
  `100000 × 24 × 56` + 5 mask sumber:
  - account: inflow/outflow/income-like credit/net/balance/low-balance days;
  - repayment: due/paid/unpaid, jumlah pinjaman, late/late30/late90, DPD,
    payment ratio;
  - e-commerce: spend/order/cancel/basket/items/harga relatif/kategori;
  - bureau: score/utilization/inquiries/delinquency/active/outstanding;
  - previous applications: jumlah, approval/rejection/cancel, amount, rate,
    installment, decision lag.
- Audit cutoff: tidak ada due date, account month, transaksi e-commerce,
  bureau report, atau previous decision setelah `decision_datetime`.
- Output: `agas/v72/output/sequences_v72.npz`.

## Fase 1 — model sequence (selesai)

- `agas/v72/training/train_seq_v72.py`: GRU bidirectional 1 layer + attention
  pooling, 5-fold stratified, early stopping pada AUC validasi.
- Dua varian: sequence murni dan sequence + blok static v70 (770 kolom).
- Hasil 5-fold × 1 seed:

| Leg | OOF AUC | OOF KS | Rank corr vs anchor |
|---|---:|---:|---:|
| sequence murni | 0.783309 | 0.407868 | 0.7598 |
| sequence + static | 0.841527 | 0.514382 | 0.9494 |
| anchor `v66_04` | 0.846840 | 0.528713 | 1.0000 |

## Fase 2 — gate dan keputusan (selesai)

- Gate yang ditetapkan sebelum run: sequence harus menambah **≥0.0005 AUC**
  pada blend dan **tidak menurunkan KS**; kandidat final harus punya bukti
  yang lebih kuat dari noise public.
- `anchor + 5% sequence murni`: AUC 0.846968 (+0.000128), KS 0.528844
  (+0.000131); bootstrap dKS +0.000519, P(dKS>0)=0.676.
- `anchor + 15% sequence+static`: AUC 0.847708 (+0.000868), KS 0.528063
  (−0.000650); bootstrap dKS +0.000287, P(dKS>0)=0.585.
- `anchor + 25% sequence+static`: AUC 0.847959 (+0.001119), KS 0.528276
  (−0.000437).
- **Gate gagal.** Sequence murni membawa sinyal baru tetapi terlalu lemah;
  sequence+static masih di bawah MLP static 0.8484. Tidak ada kandidat yang
  layak menggantikan anchor.

## Fase 3 — artefak dan kandidat (selesai)

- Kandidat valid 30.000 baris:
  - `submission_v72_01_anchor.csv` — kontrol, direkomendasikan;
  - `submission_v72_02_anchor_seq05.csv` — diagnostik, tidak dipromosikan;
  - `submission_v72_03_anchor_seqst15.csv` — ditolak.
- `summary_v72_candidates.json` memuat AUC/KS, korelasi, bootstrap, dan
  keputusan.
- `notebook_v72.ipynb` dieksekusi penuh dengan `RUN_TRAINING=False`, nol error.
- Tidak ada submit Kaggle dan tidak ada push GitHub.

## Pelajaran

1. Menambah informasi baru tidak otomatis menaikkan KS; sequence yang lemah
   hanya menambah noise pada ranking.
2. Kenaikan OOF +0.004 pada v71 juga tidak transfer ke public. Public noise
   (~0.0053 sd) lebih besar dari sebagian besar gain yang bisa diukur.
3. Anchor `v66_04` tetap satu-satunya kandidat dengan bukti public berulang.

---

# Rencana v73 — final freeze, audit anchor, dan dokumentasi

## Status

- Public `submission_v72_02_anchor_seq05.csv` = **0.55968** (2026-10-25),
  turun **−0.00456** dari anchor `v66_04` (**0.56424**).
- OOF seq05 AUC 0.846968 / KS 0.528844, Spearman 0.999501 terhadap anchor;
  kenaikan OOF tidak transfer ke public.
- v72 ditutup sebagai **negative result**. Semua kandidat pasca-anchor gagal
  melewati 0.56424.
- v73 dijalankan 2026-10-25. Audit dan notebook final sudah selesai;
  dokumentasi dan commit lokal dilengkapi tanpa push dan tanpa submit.

## Keputusan yang dikunci

**Final = `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv`**
(public 0.56424). `seq05`, `core71`, dan seluruh blend sequence/MLP ditolak.
`0.57768` adalah skor peserta lain, bukan kandidat kita. Tidak ada probe model
baru; jika user meminta submit ulang, hanya exact anchor yang boleh dikirim.

## Prinsip

- Tidak ada keluarga model, feature engineering, atau blend baru.
- Tidak ada data eksternal, AutoML, atau AutoFE.
- Semua artefak baru di `agas/v73/{training,notebook,output}`; `features.py`
  tidak diubah.
- Tidak ada leakage; semua fitur memakai cutoff `decision_datetime`.
- Notebook reproducible dijalankan penuh dengan `RUN_TRAINING=False`.
- Tidak push GitHub dan tidak submit Kaggle tanpa instruksi eksplisit.
- Semua CSV 30.000 baris, ID unik/cocok test, [0,1], tanpa NaN.

## Fase 0 — catat hasil dan freeze (selesai)

- Catat public seq05 = 0.55968 dan keputusan freeze di `progress.md`,
  `README.md`, `plan.md`, dan Obsidian.
- Hentikan eksperimen yang hanya mengejar noise public.
- Jangan menghapus artefak v72; simpan sebagai bukti negative result.

## Fase 1 — audit anchor (selesai)

- `agas/v73/training/audit_final_v73.py` memeriksa:
  - 30.000 baris, ID unik dan set-nya sama dengan `data/test_applications.csv`;
  - range `[0,1]`, tanpa NaN/inf;
  - SHA-256
    `acfb8ea882a92a414e338e89801ad7602e3c6c0faa9917887f6e750f0853c5c9`;
  - rebuild rank dari `preds_v66.npz`: selisih maksimum `1.11e-16`,
    Spearman 1.0, nol rank berubah.
- `agas/v73/output/audit_v73.json` menyimpan manifest; status **pass**.
- `agas/v73/output/submission_v73_01_anchor.csv` adalah salinan exact
  byte-identical dari anchor.

Jika audit gagal, jangan retrain. Gunakan
`agas/v72/output/submission_v72_01_anchor.csv` yang byte-identical; fallback
terakhir hanya `v70_07` setelah audit ulang dan persetujuan user.

## Fase 2 — notebook final (selesai)

- `agas/v73/notebook/notebook_v73_final.ipynb` dibangun dari resep/artefak v66.
- `RUN_TRAINING=False`; training command tetap ditampilkan untuk provenance.
- Notebook memuat audit cutoff/leakage, validasi CSV, rebuild rank, OOF
  AUC/KS, ledger public, dan perbandingan dengan kontrol v72/v73.
- Dieksekusi penuh dengan kernel bersih: nol error, output `STATUS: PASS`.
- Nama final menunggu `ParticipantNumber` dan `TeamName`; format
  `ParticipantNumber_TeamName_Notebook.ipynb`.

## Fase 3 — protokol submit

- Pilihan final di Kaggle harus anchor `v66_04`, bukan seq05.
- Tidak ada submit baru pada tahap ini. Anchor sudah punya public 0.56424 dan
  kontrol `v72_01` juga 0.56424; submit ulang tidak menambah informasi.
- Jika user meminta satu submit baru, hanya exact anchor yang diizinkan.
- Submit notebook melalui mekanisme committee, bukan hanya CSV.

## Fase 4 — dokumentasi dan deadline

- Update `agas/progress.md`, `agas/README.md`, `agas/plan.md`, dan Obsidian
  `coding-vault` note `Data Science Competition Logika UI.md`.
- Commit lokal; tanpa push GitHub dan tanpa submit tanpa instruksi.
- Deadline notebook: **30 Oktober 2026, 23:59 WIB**; target upload internal
  29 Oktober 2026.
- Ekspektasi realistis: **0.564–0.565 public**. Keberhasilan v73 adalah
  menjaga anchor dan meninggalkan paket notebook yang bersih, bukan memaksa
  kenaikan yang tidak bisa direplikasi.

## Gate berhenti

Gate sudah terpenuhi: dua probe public berturut-turut (`core71`, `seq05`) tidak
melewati 0.56424, dan tidak ada kandidat baru dengan bukti OOF yang layak.
Berhenti melakukan eksperimen model; finalkan anchor dan dokumentasikan plafon
data.
