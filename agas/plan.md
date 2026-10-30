# Rencana kerja `agas/` — v80 aktif

Dokumen ini adalah rencana eksekusi aktif. **Rencana yang sedang dijalankan
adalah v80, ada di bagian paling bawah file ini.** Rencana v71–v79 di atasnya
disimpan sebagai riwayat dan sudah selesai. Aturan kompetisi di
`agas/limit.md`, `agas/overview.md`, dan `agas/rules.md` tetap mengikat.

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

---

# Rencana v74 — perbaiki alat ukur, lalu kejar 0.578 (SUDAH DIEKSEKUSI)

> **Hasil 2026-10-26: ketiga gate gagal.** Fase 1 (heterogenitas fold) gain
> −0.00034 vs gate +0.005. Fase 2 (seleksi berbasis KS) `P(>anchor)` 0.702 vs
> gate 0.80, dan arah yang dipilihnya adalah MLP 30% yang sudah ditolak public
> dua kali. Fase 3 (portofolio) mati karena semua kandidat sekualitas punya
> ρ ≥ 0.9994 ke anchor, nilai harapan 8 tiket hanya +0.0007. Anchor `v66_04`
> tetap final. Detail di `agas/progress.md` entri 2026-10-26.

Ditulis 2026-10-26. Deadline notebook **30 Oktober 2026 23:59 WIB**.
Target user: public **0.57788** (skor tim teratas leaderboard). Anchor kita
`v66_04` = **0.56424**. Gap **+0.01364**.

## Mengapa v74 tidak mengulangi v67–v73

v67–v73 semuanya mencari fitur atau model baru, lalu memilih kandidat dengan
**OOF AUC/KS**. Delapan probe berturut-turut gagal. Kesimpulan v73 adalah
"plafon data". Ada penjelasan lain yang belum pernah diuji: **alat ukurnya
yang salah**, bukan modelnya.

Bukti yang mendasari v74 (`agas/v74/output/public_calibration_v74.json`,
dihitung 2026-10-26):

| Submission | OOF KS | Public | Offset |
|---|---:|---:|---:|
| `v61_alt` | 0.528624 | 0.56312 | +0.0345 |
| `v66_04` (anchor) | 0.528713 | 0.56424 | **+0.0355** |
| `v70_07` | 0.529397 | 0.56419 | +0.0348 |
| `v70_06` | 0.530232 | 0.56040 | +0.0302 |
| `v71 core71` | 0.532834 | 0.56111 | +0.0283 |
| `v72 seq05` | 0.528844 | 0.55968 | +0.0308 |
| `v70_01` | 0.528503 | 0.55801 | +0.0295 |

Dua fakta dari tabel itu:

1. **Offset selalu positif, 0.028–0.036.** Public selalu ~0.032 di atas OOF KS.
   Kalibrasi subsample membuktikan ini bukan bias ukuran sampel: bahkan pada
   n=750 baris, E[KS] subsample anchor hanya 0.5544 — masih jauh di bawah
   0.56424. Jadi offset ini **sistematis**, bukan keberuntungan undian.
2. **Urutan offset berbanding balik dengan OOF KS.** Kandidat dengan OOF KS
   tertinggi (`core71` 0.5328) justru punya offset terkecil (+0.0283). Korelasi
   OOF KS vs public pada 7 titik ini **negatif**.

Fakta 2 adalah inti v74. Kalau OOF KS naik tetapi public turun secara
konsisten, maka OOF KS **bukan estimator** dari public KS — ia estimator dari
sesuatu yang lain, dan setiap seleksi argmax di atasnya adalah seleksi ke arah
yang salah. Delapan kegagalan bukan plafon data; itu delapan kali memilih
dengan kompas yang terbalik.

Hipotesis mekanismenya: **heterogenitas fold**. OOF adalah gabungan prediksi
dari 10 model fold yang berbeda; tiap model punya skala skor sendiri. KS adalah
statistik berbasis ranking global, jadi perbedaan skala antar-fold mengaburkan
ranking dan menekan OOF KS. Prediksi test tidak punya masalah ini: semua 30.000
baris diskor oleh ensemble yang sama. Model yang lebih fleksibel (MLP, fitur
`extra`) punya variasi antar-fold lebih besar, jadi OOF-nya **terbantu** oleh
cara pengukuran yang salah ini sementara test-nya tidak — tepat pola yang
terlihat di tabel.

## Prinsip (tidak berubah)

- Python saja; hanya data kompetisi di `data/`. Tanpa external data, AutoML,
  AutoFE.
- Tanpa leakage: semua fitur memakai cutoff `decision_datetime`.
- Artefak baru hanya di `agas/v74/{training,notebook,output}`; `features.py`
  tidak diubah.
- Notebook reproducible, dijalankan penuh dengan `RUN_TRAINING=False`.
- Tidak push GitHub. Tidak submit Kaggle tanpa instruksi eksplisit.
- Semua CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.
- Dokumentasi ke `agas/progress.md`, `agas/README.md`, dan Obsidian
  `coding-vault` note `Data Science Competition Logika UI.md`.

## Fase 0 — kalibrasi public (SELESAI)

`agas/v74/training/calibrate_public_v74.py`. Mengukur E[KS] dan sd[KS] anchor
pada 12 ukuran subsample, 600 draw per ukuran. Output: kurva bias/sd, offset
yang tidak bisa dijelaskan bias sampel, dan nilai `E[max]` untuk k submission
independen. Status: pass, tersimpan di
`agas/v74/output/public_calibration_v74.json`.

## Fase 1 — perbaiki estimator (lever utama)

`agas/v74/training/foldnorm_v74.py`.

1. Satu run LightGBM `lean` 10-fold × 1 fold-seed, **menyimpan prediksi dan
   indeks per fold** (v66–v73 tidak pernah menyimpan ini).
2. Hitung tiga estimator pada vektor OOF yang sama:
   - `ks_pooled` — cara v57–v73;
   - `ks_foldnorm` — rank01 per fold dulu, baru digabung;
   - `ks_perfold` — rata-rata KS di dalam tiap fold.
3. Gate H1: kalau `ks_foldnorm − ks_pooled ≥ 0.005`, heterogenitas fold
   terbukti sebagai penyebab offset, dan estimator yang dipakai v74 adalah
   `ks_foldnorm`.
4. Kalau terbukti: **re-ranking semua kandidat yang sudah ada** (`preds_v66`,
   `preds_v68`, `preds_v70`, `preds_v71`, `preds_v72`) di bawah estimator baru,
   tanpa training apa pun. Kandidat yang sudah pernah dibangun tetapi dibuang
   karena OOF-nya rendah bisa saja justru yang terbaik.
5. Kalau gate gagal (`< 0.005`): H1 ditolak, catat sebagai negative result,
   lanjut ke Fase 2 dengan estimator bootstrap-mean.

## Fase 2 — optimasi ke metrik, bukan ke AUC

KS = `max_t |TPR − FPR|`: **satu titik** di kurva ROC. v57–v73 semuanya
mengoptimasi AUC, yaitu integral seluruh kurva. Dua kandidat dengan AUC sama
bisa punya KS berbeda jauh tergantung bentuk ROC di sekitar titik maksimum.

1. Ukur di mana ambang KS berada pada OOF anchor (kuantil skor, prevalensi
   lokal, lebar plateau).
2. Latih ulang leg `lean` dengan **sample weight terkonsentrasi di band itu**
   (3 level bobot), 10-fold × 1 seed, estimator Fase 1.
3. Cari bobot blend yang memaksimalkan **bootstrap-mean KS** (1.000 resample),
   bukan argmax KS dan bukan AUC.
4. Gate: mean gain ≥ **+0.002** KS dan `P(> anchor) ≥ 0.80`. Gagal → negative
   result, anchor tetap.

## Fase 3 — portofolio kandidat terdekorelasi

Pelajaran dari 8 kegagalan: kandidat yang disubmit selalu salah satu dari dua
ekstrem — duplikat anchor (ρ ≥ 0.999, tidak menambah undian baru) atau keluarga
model lain yang kualitas aslinya lebih rendah. Tidak pernah ada set kandidat
**sekualitas tapi terdekorelasi**.

- Bangun 3–4 kandidat dengan estimator Fase 1 setara anchor (selisih ≤ 0.001)
  tetapi Spearman antar-kandidat ≤ 0.995.
- Laporkan `E[max]` public untuk k submission dari kurva Fase 0, supaya user
  tahu nilai tiap slot submit yang tersisa sebelum 30 Oktober.
- Urutan submit direkomendasikan eksplisit, kandidat terkuat lebih dulu.

## Fase 4 — deliverable

- `agas/v74/notebook/notebook_v74.ipynb`: satu notebook bersih end-to-end —
  data, fitur, training (flag `RUN_TRAINING`), estimator, kandidat, validasi
  submission. Dieksekusi penuh, nol error.
- `agas/v74/output/`: submission CSV tervalidasi + `summary_v74_candidates.json`
  + log.
- Update `progress.md`, `README.md`, `plan.md`, Obsidian. Commit lokal.

## Ekspektasi jujur

- Kalau H1 terbukti dan re-ranking menemukan kandidat lain: realistis
  **0.566–0.572**, dengan peluang nyata menyentuh 0.578 karena yang berubah
  adalah arah seleksi, bukan ukuran gain.
- Kalau H1 ditolak: plafon v73 benar, hasil realistis **0.564–0.566**, dan
  nilai v74 adalah portofolio Fase 3 + paket notebook bersih.
- Yang tidak akan dikejar: gain OOF di bawah 0.001 yang sudah terbukti tidak
  transfer, dan model baru tanpa gate.

## Gate berhenti

Berhenti dan finalkan anchor kalau Fase 1 dan Fase 2 dua-duanya gagal gate.
Tidak ada probe kesembilan tanpa estimator yang terbukti.

---

# Rencana v75 — model yang benar-benar lebih kuat dari v66_04 (SUDAH DIEKSEKUSI)

> **Hasil 2026-10-26: empat hipotesis, empat gagal.** H4 learning curve masih
> menanjak tapi 20% data terakhir hanya bernilai +0.0007 AUC. H3 monotonicity
> dAUC −0.00078, negatif di ketiga seed. H5 compact/additif: baseline 490 tetap
> terbaik, `gam` −0.0016. H6 model undian public **dipalsukan** oleh kalibrasi
> ke tiga hasil nyata — `0.56424` adalah maksimum undian yang sudah diambil,
> rata-rata keluarga ini `0.5596`, sehingga P(≥0.57) di bawah 1% per tiket.
> Anchor `v66_04` tetap final. Detail di `agas/progress.md` entri 2026-10-26.

Ditulis 2026-10-26, setelah v74. User meminta eksplisit: **model yang lebih
kuat dari `v66_04`, public lebih tinggi, bukan file yang sama**. Permintaan itu
dijalankan penuh. Yang tidak diulangi adalah metodenya: v67–v74 selalu
menambah fitur/model lalu berharap public naik, delapan kali gagal.

## Apa yang v74 sudah pastikan

- Selisih OOF di bawah ~0.001 tidak terukur: sd[KS] 0.0208 pada n=3.000, dan
  KS model yang sama bervariasi 0.0158–0.0195 antar fold n=7.000.
- Karena itu v75 **hanya menerima lever yang menaikkan AUC ≥ 0.0015** dengan
  tanda yang konsisten di 3 seed. Apa pun di bawah itu adalah noise dan tidak
  akan disubmit.

## Diagnosis yang mendasari v75

Ablation per-sumber v70 dibaca ulang. `app_profile` di sana adalah **ember
sisa** (~180 fitur lintas-sumber), bukan blok aplikasi murni, jadi AUC 0.831
itu tidak berarti "terms pinjaman membawa semua sinyal". Diukur langsung,
17 kolom aplikasi + profil mentah hanya memberi **AUC 0.6529**.

Struktur data yang relevan:

- `term_months` **selalu 12**; `annual_interest_rate` hanya punya **5 nilai**
  (0.16–0.32); `proposed_monthly_installment` adalah anuitas eksak dari amount,
  rate, dan 12 bulan — rasio installment terhadap formula anuitas = 1.000000
  dengan sd 0.000000. Jadi tidak ada informasi tersembunyi di terms pinjaman.
- Leg `linear` (spline logistic) mencapai AUC **0.8428** melawan **0.8484** leg
  boosted terbaik. Pada data kredit nyata jarak itu jauh lebih lebar. Jaraknya
  yang sempit adalah bukti bahwa respons mendekati **additif dan monoton** pada
  tiap fitur risiko.

## Hipotesis v75

**H3 — monotonicity.** Kalau respons memang monoton pada tiap fitur risiko,
maka memaksakan monotonisitas pada LightGBM membuang varians tanpa menambah
bias: argumen scorecard kredit yang standar. Pohon bebas menghabiskan kapasitas
untuk melawan noise dengan split non-monoton yang tidak ada di generator.
**Belum pernah dicoba di 18 versi.**

Arah constraint diturunkan **di dalam tiap training fold** dari tanda Spearman
fitur–label (`RHO_MIN` 0.015); fitur yang lebih lemah dari itu dan semua
kategorikal dibiarkan bebas. Tidak ada fold yang melihat labelnya sendiri.

**H4 — plafon informasi.** Learning curve pada 20/40/60/80/100 persen baris
training menjawab apakah plateau 0.850 itu batas data atau batas model. Kalau
AUC masih naik di 100 persen, masalahnya varians dan H3 relevan. Kalau datar,
datanya habis dan tidak ada feature engineering yang akan menggerakkannya.

## Fase A — probe (gate sebelum membangun apa pun)

`agas/v75/training/probe_v75.py`. Learning curve + perbandingan monotone vs
bebas pada 5 fold × 3 seed, config cepat (lr 0.03) karena yang dibandingkan
adalah bentuk, bukan angka absolut.

Gate A: **mean dAUC ≥ +0.0010 dan positif di ketiga seed**. Gagal → H3 ditolak,
catat sebagai negative result, dan tidak ada leg baru yang dibangun.

## Fase B — leg monoton penuh

Hanya kalau Fase A lolos. Latih leg monoton dengan config produksi v66
(lr 0.01, 10 fold, multi fold-seed, multi booster seed) pada kolom `lean` dan
`full`, lalu:

1. uji sebagai leg mandiri terhadap `lgb_lean` dan `lgb_full`;
2. masukkan ke rantai blend v61/v66 yang sudah terbukti, menggantikan atau
   menemani leg bebas yang sepadan;
3. Gate B: AUC blend naik ≥ **0.0015** terhadap anchor pada 5 fold × 3 seed,
   dan `P(>anchor)` bootstrap ≥ **0.80** dengan validasi out-of-selection
   separuh-separuh seperti v74 Fase 2.

## Fase C — kandidat dan submission

- Kandidat final dibangun dari rantai yang sama (`trees4 → core → tm30 →
  ext0.5`) supaya satu-satunya yang berubah adalah kualitas leg, bukan resep.
- Validasi CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.
- Kandidat hanya dipromosikan kalau Gate B lolos. Kalau tidak, anchor tetap dan
  itu dilaporkan apa adanya.

## Fase D — dokumentasi

- `agas/v75/notebook/notebook_v75.ipynb` end-to-end, `RUN_TRAINING` flag,
  dieksekusi penuh nol error.
- Update `agas/progress.md`, `agas/README.md`, `agas/plan.md`, dan Obsidian
  `coding-vault` note. Commit lokal tanpa push.

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
  Monotone constraint adalah spesifikasi model yang ditulis tangan, bukan
  pencarian otomatis; arah constraint berasal dari satu statistik yang
  didefinisikan eksplisit.
- Tanpa leakage; semua fitur memakai cutoff `decision_datetime`.
- `features.py` tidak diubah.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.

---

# Rencana v76 — leaderboard sebagai alat ukur, bukan undian (SUDAH DIEKSEKUSI)

> **Hasil 2026-10-26: 8 kandidat sweep dibangun dan divalidasi, notebook
> dieksekusi nol error.** Belum ada submission v76 yang dikirim. Langkah
> berikutnya ada pada user: kirim `submission_v76_01_core3to1.csv` lebih dulu,
> lalu terapkan aturan go/no-go di Fase 0. Detail di `agas/progress.md` entri
> 2026-10-26.

Ditulis 2026-10-26 setelah dua hasil public v75 masuk. User meminta cara lain
menaikkan skor public untuk mengejar leaderboard.

## Hasil baru yang mengubah segalanya

| submission | public |
|---|---:|
| `submission_v75_03_mix80_mlp.csv` | 0.56073 |
| `submission_v75_04_pair_mlp_v70_extra.csv` | 0.56045 |

Kalibrasi v75 meramalkan E[public] = **0.5596** untuk kedua file ini. Aktualnya
0.56073 dan 0.56045 — **prediksinya tepat**, menyimpang +0.0011 dan +0.0008,
jauh di dalam sebaran. Model keluarga itu terkonfirmasi.

Dengan 8 observasi, ledger public terbelah dua cluster yang terpisah tajam:

| Cluster | isi | n | mean | sd |
|---|---|---:|---:|---:|
| **A** | rantai persis `ext0.5 \| tm30 \| lgb_lean` | 3 | **0.56418** | **0.00007** |
| **B** | bukan rantai itu | 5 | 0.56000 | 0.00123 |

Cluster A: `v66_04` 0.56424, `v70_07` 0.56419, `v68_04` 0.56411.
Cluster B: `core71` 0.56111, `mix80_mlp` 0.56073, `pair_mlp` 0.56045,
`seq05` 0.55968, `v70_01` 0.55801.

Dua fakta yang keluar dari situ:

1. **Keunggulan +0.00418 cluster A nyata, bukan keberuntungan.** Ketiga
   anggotanya dilatih dengan feature set dan jumlah seed yang berbeda — v66
   enam fold-seed, v68 sepuluh, v70_07 fitur v70 sama sekali lain — tetapi
   mendarat dalam rentang 0.00013. Maksimum dari 5 undian cluster B hanya akan
   0.56142, jadi 0.56424 tidak bisa dijelaskan sebagai undian terbaik.
2. **Di dalam rantai itu, leaderboard adalah alat ukur ber-sd 0.00007.**
   Ini pertama kalinya di seluruh proyek public bisa dipakai sebagai
   instrumen presisi, bukan sumber noise. Selisih di atas 0.0002 adalah sinyal.

## Apa yang belum pernah diukur

Kualitas leg sudah terbukti **tidak** menggerakkan cluster A: v70_07 memakai
fitur yang sama sekali berbeda dan v68 memakai sepuluh fold-seed, dua-duanya
mendarat di 0.5641–0.5642. Jadi public KS rantai ini jenuh pada level leg.

Yang tersisa adalah **parameter rantainya sendiri**, dan hanya satu nilai yang
pernah diukur di public:

- rasio ekstrapolasi `r` = 0.5 (satu-satunya yang pernah disubmit dengan
  `tm30` + `lgb_lean` pada leg kuat);
- bobot linear `tm` = 0.30;
- bobot core `trees4 : lgbx` = 2 : 1;
- leg referensi ekstrapolasi = `lgb_lean`.

v61 dan v66 memang menyapu `r` dan `tm`, tetapi memilih dengan **OOF AUC** —
kunci yang v74 dan v75 buktikan tidak berkorelasi dengan public. Jadi sumbu ini
secara efektif belum pernah diukur pada metrik yang menentukan.

Cluster C memberi petunjuk arah: `qc_00` (ext1.5, ref `fac_full`) 0.56171,
`v58ext` (ext1.5, ref `fac`) 0.55900, `v66_02` (plateau tm20, `fac_full`)
0.55877. Mengganti leg referensi dari `lgb_lean` mahal. Jadi `ref=lgb_lean`
dipertahankan di seluruh sweep.

## Hipotesis

**H7 — fragility vs gradient.** Dua kemungkinan, dan satu submission bisa
memisahkannya.

- *Gradient*: public KS adalah fungsi mulus dari parameter rantai, dan 0.5642
  adalah nilai di `r=0.5`. Kalau begitu menggeser `r` memberi gradien terukur
  dan optimumnya mungkin bukan di 0.5.
- *Fragility*: 0.5642 adalah maksimum lokal yang rapuh; gangguan sekecil apa
  pun menjatuhkannya ke ~0.560. Bukti pendukung: `seq05` dengan ρ 0.99950 turun
  0.00456.

## Fase 0 — kandidat kontrol yang memisahkan dua hipotesis

Gangguan terkecil yang bisa dibuat di dalam rantai: ubah bobot core dari
2:1 menjadi 3:1. ρ terhadap anchor = **0.999986**, jauh lebih dekat daripada
`seq05`.

- Kalau file ini tetap ~0.5641 → hipotesis *gradient*, sweep diteruskan.
- Kalau jatuh ke ~0.560 → hipotesis *fragility*, **sweep dihentikan**, anchor
  dibekukan, dan tidak ada submission lanjutan yang dibuang.

Ini submission paling informatif yang tersedia dan harus dikirim lebih dulu.

## Fase 1 — sweep satu-sumbu (hanya kalau Fase 0 lolos)

Semua memakai `ref=lgb_lean`, legs v66 yang sama, satu parameter diubah per
kandidat:

| langkah | ρ ke anchor | sumbu |
|---|---:|---|
| `core3to1` | 0.999986 | bobot core (kontrol Fase 0) |
| `r0.75` | 0.999803 | rasio ekstrapolasi, naik |
| `r0.25` | 0.999803 | rasio ekstrapolasi, turun |
| `tm0.40` | 0.999425 | bobot linear, naik |
| `tm0.20` | 0.999425 | bobot linear, turun |
| `r1.00` | 0.999215 | ekstrapolasi, langkah besar |

Urutan submit dirancang untuk informasi, bukan untuk harapan: kontrol dulu,
lalu dua titik mengapit `r=0.5` supaya gradien dan kelengkungan langsung
terbaca pada sd 0.00007.

## Fase 2 — artefak

- `agas/v76/training/make_submission_v76.py` — ledger, statistik cluster,
  sweep, urutan submit, aturan go/no-go.
- `agas/v76/notebook/notebook_v76.ipynb` — end-to-end, `RUN_TRAINING` flag,
  dieksekusi penuh nol error.
- `agas/v76/output/` — CSV kandidat + `summary_v76.json`.
- Update `progress.md`, `README.md`, `plan.md`, Obsidian. Commit lokal.

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
- Tanpa leakage; semua fitur memakai cutoff `decision_datetime`.
- `features.py` tidak diubah; v76 tidak melatih model baru sama sekali, hanya
  merakit ulang leg yang sudah ada.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.
- CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.

## Ekspektasi jujur

- Fase 0 adalah tes hipotesis, bukan upaya menaikkan skor. Peluang terbesarnya
  adalah menghemat submission, bukan menambah skor.
- Kalau hipotesis *gradient* benar dan optimumnya bukan di `r=0.5`, kenaikan
  yang masuk akal **+0.001 sampai +0.004**, membawa public ke 0.565–0.568.
- Kalau *fragility* benar, v76 berhenti setelah satu submission dan anchor
  dibekukan. Itu hasil yang sah dan menghemat slot.
- 0.57788 tetap tidak dijanjikan oleh satu pun pengukuran yang ada.

---

# Rencana v77 — ikuti gradien yang baru ditemukan (SELESAI)

> **Status 2026-10-27: sumbu r ditutup.** `r=1.00` = 0.56464; kuadratik tiga
> titik berpuncak di r=0.7864 nilai 0.565217, gain tersisa +0.000017. Netto
> sumbu ini +0.00102. Dilanjutkan oleh rencana v78 di bawah.
>
> **Status 2026-10-26:** Fase 2 selesai (hipotesis dipatahkan, leg tetap
> dipakai sebagai lever terpisah). Fase 1 menunggu satu submission:
> `agas/v77/output/submission_v77_r1p000.csv`. Final pick sudah berubah ke
> `agas/v76/output/submission_v76_02_r0.75.csv` (0.56520).

Ditulis 2026-10-26 setelah `submission_v76_02_r0.75.csv` = **0.56520**.

## Terobosan: kandidat pertama yang melewati anchor

| kandidat | public | vs anchor |
|---|---:|---:|
| cluster A (`r=0.5`), 3 pengukuran | 0.56418 (sd 0.00007) | — |
| **`r=0.75`** | **0.56520** | **+0.00102 = 14.6 sd** |

Dalam 20 versi, ini kandidat pertama yang mengalahkan 0.56424. Dan karena
`r=0.75` ada pada ρ 0.999803 — gangguan yang lebih besar daripada `seq05`
(ρ 0.99950) yang turun 0.00456 — hasil ini sekaligus **menolak hipotesis
fragility** dan **mengonfirmasi hipotesis gradient** dari v76 Fase 0.

## Mengapa naik: mekanismenya aljabar, bukan tebakan

Ekstrapolasi rasio `r` identik dengan kontras berbobot:

```
base + r·(base − lgb_lean)  ≡  base − w·lgb_lean ,   w = r/(1+r)
```

Terverifikasi numerik untuk r ∈ {0.25 … 3.0}: beda maksimum 3.3e-5, yaitu
tepat satu posisi rank dari tie-breaking.

Bobot neto `lgb_lean` (ia juga ada di dalam `base` dengan bobot
`0.7 × 2/3 × 1/4 = 0.11667`):

| r | w | bobot neto `lgb_lean` |
|---:|---:|---:|
| 0.00 | 0.000 | +0.11667 |
| 0.50 | 0.333 | −0.21667 |
| **0.75** | 0.429 | **−0.31190** |
| 1.00 | 0.500 | −0.38333 |
| 1.50 | 0.600 | −0.48333 |
| 2.00 | 0.667 | −0.55000 |

**Interpretasi pertama saya SALAH dan sudah ditarik.** Saya menduga
`base − w·lgb_lean` adalah kontras kolom-penuh minus kolom-lean, sehingga
menaikkan `w` memperkuat 108 kolom yang `lean` buang (demografi, `gn_`,
`ec_share_`, `buh_`, `cs_shift`, `ec_catqty_`). Dugaan itu **diuji dan
dipatahkan**: leg yang dilatih khusus pada 108 kolom itu (OOF AUC 0.836393)
bergerak ke arah yang **berlawanan** dari menaikkan `r`.

| arah "naikkan r" vs | Pearson |
|---|---:|
| tambah 20% leg `dropped` | **−0.68** |
| tambah 20% `lgb_lean` | **−0.9959** |
| tambah 20% leg apa pun | −0.75 … −0.996 |

Menaikkan `r` **mengurangkan**, bukan menambahkan, jadi arahnya antikorelasi
dengan penambahan leg apa pun, dan paling antikorelasi dengan `lgb_lean`
sendiri — persis yang aljabar katakan.

**Mekanisme yang bertahan setelah uji itu.** Karena `lgb_lean` berkorelasi
~0.99 dengan `base`, mengurangkan kelipatan besar darinya **memperbesar residu
kecil** di antara keduanya. Jadi menaikkan `r` adalah **de-shrinkage**: ia
menjauhkan ranking dari konsensus sepanjang arah ketidaksepakatan
`base − lgb_lean`, dan dengan itu mengurutkan ulang baris-baris yang di
konsensus nyaris seri. Public memberi imbalan pada itu — satu kali, sejauh ini.
Pernyataan ini sengaja berhenti pada apa yang terukur; mekanisme substantif
mengapa residu itu membawa sinyal belum terbukti.

## OOF anti-korelasi di sumbu ini

| kandidat | OOF AUC | dAUC | dKS | public |
|---|---:|---:|---:|---:|
| `r=0.5` | 0.846840 | — | — | 0.56418 |
| `r=0.75` | 0.846490 | **−0.000350** | **−0.000920** | **0.56520** |
| `r=1.00` | 0.846034 | −0.000806 | −0.001842 | belum |

OOF turun, public naik. Itu menjelaskan mengapa v61 dan v66 memilih `r=0.5`:
keduanya memilih dengan argmax OOF AUC, dan di sumbu ini OOF menunjuk ke arah
yang **salah**. Konsekuensi langsung: `r=1.00` punya OOF lebih buruk lagi, jadi
kalau anti-korelasinya bertahan, public-nya lebih tinggi.

## Fase 1 — line search pada r (prioritas utama)

Sudah ada dua titik. Langkah berikutnya satu submission sekaligus, karena
sd instrumen 0.00007 membuat tiap titik hampir bebas noise.

| langkah | file | prediksi terdaftar |
|---:|---|---|
| 1 | `submission_v76_06_r1.00.csv` (sudah ada) | kalau gradien linear lanjut: **0.5660–0.5665** |
| 2 | `r=1.25` atau `r=0.875` | tergantung hasil langkah 1 |
| 3 | `r=1.50` | hanya kalau r=1.25 masih naik |

Aturan keputusan setelah `r=1.00`:

- **> 0.56520** → gradien lanjut, kirim `r=1.25`, lalu `r=1.50`.
- **0.56420–0.56520** → optimum ada antara 0.75 dan 1.00; kirim `r=0.875`.
- **< 0.56420** → optimum dekat 0.75; kirim `r=0.625` untuk mengapit.

## Fase 2 — SELESAI: interpretasi diuji, dipatahkan, legnya tetap berguna

`agas/v77/training/train_dropped_v77.py` sudah dijalankan: leg LightGBM pada
108 kolom buangan, 10 fold × 2 fold-seed, 5.5 menit. **OOF AUC 0.836393,
KS 0.506835** — kuat untuk 108 kolom, dan didominasi `gn_` (66 dari 108), bukan
demografi mentah seperti dugaan saya.

Uji arah memastikan leg ini **bukan** versi bersih dari menaikkan `r`
(Pearson −0.68). Jadi ia bukan penjelasan, melainkan **lever terpisah**: leg
independen ber-AUC 0.836 yang belum pernah masuk rantai.

Kandidat `base + u·dropped` untuk u ∈ {0.10, 0.20, 0.35} dan kombinasinya
dengan `r` terbaik dibangun sebagai jalur kedua, diurutkan **setelah** line
search `r` selesai, karena `r` punya gradien terukur sementara jalur ini belum.

## Fase 3 — sumbu lain, setelah r selesai

`tm` dan bobot `core` belum diukur di public dan mungkin berinteraksi dengan
`r`. Diukur hanya setelah `r` punya optimum, satu sumbu per submission.

## Fase 4 — artefak

- `agas/v77/training/make_submission_v77.py` — sweep r diperluas + ledger.
- `agas/v77/notebook/notebook_v77.ipynb` — end-to-end, nol error.
- Update `progress.md`, `README.md`, `plan.md`, Obsidian. Commit lokal.

## Final pick

**Berubah.** Final pick Kaggle sekarang
`agas/v76/output/submission_v76_02_r0.75.csv` (0.56520), bukan lagi anchor
`v66_04` (0.56424). Ganti lagi hanya kalau ada titik sweep yang melewati
0.56520.

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
- Tanpa leakage; semua fitur memakai cutoff `decision_datetime`.
- `features.py` tidak diubah.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.
- CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.

## Ekspektasi jujur

- Gradien linear tidak akan berlanjut selamanya; permukaannya harus berbalik.
  `qc_00` pada r=1.5 dengan `ref=fac_full` hanya 0.56171, walaupun itu sumbu
  referensi yang berbeda jadi bukan pembanding langsung.
- Kalau optimumnya sekitar r=1.0–1.25, public yang masuk akal **0.566–0.567**.
  Itu masih di bawah 0.57788, tetapi untuk pertama kalinya ada sumbu dengan
  gradien terukur, bukan undian.
- Satu pengukuran tetap satu pengukuran. `r=1.00` adalah tes, dan kalau ia
  turun, +0.00102 di `r=0.75` harus dicurigai sebagai titik tunggal.

---

# Rencana v78 — sumbu referensi dan training panjang (AKTIF)

Ditulis 2026-10-27 setelah `r=1.00` = 0.56464 menutup sumbu ekstrapolasi.

## Posisi sekarang

| sumbu | status | hasil |
|---|---|---|
| ekstrapolasi `r` | **tertutup** | +0.00102, puncak r=0.7864 |
| leg referensi | **dibuka v78** | belum ada pengukuran public |
| `tm`, bobot `core` | dibangun, antre | belum ada pengukuran public |
| kualitas leg (training) | run panjang berjalan | OOF naik, public belum diuji |

## Fase 1 — sumbu referensi: DITUTUP (2026-10-27)

> `ref_lgb_full_matched` = **0.55934**, −0.00586 dari `r=0.75`. Pilihan
> `lgb_lean` menanggung beban. Diagnostik: efek ekstrapolasi hanya berkorelasi
> **+0.33** dengan kontras kolom full-minus-lean, jadi pembacaan berbasis kolom
> tetap tertolak; dan `base − lgb_full` 81% sejalan dengan arah yang menang
> namun tetap kalah 0.0059, jadi permukaannya peka pada residu spesifik bukan
> arah umum. Empat referensi sisa sekelas dengan yang gagal; dua yang jauh
> tanpa dasar struktural. **Lanjut ke Fase 3 (`tm`, `core`), lalu Fase 2.**

## Fase 1 — sumbu referensi (rencana awal, sudah dieksekusi)

Satu submission per langkah, kekuatan pengurangan disamakan di −0.31190:

1. `submission_v78_ref_lgb_full_matched.csv` — kolom berbeda, keluarga sama
2. `submission_v78_ref_fac_full_matched.csv` — leg di luar `base`
3. `submission_v78_ref_xgb_full_matched.csv` — keluarga model berbeda

**Catatan dedup:** untuk anggota `trees4` (`lgb_full`, `cat_full`, `xgb_full`)
bobot di dalam `base` sama dengan `lgb_lean`, jadi matched `r` = 0.749985 dan
file `_matched` praktis identik dengan `_r075` (beda 295–549 rank dari 30.000).
Kirim salah satu saja. Kerangka matched hanya bermakna untuk `fac_full`,
`lgbx_full`, `linear`, `dropped`, `trees4`.

Aturan: kalau ada yang melewati 0.56520, sumbu ini hidup dan `r` optimumnya
perlu dipetakan ulang **untuk referensi itu** (tiap referensi punya kelengkungan
sendiri). Kalau ketiganya di bawah, `lgb_lean` memang pilihan yang benar dan
sumbu ini ditutup.

## Fase 2 — rantai dari leg hasil training panjang

Setelah run 10 fold × 6 fold-seed × 3 booster seed selesai: bangun rantai pada
`r = 0.75` dengan leg baru, uji **sekali** di public. Perbandingan OOF v76b
(`hybrid` −0.000088) tidak bisa memutuskan ini, karena OOF tidak melacak public
di wilayah ini.

## Fase 3 — `tm` dan bobot `core`

Hanya setelah Fase 1 dan 2 selesai. Kandidat sudah dibangun.

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
- Tanpa leakage; semua fitur memakai cutoff `decision_datetime`.
- `features.py` tidak diubah.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.
- CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.
- **Tanpa gate OOF** di sumbu-sumbu ini: OOF terukur tidak melacak public.

## Ekspektasi jujur

Sumbu `r` memberi +0.00102 dan itu kenaikan nyata pertama dalam 21 versi.
Tidak ada alasan menduga sumbu referensi memberi lebih; ia hanya satu-satunya
yang belum diukur. 0.57788 tetap tidak dijanjikan oleh satu pun pengukuran.

---

# Rencana v79 — kepatuhan notebook, sumbu terakhir, lalu freeze

Ditulis 2026-10-28 setelah tiga hasil public masuk. Deadline notebook
**30 Oktober 2026, 23:59 WIB** — dua hari.

## Hasil baru

| submission | public | vs titik terbaik |
|---|---:|---:|
| `v78_ref_lgb_full_matched` | 0.55934 | −0.00586 |
| `v78_tm20_r075` | 0.56158 | −0.00362 |
| `v78legs_r075` (hasil training 8.7 jam) | 0.56159 | −0.00361 |

**Kualitas leg sekarang terukur merugikan.** Run 8.7 jam menaikkan tiga leg
+0.00075…+0.00083 AUC dengan A/B bit-identical di empat leg kontrol, dan
rantainya turun **−0.00361** di public. v76b menyimpulkan ini dari OOF dan saya
menolak kesimpulannya karena OOF tidak punya standing; ternyata kesimpulannya
benar, hanya alasannya yang lemah. Jalur kualitas leg **ditutup, dengan bukti
public**.

## Struktur yang terukur: puncak sempit

Biaya satu langkah dari titik terbaik (`r=0.75`, `ref=lgb_lean`, `tm=0.30`,
`core=2:1`, leg v66):

| perubahan | biaya |
|---|---:|
| `r` → 0.50 | −0.00096 |
| `r` → 1.00 | −0.00056 |
| `tm` → 0.20 | −0.00362 |
| `ref` → `lgb_full` | −0.00586 |
| leg ditingkatkan +0.0008 AUC | −0.00361 |

Hanya `r` yang landai. Semua sumbu lain merugikan 0.0036–0.0059 untuk satu
langkah. Sepuluh dari sebelas probe pasca-`r` merugikan.

## Fase 0 — KEPATUHAN ATURAN (prioritas tertinggi)

`rules.md` bagian 7: *"The submitted notebook must correspond to the team's
competition approach and submission."*

Notebook committee di root (`DSC0152_kali ini beneran deh_Notebook.ipynb`)
mereproduksi `rank01(tm30 + 0.5*(tm30 − lgb_lean))` → public 0.56424. **Final
pick kita sekarang `r = 0.75` → 0.56520.** Notebook itu **tidak sesuai** dengan
submission final.

- `agas/v79/training/make_notebook_v79.py` membangun notebook committee baru
  yang mereproduksi **persis** `submission_v76_02_r0.75.csv`, diverifikasi nol
  rank berbeda dan hash cocok.
- Memuat: audit cutoff, fitur, model, rantai `r=0.75`, ledger public lengkap,
  tabel biaya per sumbu, validasi CSV.
- Dieksekusi penuh dengan `RUN_TRAINING=False`, nol error.
- Versi lama dicadangkan, tidak dihapus.

## Fase 1 — sumbu terakhir yang belum diukur

Satu submission per langkah. Semua dibangun dari prediksi leg v66 tersimpan.

1. `submission_v78_tm40_r075.csv` — mengapit `tm=0.30` dari atas. `tm=0.20`
   sudah merugikan 0.00362; kalau `tm=0.40` juga merugikan, `tm=0.30` adalah
   optimum interior dan sumbu ini tertutup.
2. `submission_v78_core3to1_r075.csv` — ρ 0.999980, langkah terkecil yang
   pernah dibuat. Belum pernah diukur.
3. **Keanggotaan `trees4`** — belum pernah dibangun sama sekali.
   `agas/v79/training/make_submission_v79.py` membangun varian yang membuang
   satu anggota atau menambah `fac_full`, pada `r=0.75` dengan `tm`, `core`,
   dan `ref` ditahan.

## Fase 2 — freeze

Gate berhenti: kalau Fase 1 tidak menghasilkan apa pun di atas 0.56520,
bekukan `submission_v76_02_r0.75.csv` sebagai final dan hentikan probe.

Dengan sepuluh dari sebelas probe pasca-`r` merugikan dan hanya sumbu `r` yang
memberi kenaikan, ekspektasi realistis Fase 1 adalah **netral sampai negatif**.
Nilainya adalah menutup sumbu dengan pengukuran, bukan dengan dugaan.

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
- Tanpa leakage; semua fitur memakai cutoff `decision_datetime`.
- `features.py` tidak diubah.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.
- CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.
- **Tanpa gate OOF**: terukur tidak melacak public di sumbu-sumbu ini.

## Ekspektasi jujur

0.56520 kemungkinan besar adalah titik terbaik yang bisa dicapai paket ini.
Prioritas dua hari terakhir adalah **melindungi** skor itu: notebook yang
sesuai aturan (10% nilai, dinilai dari reproducibility) bernilai lebih besar
daripada sisa sumbu yang ekspektasinya negatif.

---

# Rencana v80 — ikuti gradien core, lalu freeze (AKTIF)

Ditulis 2026-10-28. Deadline notebook **30 Oktober 2026, 23:59 WIB**.

## Status sumbu, tujuh pengukuran

| sumbu | status | bukti |
|---|---|---|
| `r` | **tertutup** | kuadratik, puncak 0.786, sisa +0.000017 |
| `tm` | **tertutup** | kuadratik, puncak 0.295, sisa +0.000009 |
| `ref` | **tertutup** | −0.00586 satu pengukuran |
| kualitas leg | **tertutup** | −0.00361, bukti public, A/B bit-identical |
| **`core`** | **satu sisi, gradien belum diikuti** | 3:1 → 2:1 naik +0.00226 |
| keanggotaan `trees4` | belum diukur | 4 kandidat siap |

## Fase 1 — `core` ke arah yang belum diuji

Submit `agas/v78/output/submission_v78_core1to1_r075.csv`.

- **> 0.56520** → gradien jalan: `v80_core0p75_r075`, lalu `core0p50`
- **0.56294–0.56520** → puncak antara 1 dan 2: `v80_core1p50_r075`
- **< 0.56294** → 2:1 optimum interior, sumbu tertutup

## Fase 2 — sumbu terakhir

`agas/v79/output/submission_v79_trees5_add_fac_full.csv` (keanggotaan `trees4`,
belum diuji 22 versi).

## Fase 3 — freeze

Kalau Fase 1 dan 2 tidak melewati 0.56520: bekukan
`agas/v76/output/submission_v76_02_r0.75.csv`, hentikan probe, dan pastikan
notebook committee tetap byte-identical dengannya (sudah, sha256
`35260db3…f07c733b`).

## Prinsip (tidak berubah)

- Python saja; hanya data di `data/`. Tanpa external data, AutoML, AutoFE.
- Tanpa leakage; cutoff `decision_datetime`. `features.py` tidak diubah.
- Tidak push GitHub, tidak submit Kaggle tanpa instruksi.
- CSV: 30.000 baris, ID unik dan cocok test, [0,1], tanpa NaN.
- Tanpa gate OOF: terukur anti-korelasi dengan public di sumbu `r` dan `core`.

## Ekspektasi jujur

Enam dari tujuh sumbu tertutup, dan dua yang memberi kenaikan (`r`, dan mungkin
`core`) adalah yang paling landai. Peluang `core 1:1` melewati 0.56520 ada
karena gradiennya terukur, tetapi besarnya kemungkinan kecil — ordo +0.001,
bukan +0.01. 0.57788 tetap tidak dijanjikan oleh satu pun pengukuran.

---

# Rencana v84 — tiga slot terakhir di sumbu identitas slot-2, notebook siap ganti (AKTIF)

Ditulis 2026-10-30 14:00 WIB. Deadline notebook **hari ini 23:59 WIB**.
Skor terbaik: `agas/v76/output/submission_v76_02_r0.75.csv` = **0.56524**
(public yang dilihat user). Sisa slot hari ini: **3**.

## Kenapa sumbu ini

Enam sumbu parameter (`r`, `tm`, `core`, `ref`, kualitas leg, komposisi) sudah
tertutup oleh pengukuran public. Satu-satunya pertanyaan struktural yang belum
pernah dijawab public adalah **leg mana yang menempati slot kedua `core`**
(bobot 1/3, dua kali satu leg `trees4`). v83 sudah membangun kandidatnya;
`cat_full` dan `xgb_full` adalah dua yang selaras dengan arah yang pernah naik
dan berlawanan dengan arah yang pernah turun.

## Pohon keputusan (3 slot)

Parameter slot: `second = (1 − a)·lgbx_full + a·LEG`; `a = 0` adalah final sekarang.

| slot | submit | lanjut |
|---|---|---|
| 1 | `v83/slot2_cat_full` (`a = 1`) | > 0.56524 → slot 2 = `cat a=1.5`; selain itu → slot 2 = `v83/slot2_xgb_full` |
| 2a | `v84/slot2_cat_a1p5` | slot 3 = puncak kuadratik `a ∈ {0, 1, 1.5}` (dibangun setelah skor) |
| 2b | `v83/slot2_xgb_full` | > 0.56524 → slot 3 = `xgb a=1.5`; selain itu → slot 3 = `v84/slot2_catxgb` (rata-rata keduanya) |

Semua kandidat cabang dibangun **sekarang** supaya slot tidak menunggu kode.

## Notebook

1. `agas/v84/training/switch_notebook_v84.py <csv>` — menambal notebook committee
   (konstanta `FINAL_CSV`, `SECOND`, `A_MIX`, skor, ledger), mengeksekusi ulang,
   dan meng-assert reproduksi 0 rank berbeda + byte-identical.
2. Dijalankan **sekarang dalam mode v76** untuk membuktikan pipeline tidak
   mengubah apa pun. Notebook lama dicadangkan di `agas/OtherTries/`.
3. Kalau final pick berubah: satu perintah, lalu audit `finalize_v81.py`.
4. **Batas aman: 22:30 WIB** — setelah itu final pick dibekukan apa pun hasilnya.

## Ekspektasi jujur

Ruang realistis **±0.001**. 0.57788 butuh +0.0126 dan tidak ada sumbu blend yang
punya ruang sebesar itu; mengejar itu dalam 10 jam dengan retrain penuh (6.5 jam)
mempertaruhkan notebook yang bernilai 10%.
