# Rencana kerja `agas/` — v74 aktif

Dokumen ini adalah rencana eksekusi aktif. **Rencana yang sedang dijalankan
adalah v74, ada di bagian paling bawah file ini.** Rencana v71–v73 di atasnya
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
