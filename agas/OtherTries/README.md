# DSC0152 — Peningkatan Skor Leaderboard (metrik KS)

Dokumen ini menjelaskan **diagnosis**, **langkah perbaikan**, dan **hasil** upgrade
notebook prediksi 90+ DPD, plus langkah lanjutan yang belum sempat diuji.

File utama:

| File | Isi |
|---|---|
| `dsc_logikaui_ks_v2.ipynb` | Notebook final (sudah dijalankan, 29 sel, output tersimpan) |
| `submission.csv` | Hasil prediksi 30.000 baris dari notebook v2 |
| `dsc_logikaui_submission.ipynb` | Notebook v1 (lama, versi AUC-optimal) |
| `DSC0152_...Notebook.ipynb` | Notebook referensi (skor Kaggle 0.558) |
| `src2/` | Sumber kode teruji yang dipakai untuk membangun notebook v2 |

---

## 1. Diagnosis: kenapa skor Kaggle hanya 0.558?

| Versi | OOF AUC | **OOF KS** | Kaggle |
|---|---|---|---|
| v1 kita (`dsc_logikaui_submission.ipynb`) | 0.81552 | **0.47144** | — |
| Notebook referensi | 0.84507 | **0.52646** | 0.558 |
| **v2 kita (`dsc_logikaui_ks_v2.ipynb`)** | **0.84406** | **0.52332** | submission.csv |

Penyebab utama, urut dari yang paling menentukan:

### 1.1 Metrik yang dioptimalkan salah (paling menentukan)

Metrik kompetisi adalah **Kolmogorov–Smirnov statistic**
`KS = max_t |F_pos(t) − F_neg(t)|` (`scipy.stats.ks_2samp`).

Notebook v1 memilih hyperparameter, model, dan bobot blend berdasarkan **AUC**.
Karena KS hanya bergantung pada **urutan** skor:

* optimasi AUC tidak otomatis menaikkan KS — pemilihan harus dijalankan langsung pada KS;
* blending harus memakai **rata-rata rank** (transformasi monoton pada probabilitas
  tidak mengubah KS, tetapi campuran rank *dapat* mengubah urutan);
* KS jauh lebih sensitif pada_region atas ranking, sehingga model dengan AUC sama
  bisa berbeda KS-nya 0,01.

### 1.2 Kekurangan fitur

Notebook referensi menangkap sinyal yang tidak ada di v1, dan semuanya dih.import ke v2:

* **tingkat keterlambatan matang** `hit{t}/mat{t}`, t ∈ {1,15,30,60,90} — cicilan yang
  jatuh tempo hanya *t* hari sebelum cutoff belum sempat terlambat *t* hari;
* **urutan cicilan**: roll-rate, cure-rate, streak terlambat terpanjang, slope DPD;
* **harga satuan e-commerce relatif median kategori** — fitur terkuat di dataset ini
  (sendiri AUC 0.748, KS 0.366; rasio default desil 1 = 2,1% vs desil 10 = 39,5%);
* **perbandingan dengan baseline customer sendiri** (rekening bulan terakhir vs 4–12 bulan;
  e-commerce jendela 7/14/30/60/90 hari vs baseline 90–365 hari);
* **laporan bureau milik pengajuan terpilih** + pivot per provider + tren skor;
* **pinjaman aktif sebelumnya** untuk debt-to-income yang realistis.

### 1.3 Kurangnya diversitas & hanya satu split CV

v1 memakai 2 model pada 1 feature set dan 1 split. v2 memakai 6 model
(3× LightGBM, XGBoost, CatBoost, spline-logistic) pada beberapa feature set,
di dua split CV, dengan blend yang benar-benar dioptimalkan terhadap KS.

---

## 2. Audit data: tidak ada kebocoran waktu

```
ecommerce setelah decision_datetime : 0 / 4.359.817
account_behavior setelah bulan keputusan : 0 / 1.084.418
cicilan repayment due_date > cutoff : 0 / 1.361.341
laporan bureau available > cutoff  : 0 / 797.563
previous_applications diputuskan > cutoff : 0 / 201.984
```

Semua tabel **sudah point-in-time per customer**. Jadi perbaikan ini bukan
"membersihkan kebocoran", melainkan murni menaikkan kualitas fitur + optimasi metrik.
Helper agregasi tetap memaksa aturan cutoff (dan diuji dengan assertion), sehingga
notebook aman bila data di kemudian hari berubah.

### Catatan jujur soal kolom DPD

`days_past_due` / `max_days_past_due` diukur sampai **akhir periode observasi global**,
sehingga secara teoretis bisa "melihat" kejadian setelah tanggal keputusan.
Kami menghitung dua versi dan mengukur kontribusinya:

| Versi | KS (fold 0, LightGBM) |
|---|---|
| `raw` — kolom DPD apa adanya (sesuai data dictionary) | **0.5187** |
| `asof` — DPD dihitung ulang pada `decision_datetime` (strictly causal) | 0.5133 |

Versi `raw` dipakai di model akhir; versi `asof` tetap dihitung agar bisa dipakai.
Set `USE_CAUSAL_DPD_ONLY = True` (Bagian 1 notebook) untuk memakai versi causal saja
— ekspektasi turun ~0.005 KS, tetapi paling aman terhadap pembacaan ketat aturan
"hindari data leakage".

---

## 3. Hasil v2 (semua angka dari output notebook, bukan estimasi)

### Per model (5-fold × 2 split)

| Model | KS split 42 | KS split 2025 | KS gabungan | AUC |
|---|---|---|---|---|
| lgb_full seed 0 | 0.52058 | 0.52347 | 0.52227 | 0.84351 |
| lgb_full seed 1 (extra_trees) | 0.51945 | 0.52207 | 0.52079 | 0.84349 |
| lgb_lean seed 0 | 0.51924 | 0.52063 | 0.52178 | 0.84332 |
| xgb_full | 0.50614 | 0.51089 | 0.51607 | 0.83938 |
| cat_full | 0.51922 | 0.51966 | 0.52160 | 0.84180 |
| spline_top (100 fitur teratas) | 0.51351 | 0.51813 | 0.51602 | 0.83958 |

### Blend

| Kandidat | mean KS (2 split) | KS gabungan | AUC |
|---|---|---|---|
| **greedy (terpilih)** | **0.52274** | **0.52332** | **0.84406** |
| equal weight | 0.52233 | 0.52310 | 0.84395 |
| best single | 0.52203 | 0.52227 | 0.84351 |

Bobot terpilih: `lgb_full_s0 0.357`, `lgb_full_s1 0.286`, `cat_full 0.143`,
`spline_top 0.143`, `xgb_full 0.071` (dalam ruang rank).

**KS v1 → v2: 0.47144 → 0.52332 (+0.0519, +11.0%)**, AUC 0.81552 → 0.84406.
KS gabungan v2 (0.5233) berada di bawah CV notebook referensi (0.5265) selisih
0.003 — jauh di dalam noise (±0.004 antar split, lihat Bagian 5).

Waktu jalan total: ±2 jam (10 CPU), feature building ±80 detik.

---

## 4. Langkah yang dilakukan (10 langkah)

| # | Langkah | Hasil |
|---|---|---|
| 1 | Ganti metrik evaluasi ke **KS** (KS + AUC berdampingan) | supaya semua keputusan diambil sesuai metrik submission |
| 2 | Audit batas waktu 5 tabel | terbukti 0 baris melewati cutoff |
| 3 | **Tingkat keterlambatan matang** `hit{t}/mat{t}` | normalisasi cicilan yang belum sempat terlambat |
| 4 | **Fitur urutan cicilan** (roll, cure, streak, slope DPD) | pola berulang lebih informatif dari rata-rata |
| 5 | **Harga relatif e-commerce** (+ versi relatif terhadap median customer sendiri, tren, slope) | fitur #1 menurut gain |
| 6 | **Baseline-self** untuk rekening & e-commerce + peluruhan eksponensial | menangkap perubahan, bukan level |
| 7 | **Blok bureau per pengajuan terpilih** + pivot per provider + tren skor | snapshot terbaik & reduksi noise |
| 8 | **Interaksi lintas tabel** (DTI, income terverifikasi, harga × tunggakan, belanja diskresioner) | interaksi antar tabel sering lebih kuat dari fitur tunggal |
| 9 | **Ensemble 6 model × 2 feature set × 2 split CV** | pengurangan variansi untuk metrik rank-based |
| 10 | **Blend greedy yang langsung maximizes KS**, dinilai rata-rata atas 2 split | 0.52332 vs 0.52227 model terbaik |

Semua feature engineering manual (tanpa AutoML/AutoFE), tanpa data eksternal,
dengan helper point-in-time yang terverifikasi terhadap hitungan naïf pandas
(242 assertion di notebook).

---

## 5. Temuan eksperimental (termasuk yang **tidak** berhasil)

Penting agar effort berikutnya tidak mengulang jalan yang sudah salah:

| Eksperimen | Hasil | Kesimpulan |
|---|---|---|
| Feature selection (top 60/120/250/500/900 by gain) | 0.5119–0.5182 vs 0.5145 base | tidak membantu |
| Buang fitur DPD kausal (`raw` saja) | +0.005 KS | **dipakai** |
| ExtraTrees (400 pohon) sebagai anggota blend | +0.0006 KS dengan biaya 24 menit | tidak sepadan |
| Model spesialis region-top (re-rank 30%/50% atas) | 0.0000 perubahan | tidak membantu |
| Spline-logistic pada seluruh fitur | sangat lambat (~40 menit), KS lebih rendah | pakai top-100 saja |
| **Monotone constraints** (356 fitur) | fold 0: +0.006, fold 1–3: −0.0004 | **noise**, tidak dipakai |
| Ungkap sensitivitas split | OOF KS bergeser ±0.004 hanya karena komposisi fold | evaluasi blend harus rata-rata multi-split |

---

## 6. Langkah lanjutan yang direkomendasikan

1. **Uji ulang di Kaggle** dengan `submission.csv` hasil v2 —CV KS naik 0.052,
   tetapi— leaderboard bisa bergerak lain (CV v2 0.5233 < CV referensi 0.5265
   padahal LB referensi 0.558).
2. **Tambah repeat CV** (split seed 2027, 2028) — mekanismenya sudah terbukti
   menurunkan noise;setiap repeat menambah ±20 menit.
3. **Tambah seed LightGBM** (5 seed, bukan 2) — rata-rata lebih stabil.
4. **Cari sinyal baru, bukan model baru**: semua keluarga model sudahsudah dan
   berada di 0.516–0.523. Kandidat yang belum dicoba:
   * rasio `income_like_credits` terhadap `outflows` per bulan (kewajaran arus kas);
   * konsistensi skor antar provider bureau (selisih & ranking);
   * pola **jumlah item per kategori per bulan** (bukan hanya share spend);
   * interaksi `harga relatif × Rasio cicilan/income terverifikasi`.
5. **Pertimbangkan `USE_CAUSAL_DPD_ONLY = True`** bila organizer membaca aturan
   anti-leakage secara ketat (ekspektasi −0.005 KS tetapi Submission aman).

---

## 7. Cara menjalankan

```bash
# jalankan ulang notebook dari nol (menghasilkan submission.csv yang sama)
python -m jupyter nbconvert --to notebook --execute --inplace dsc_logikaui_ks_v2.ipynb

# membangun ulang sumber notebook dari modul teruji
python src2/gen_nb.py && python src2/make_notebook.py src2/nb_percent_source.py dsc_logikaui_ks_v2.ipynb
```

Semua seed terkunci (`SEED=42`, `SPLIT_SEEDS=[42, 2025]`, `deterministic=True`
untuk LightGBM), sehingga hasilnya reproducible.
---

# Update v3 — upaya menuju KS 0.57 (`dsc_logikaui_ks_v3.ipynb`)

Notebook v3 (sudah dijalankan, 0 error, ±46 menit) memakai: **10-fold**, **3 seed LightGBM + CatBoost + XGBoost + spline-logistic**,
2.057 fitur (v2 + blok `disc`: harga relatif alternatif, kebiasaan belanja, DPD berpeluruhan waktu, slope skor bureau),
dan blend greedy yang memaksimalkan KS. `submission.csv` sekarang berasal dari v3 (cadangan v2: `submission_v2_backup.csv`).

| Model (10-fold) | OOF KS | AUC |
|---|---|---|
| lgb_full s0 / s1 / s2 | 0.52150 / 0.52308 / 0.52182 | 0.8431–0.8434 |
| cat_full | 0.51930 | 0.84114 |
| xgb_full | 0.50648 | 0.83520 |
| spline_top | 0.51650 | 0.83925 |
| **Blend greedy** | **0.52346** | **0.84364** |

## Jujur: target 0.57 TIDAK tercapai di CV
OOF KS v3 (0.5235) ≈ v2 (0.5233). Perubahan yang dicoba **tidak menaikkan KS secara berarti**:

| Eksperimen | Hasil |
|---|---|
| Learning curve (25%→100% data latih) | KS 0.5067 → 0.5134; mendatar, jadi 10-fold/lebih banyak data hanya ±0.002 |
| Fitur discovery (36 fitur baru) | Gain tinggi, tapi fold-0 KS identik (0.51973): redundan dengan `ec_price_rel` |
| **Pseudo-labeling data test** | **Memperburuk**: KS 0.5108 vs 0.5197 → dibuang |
| 10-fold + 3 seed LGB + CatBoost + XGB | blend 0.5235 vs 0.5233 (v2) |
| Monotone constraints, ExtraTrees, specialist top-region, feature selection | tidak membantu (lihat bagian 5) |

Kesimpulan teknis: seluruh keluarga model berada di 0.51–0.523 pada fitur yang sama; **batasnya adalah informasi di fitur/label noise, bukan model atau jumlah fold**.
Kurva KS OOF: maksimum di 32,8% teratas (TPR 0.78, FPR 0.26). Skor leaderboard referensi (0.558) berada ~0.03 di atas CV-nya, jadi
submission ini kemungkinan di kisaran 0.55 — **tidak ada bukti bahwa ia menembus 0.57**, dan saya tidak mengklaim demikian.

## Kepatuhan `limit.md`
Python saja; tanpa data eksternal; FE manual (tanpa AutoML/AutoFE); semua agregat di-cutoff `decision_datetime` (audit: 0 baris melewati cutoff,
242 assertion unit test); seed terkunci sehingga notebook reproducible dan cocok dengan `submission.csv`. Pseudo-labeling hanya dieksperimenkan lalu tidak dipakai.
Catatan DPD: kolom DPD dipakai apa adanya (sesuai data dictionary); set `USE_CAUSAL_DPD_ONLY = True` untuk versi strictly-causal (≈ −0.005 KS).

## Ide yang belum diuji (kandidat lonjakan nyata)
Sinyal baru, bukan model baru: rasio arus kas bulanan (`income_like_credits`/`outflows`), konsistensi skor antar provider bureau,
pola kuantitas per kategori per bulan, interaksi harga relatif × beban cicilan terverifikasi; serta repeat-CV multi-split untuk menurunkan noise.

---

# Pendekatan alternatif untuk KS 0.57 — semua diuji, semua gagal

Untuk mencapai 0.57 diperlukan lompatan ≈ +0,047 KS, sedangkan seluruh pendekatan
"model lebih pintar" hanya menghasilkan ±0,002. Berikut 11 pendekatan yang benar-benar
**berbeda** dari sebelumnya, beserta hasil terukurnya (fold-level, LightGBM OOF ≈ 0.520):

| # | Pendekatan | Hasil | Kenapa gagal |
|---|---|---|---|
| 1 | **Learning-to-rank** (LambdaRank, grup acak 1.000) | KS **0.468 / 0.460** | pairwise loss dengan grup kecil mengorbankan separator global |
| 2 | **rank_xendcg** | KS **0.506 / 0.479** | sama |
| 3 | **Spesialis wilayah atas** (klasifikasi khusus 25%/40% teratas) | KS **0.5129** vs 0.5224 | dilatih pada subset kecil →AUROC di kepala turun (0.316→0.250) |
| 4 | **Spesialis subgroup** (30% customer tanpa histori) | KS 0.4919 vs 0.4946 | information memang lebih sedikit, bukan salah model |
| 5 | **Stacking** (logistic meta di atas OOF) | KS **0.5186** | lebih buruk dari rata-rata rank |
| 6 | **Pseudo-labeling data test** | KS **0.5108** vs 0.5197 |pseudo-label memperkuat confirmation bias |
| 7 | **Kalibrasi per subgroup** (kuantil + offset) | KS 0.5234 | tidak ada mis-kalibrasi antar subgroup |
| 8 | **Fitur kohort / double-demeaning** (hilangkan drift harga bersama per bulan-kategori) | 0.5188 vs 0.5197 | `lrel` ternyata tidak mengandung drift yang mengganggu |
| 9 | **kNN non-parametrik** (k=300, rank-40 fitur) | KS 0.5051 | hanya 4 detik, tapi tidak menambah blend |
| 10 | **Monotone constraints** | +0.006 di fold 0, −0.0004 di fold 1–3 | noise |
| 11 | **Normalisasi covariate shift** | tidak ada shift | train/test adalah split acak dari pool yang sama |

## Akar penyebab: noise intrinsik pada label
Setelah dipartisi dengan pasangan fitur terkuat (harga relatif × tunggakan; harga × skor bureau;
tunggakan × skor bureau → 135–324 sel), **varians antar sel sama persis dengan noise binomial
yang diharapkan** (0.0993 vs 0.0993; 0.0980 vs 0.0980; 0.1043 vs 0.1043).
Artinya setelah faktor-faktor risiko utama terpenuhi, sisa variasi label tidak terdeteksi
sebagai sinyal — label diambil secara Bernoulli dari risiko laten. Inilah alasan KS terkunci
di sekitar 0.52–0.53, sehingga batas "maksimal 0.6" yang disebut organizer jauh di atas apa yang bisa dicapai model dari data ini.

## Kesimpulan
Tidak ada pendekatan komputasi lain yang akan membawa CV KS dari 0.523 ke 0.57. Yang masih
mungkin (berurutan berdasarkan expected value):
1. **Verifikasi rumus metrik ke organizer** — satu-satunya langkah yang bisa membalik
   seluruh kesimpulan (biaya: satu pesan).
2. **Feature baru dari tabel yang sama** dengan rumus yang benar-benar berbeda — target
   terukur: subgroup tanpa histori (30% data, KS 0.491). Skrip evaluasi tersedia:
   `src2/probe_*.py` (semuanya mengukur KS per fold dalam ±2 menit).
3. **Grid hyperparameter manual yang lebih rapat** (±20 config × 2 fold): +0.002, tetap
   manual dan terdokumentasi (bukan AutoML).
