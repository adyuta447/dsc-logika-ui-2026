"""Generate `agas/v70/notebook/notebook_v70.ipynb`.

The notebook is a reproducible report, not a second training run: every number is
read from the artifacts in `agas/v70/output`, and the training section only
prints the command that produced `preds_v70.npz`. Set RUN_TRAINING=True to
actually re-run it.
"""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "output"
NB = HERE.parent / "notebook" / "notebook_v70.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


cells = []

cells.append(md(r"""
# DSC0152 — notebook v70

## Kenapa v70 ada

Public terbaik tetap **0.56424** dari `v66_04` (`ext0.5 | tm30 | lgb_lean`). `v69_10`
menguji arah yang sama tetapi mengganti pivot `lgb_lean` dengan `extra`, dan
hasilnya turun ke **0.55989**. Pelajarannya: pivot dan resep ekstrapolasi v66
dipertahankan; yang boleh berubah adalah **kualitas core** yang diekstrapolasi.

v70 karena itu hanya mengubah dua hal:

1. **Fitur lintas sumber baru** — panel bulanan repayment × cash flow × e-commerce,
   status pinjaman, dan kedalaman e-commerce. Semua dihitung pada `decision_datetime`
   masing-masing aplikasi, jadi tidak ada leakage.
2. **Training lebih banyak** — 15 fold, 10 fold seed, 2 varian LightGBM. Leg v66
   tidak dilatih ulang; prediksi OOF/test-nya dipakai lagi apa adanya, sehingga
   arah public yang sudah terbukti tidak ikut berubah.

Hasil utama: leg `v70_extra` bagged **AUC 0.848369 / KS 0.528503**, di atas
`v69_extra` (0.848132 / 0.527736). Kandidat ekstrapolasi terbaik
`plateau|tm30w7|v70_extra|lgb_lean` mencapai **KS OOF 0.530232** dengan
Spearman 0.9984 terhadap `v66_04`.
"""))

cells.append(code(r"""
import json, os, sys, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score

RUN_TRAINING = False
FOLDS = 15
FOLD_SEEDS = '42,7,13,21,5,101,3,17,11,23'
LGB_SEEDS = '0'
VARIANTS = 'extra,stiff7'
DROP_PREFIXES = 'cf_,pv_'

HERE = Path.cwd().resolve()
ROOT = HERE
for cand in [HERE, *HERE.parents]:
    if (cand / 'data').is_dir() and (cand / 'agas').is_dir():
        ROOT = cand
        break
AGAS = ROOT / 'agas'
VER = AGAS / 'v70'
OUT = VER / 'output'
TRAINING = VER / 'training'
sys.path.insert(0, str(AGAS / 'shared'))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

print('repo  :', ROOT)
print('version:', VER)
print('output :', OUT)
print('train  :', TRAINING)
assert (ROOT / 'data/train_applications.csv').exists()
"""))

cells.append(md(r"""
## 1. Probe sumber sinyal

Sebelum menambah fitur, tiap sumber diuji sendirian dengan 5 fold dan seed 42.
Tujuannya hanya memastikan blok baru punya sinyal mentah, bukan untuk memilih
model final.
"""))

cells.append(code(r"""
z = np.load(OUT / 'source_auc_v70.npz', allow_pickle=True)
y = z['y']
rows = []
for k in z.files:
    if not k.startswith('oof_'):
        continue
    p = z[k]
    rows.append({'source': k[4:], 'auc': roc_auc_score(y, p),
                 'ks': ks_2samp(p[y == 1], p[y == 0]).statistic})
src = pd.DataFrame(rows).sort_values('auc', ascending=False)
display(src.style.format({'auc': '{:.4f}', 'ks': '{:.4f}'}))

# per-source rank blend: does simple averaging recover the all-feature model?
names = [k for k in z.files if k.startswith('oof_') and k != 'oof_all']
blend = np.mean([rankdata(z[k]) / len(y) for k in names], axis=0)
print(f"rank blend of {len(names)} sources: AUC {roc_auc_score(y, blend):.4f} "
      f"(all-feature model {roc_auc_score(y, z['oof_all']):.4f})")
"""))

cells.append(md(r"""
## 2. Blok fitur baru dan screening

Empat blok dibangun di `agas/v70/training/feat_v70.py`:

| prefix | isi | jumlah |
|---|---|---:|
| `ms_` | panel bulanan repayment (due/late/unpaid) × cash flow (net, saldo, low-balance days) | 67 |
| `ls_` | status pinjaman berjalan: late/dpd per pinjaman, aktif vs lunas, rasio bayar | 40 |
| `ed_` | kedalaman e-commerce: keranjang, kategori, tren harga, konsentrasi | 51 |
| `ss_` | panel 3 arah: late × net cash flow × e-commerce spend | 42 |

Screening memakai resep yang sama untuk semua baris: 5 fold, fold seed
42/7/13, satu LGB seed, config `extra`, factor features aktif. Yang berubah
hanya blok yang dimasukkan.
"""))

cells.append(code(r"""
rows = []
labels = {
    's0_base': 'baseline (v69 lean + factor)',
    's1_ms': '+ monthstress',
    's2_ls': '+ loanstatus',
    's3_ed': '+ ecomdepth',
    's4_all': '+ ms + ls + ed',
    's5_ss': '+ spendstress',
    's6_allss': '+ all + spendstress',
}
for key, label in labels.items():
    d = json.loads((OUT / f'summary_{key}.json').read_text())
    rows.append({'run': key, 'blok': label, 'n_lean': d['n_lean'],
                 'auc': d['mean_auc'], 'ks': d['mean_ks'],
                 'auc_sd': d['sd_auc'], 'menit': d['minutes']})
scr = pd.DataFrame(rows)
display(scr.style.format({'auc': '{:.6f}', 'ks': '{:.6f}',
                          'auc_sd': '{:.6f}', 'menit': '{:.1f}'}))

base = scr.loc[scr.run == 's0_base'].iloc[0]
scr['d_auc'] = scr.auc - base.auc
scr['d_ks'] = scr.ks - base.ks
print(scr[['run', 'd_auc', 'd_ks']].to_string(index=False))
print('\nKesimpulan: ms_ konsisten positif; ls_ netral; ed_ negatif sendirian '
      'tetapi +all sedikit di atas +ms; ss_ negatif dan dibuang.')
"""))

cells.append(md(r"""
## 3. Training final

Leg v66 (`lgb_full`, `lgb_lean`, `lgbx_full`, `cat_full`, `xgb_full`, `linear`)
dipakai ulang dari `agas/v66/output/preds_v66.npz`. Yang baru hanya dua varian
`lgb_lean` v70 di atas feature set `+all` (684 kolom lean, 690 dengan factor):

- **extra** — `num_leaves=15`, `feature_fraction=0.45`, `extra_trees=True`
- **stiff7** — `num_leaves=7`, `feature_fraction=0.20`, `lambda_l2=40`

Konfigurasi: 15 fold, fold seed `42,7,13,21,5,101,3,17,11,23`, LGB seed 0.
"""))

cells.append(code(r"""
cmd = (f"python3 {TRAINING}/train_lean_v70.py "
       f"--features {OUT}/features_v70.pkl --variants {VARIANTS} "
       f"--folds {FOLDS} --fold-seeds {FOLD_SEEDS} --lgb-seeds {LGB_SEEDS} "
       f"--drop-prefixes {DROP_PREFIXES} --out-prefix v70")
print(cmd)
if RUN_TRAINING:
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT)
else:
    print('\nRUN_TRAINING=False: memakai preds_v70.npz yang sudah ada.')
assert (OUT / 'preds_v70.npz').exists(), 'preds_v70.npz belum ada; jalankan training dulu'
"""))

cells.append(code(r"""
z70 = np.load(OUT / 'preds_v70.npz', allow_pickle=True)
y70 = z70['y'].astype(int)
per_seed = json.loads((OUT / 'summary_v70.json').read_text())['per_seed']

rows = []
for seed, d in per_seed.items():
    for v, m in d.items():
        rows.append({'fold_seed': int(seed), 'variant': v,
                     'auc': m['auc'], 'ks': m['ks']})
seeds = pd.DataFrame(rows)
display(seeds.pivot(index='fold_seed', columns='variant', values='auc')
        .style.format('{:.6f}'))

summary = json.loads((OUT / 'summary_v70.json').read_text())
bag = pd.DataFrame([
    {'variant': v, 'auc': summary[v]['auc'], 'ks': summary[v]['ks'],
     'n_seeds': len(summary['fold_seeds']), 'folds': summary['folds']}
    for v in summary['variants']
])
display(bag.style.format({'auc': '{:.6f}', 'ks': '{:.6f}'}))
print('v69_extra: AUC 0.848132 / KS 0.527736')
print(f"v70_extra: AUC {summary['extra']['auc']:.6f} / KS {summary['extra']['ks']:.6f}")
print(f"delta     : AUC {summary['extra']['auc'] - 0.848132:+.6f} / "
      f"KS {summary['extra']['ks'] - 0.527736:+.6f}")
"""))

cells.append(md(r"""
## 4. Rantai blend dan kandidat

Rantai v66 tidak diubah:

```
trees4 = mean(rank(lgb_full, lgb_lean, cat_full, xgb_full))
core   = (2 * trees4 + rank(lgbx_full)) / 3
tm30   = 0.7 * core + 0.3 * rank(linear)
ext_r  = rank01(tm30 + r * (tm30 - rank(lgb_lean)))
```

v70 menambahkan leg baru ke `trees`, dan menguji bobot `w` yang lebih besar
untuk leg baru (`tm30w3/5/7`). Ekstrapolasi tetap terhadap `lgb_lean` v66.
Pemilihan kandidat memakai AUC (bukan KS mentah), lantai AUC 0.0025 di bawah
kandidat terbaik, dan deduplikasi Spearman 0.9995.
"""))

cells.append(code(r"""
cand = json.loads((OUT / 'summary_v70_candidates.json').read_text())
top = pd.DataFrame(cand['candidates'])
cols = ['slot', 'name', 'auc', 'ks', 'ks_minus_sigma', 'spearman_v66_04', 'valid']
display(top[cols].head(12).style.format({
    'auc': '{:.6f}', 'ks': '{:.6f}', 'ks_minus_sigma': '{:.6f}',
    'spearman_v66_04': '{:.6f}'}))
print(f"pool={cand['pool_size']} eligible={cand['eligible']} kept={cand['kept']}")
print('best v70 leg:', cand['best_v70_leg'])
"""))

cells.append(md(r"""
## 5. Validasi CSV

Semua file harus lolos empat pemeriksaan: 30.000 baris, `application_id` unik
dan persis sama dengan `data/test_applications.csv`, probabilitas di [0, 1],
tidak ada NaN.
"""))

cells.append(code(r"""
test_ids = set(pd.read_csv(ROOT / 'data/test_applications.csv',
                           usecols=['application_id']).application_id)
rows = []
for p in sorted((OUT).glob('submission_v70_*.csv')):
    d = pd.read_csv(p)
    ok = (len(d) == 30000 and d.application_id.is_unique
          and set(d.application_id) == test_ids
          and d.default_probability.notna().all()
          and d.default_probability.between(0, 1).all())
    rows.append({'file': p.name, 'rows': len(d), 'valid': bool(ok),
                 'min': d.default_probability.min(),
                 'max': d.default_probability.max()})
val = pd.DataFrame(rows)
print(f'{len(val)} file, {val.valid.sum()} valid, {(~val.valid).sum()} invalid')
display(val.head(25).style.format({'min': '{:.4f}', 'max': '{:.4f}'}))
assert val.valid.all()
"""))

cells.append(md(r"""
## 6. Rekomendasi submit

Tiga kandidat yang paling masuk akal, berurutan:

1. **`submission_v70_06_plateau_tm30w7_v70_extra_lgb_lean.csv`** — pilihan utama.
   KS OOF tertinggi (0.530232), AUC 0.847623, masih keluarga ekstrapolasi v66
   (Spearman 0.9984), dan rasio 0.25/0.5/0.75 dirata-rata sehingga tidak
   bergantung pada satu rasio yang kebetulan bagus.
2. **`submission_v70_07_ext0.5_tm30v70_lgb_lean.csv`** — pilihan konservatif.
   Paling dekat dengan resep public 0.56424 (Spearman 0.99967), memakai kedua
   varian v70 dengan bobot sama.
3. **`submission_v70_01_leg_v70_extra.csv`** — kontrol AUC. Model mentah
   terkuat (AUC 0.848369) tanpa ekstrapolasi; arah yang belum pernah diuji di
   public.

Kalau hanya boleh satu: **slot 06**. Kalau boleh dua: **06 + 07**.
"""))

cells.append(code(r"""
best = {c['slot']: c for c in cand['candidates']}
for slot in (6, 7, 1):
    c = best[slot]
    print(f"slot {slot:02d}  {c['file']}")
    print(f"         AUC {c['auc']:.6f}  KS {c['ks']:.6f}  "
          f"Spearman vs v66_04 {c['spearman_v66_04']:.6f}")
"""))

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
NB.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB)
print('wrote', NB)
