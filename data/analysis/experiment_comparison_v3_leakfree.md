# Step 7 — Need Classification Experiment Comparison (v3, LEAK-FREE)

## Header

- **Status**: CANONICAL comparison (supersedes `experiment_comparison_v2.*`)
- **Why v3?** v2 compared exp3 unfairly: the silver pretraining pool contained every gold_combined id (train=1547, val=193, test=194). v3 rebuilds the silver pool with all 1934 gold ids excluded.
- **Test CSV (shared across all 3 experiments)**: `data/modeling/need_classification_gold_combined/test.csv` (194 rows)
- **Val CSV**: `data/modeling/need_classification_gold_combined/val.csv` (193 rows)
- **GPU**: NVIDIA GeForce RTX 5090 Laptop (CUDA 12.8, torch 2.10.0+cu128) — required, no CPU fallback
- **Leak audit**: old silver had 1934/1934 gold-id overlap; new silver_exgold has 0/1934.

## Shared experimental setup

- Backbone: `dbmdz/bert-base-turkish-cased` (BERTurk, cased, 110M params)
- Task: multi-label classification, 9 need categories, sigmoid + per-label threshold
- Loss: BCEWithLogitsLoss; exp2 and exp3 additionally weight positives by `pos_weight = neg / pos` clipped to [1, 50]
- Per-label thresholds: multilabel-stratified 5-fold CV on `train+val` (k=5, seed=42, grid [0.05, 0.95] step 0.01, smoothed F1 α=1.0, strategy `oof_global`)

## Experiments

| Key | Base | Pos-weight | LR | Silver leak-free |
|---|---|---|---|---|
| `exp1_gold_v2_bce` | `dbmdz/bert-base-turkish-cased` | False | 2e-5 | True |
| `exp2_gold_v2_posw` | `dbmdz/bert-base-turkish-cased` | True | 2e-5 | True |
| `exp3_silver_then_gold_v3_exgold` | `models/need_classification_silver_63k_exgold/final` | True | 1e-5 | True |

## Summary metrics on `test.csv` (194 rows, CV-tuned thresholds)

| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |
|---|---|---|---|---|
| `exp1_gold_v2_bce` | 0.7325 | 0.4137 | 0.6734 | 0.8029 |
| `exp2_gold_v2_posw` | 0.7298 | 0.5560 | 0.7022 | 0.7596 |
| `exp3_silver_then_gold_v3_exgold` | 0.8998 | 0.8753 | 0.9154 | 0.8846 |

### Same metrics on `val.csv` (193 rows)

| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |
|---|---|---|---|---|
| `exp1_gold_v2_bce` | 0.7758 | 0.4308 | 0.7061 | 0.8607 |
| `exp2_gold_v2_posw` | 0.7470 | 0.5083 | 0.7243 | 0.7711 |
| `exp3_silver_then_gold_v3_exgold` | 0.9040 | 0.8313 | 0.9179 | 0.8905 |

## Per-label F1 on test

| label | `exp1_gold_v2_bce` | `exp2_gold_v2_posw` | `exp3_silver_then_gold_v3_exgold` | best |
|---|---|---|---|---|
| arama_kurtarma | 0.9279 | 0.7792 | 0.9686 | `exp3_silver_then_gold_v3_exgold` |
| saglik | 0.0000 | 0.7273 | 1.0000 | `exp3_silver_then_gold_v3_exgold` |
| barinma | 0.7692 | 0.7097 | 0.9565 | `exp3_silver_then_gold_v3_exgold` |
| gida_su | 0.5385 | 0.7143 | 0.8889 | `exp3_silver_then_gold_v3_exgold` |
| altyapi | 0.0000 | 0.0000 | 1.0000 | `exp3_silver_then_gold_v3_exgold` |
| guvenlik | 0.0690 | 0.6667 | 0.5714 | `exp2_gold_v2_posw` |
| lojistik | 0.6585 | 0.6471 | 0.8116 | `exp3_silver_then_gold_v3_exgold` |
| psikolojik | 0.0000 | 0.0000 | 1.0000 | `exp3_silver_then_gold_v3_exgold` |
| bilgi_paylasimi | 0.7600 | 0.7600 | 0.6809 | `exp1_gold_v2_bce` |

## Per-label thresholds used at test time

| label | `exp1_gold_v2_bce` | `exp2_gold_v2_posw` | `exp3_silver_then_gold_v3_exgold` |
|---|---|---|---|
| arama_kurtarma | 0.6000 | 0.5600 | 0.6000 |
| saglik | 0.1200 | 0.8400 | 0.9000 |
| barinma | 0.4400 | 0.7100 | 0.9300 |
| gida_su | 0.2400 | 0.7600 | 0.1400 |
| altyapi | 0.0800 | 0.5800 | 0.7300 |
| guvenlik | 0.0600 | 0.5800 | 0.7400 |
| lojistik | 0.2200 | 0.5800 | 0.9100 |
| psikolojik | 0.1800 | 0.5500 | 0.8600 |
| bilgi_paylasimi | 0.3600 | 0.8000 | 0.8700 |

## Threshold provenance

- **exp1_gold_v2_bce**
  - thresholds: `models/exp1_gold_v2_bce/thresholds_cv.json`
  - meta: `models/exp1_gold_v2_bce/threshold_tuning_cv_meta.json`
  - strategy=`oof_global`, k=5, smoothing α=1.0
  - type: `cv`
- **exp2_gold_v2_posw**
  - thresholds: `models/exp2_gold_v2_posw/thresholds_cv.json`
  - meta: `models/exp2_gold_v2_posw/threshold_tuning_cv_meta.json`
  - strategy=`oof_global`, k=5, smoothing α=1.0
  - type: `cv`
- **exp3_silver_then_gold_v3_exgold**
  - thresholds: `models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json`
  - meta: `models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_meta.json`
  - strategy=`oof_global`, k=5, smoothing α=1.0
  - type: `cv`

## Rare-label honesty block

Three labels have very thin positive support in the gold pool (`altyapi`=20, `guvenlik`=35, `psikolojik`=10 positives across 1740 rows). On the 194-row test the positive counts are even smaller (`altyapi`=3, `guvenlik`=4, `psikolojik`=1). A single correct prediction can saturate F1; read rare-label numbers as **qualitative** signal only.

| label | pool positives | test positives | `exp1_gold_v2_bce` | `exp2_gold_v2_posw` | `exp3_silver_then_gold_v3_exgold` |
|---|---|---|---|---|---|
| altyapi | 20 | 3 | 0.0000 | 0.0000 | 1.0000 |
| guvenlik | 35 | 4 | 0.0690 | 0.6667 | 0.5714 |
| psikolojik | 10 | 1 | 0.0000 | 0.0000 | 1.0000 |

## Winners

- **Best test f1_micro**: `exp3_silver_then_gold_v3_exgold`
- **Best test f1_macro**: `exp3_silver_then_gold_v3_exgold`
- **Best test precision_micro**: `exp3_silver_then_gold_v3_exgold`
- **Best test recall_micro**: `exp3_silver_then_gold_v3_exgold`

Per-label winners on test:

- `arama_kurtarma` → `exp3_silver_then_gold_v3_exgold`
- `saglik` → `exp3_silver_then_gold_v3_exgold`
- `barinma` → `exp3_silver_then_gold_v3_exgold`
- `gida_su` → `exp3_silver_then_gold_v3_exgold`
- `altyapi` → `exp3_silver_then_gold_v3_exgold`
- `guvenlik` → `exp2_gold_v2_posw`
- `lojistik` → `exp3_silver_then_gold_v3_exgold`
- `psikolojik` → `exp3_silver_then_gold_v3_exgold`
- `bilgi_paylasimi` → `exp1_gold_v2_bce`

## Takeaway

- `exp3_silver_then_gold_v3_exgold` **still wins** after the leakage fix. This is now a fair comparison: the silver pretrain corpus no longer contains any gold_combined id, so the advantage reflects genuine transfer from distant-supervision priors rather than contamination.
- exp1 vs exp2: pos_weight roughly trades micro-precision for rare-label recall — macro-F1 rises ~0.41 → ~0.56 while micro-F1 stays ~0.73.
- Silver prior effect, fairly measured: the gap between exp3_v3_exgold and exp2 on rare labels is the clean signal for "does distant supervision help rare classes?" Answer: yes — altyapi and psikolojik go from F1=0 to saturated, and without the leak this gain is now a legitimate modeling result (small-support caveat still applies).

## Critical risks & caveats

- **Test set is small** (194 rows). Absolute metrics have wide CIs; rare labels with 1–4 test positives can flip between F1=0 and F1=1 on a single prediction.
- **Silver signal is rule-based**, not human-labeled. Silver metrics (~0.99 F1) measure agreement with `scripts/ai_prefill_annotations.py` patterns, not real ground truth — this is a distant-supervision pretraining signal only.
- **Silver_exgold was freshly trained in step 7** (49k train rows). The previous `models/need_classification_silver_63k/final` remains on disk for reproducing the historical (contaminated) v2 comparison — it is NOT canonical anymore.
- **exp3 v3 CV thresholds** were produced by re-running 5-fold multilabel-stratified CV with silver_exgold→gold fine-tune per fold. Scratch dir: `models/_cv_tuning_scratch_exp3_v3_exgold`.
- **`models/final/selection.json` has NOT been touched.** Step 9 artifacts are left untouched by this step — the v3 winner is documented but not promoted.
- **exp1 / exp2 reuse existing artifacts.** Their models never used silver weights, so they were not affected by the leakage and did not need retraining.

## Per-experiment notes

### `exp1_gold_v2_bce`

- **model_dir**: `models/exp1_gold_v2_bce/final`
- **config**: `data/modeling/experiments/exp1_gold_v2_bce.yaml`
- **base model**: `dbmdz/bert-base-turkish-cased`
- **silver provenance**: none (trained directly from BERTurk base, no silver pretraining)
- **leak_free**: True
- **training**: Gold-only BCE (no pos_weight), LR=2e-5, 3 epochs, BS=16
- **threshold source**: `models/exp1_gold_v2_bce/thresholds_cv.json` (type: `cv`)
- **test metrics**: f1_micro=0.7325, f1_macro=0.4137, P=0.6734, R=0.8029
- **notes**: Gold-only baseline with vanilla BCE. No silver pretraining, so the silver-leakage issue does not apply. Rare labels (saglik, altyapi, psikolojik) collapse to F1=0 on test — expected failure mode without pos_weight.

### `exp2_gold_v2_posw`

- **model_dir**: `models/exp2_gold_v2_posw/final`
- **config**: `data/modeling/experiments/exp2_gold_v2_posw.yaml`
- **base model**: `dbmdz/bert-base-turkish-cased`
- **silver provenance**: none (trained directly from BERTurk base, no silver pretraining)
- **leak_free**: True
- **training**: Gold-only + pos_weight=neg/pos (clipped [1,50]), LR=2e-5, 3 epochs, BS=16
- **threshold source**: `models/exp2_gold_v2_posw/thresholds_cv.json` (type: `cv`)
- **test metrics**: f1_micro=0.7298, f1_macro=0.5560, P=0.7022, R=0.7596
- **notes**: Pos-weight recovers saglik, guvenlik, gida_su vs exp1. No silver pretraining, so the silver-leakage issue does not apply. altyapi and psikolojik still F1=0 — positive support in pool (20 / 10 rows) is too thin for pure-gold to recover.

### `exp3_silver_then_gold_v3_exgold`

- **model_dir**: `models/exp3_silver_then_gold_v3_exgold/final`
- **config**: `data/modeling/experiments/exp3_silver_then_gold_v3_exgold.yaml`
- **base model**: `models/need_classification_silver_63k_exgold/final`
- **silver provenance**: data/modeling/need_classification_silver_63k_exgold/*.csv — built from emergency_geolocated_96k with gold id exclusion (0 overlap with any gold_combined split)
- **leak_free**: True
- **training**: LEAK-FREE silver-pretrain (61,246 distant-label rows, gold ids EXCLUDED, BERTurk + pos_weight, LR=2e-5, 3 epochs) then gold fine-tune with pos_weight, LR=1e-5, 3 epochs, BS=16
- **threshold source**: `models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json` (type: `cv`)
- **test metrics**: f1_micro=0.8998, f1_macro=0.8753, P=0.9154, R=0.8846
- **notes**: Replaces the contaminated exp3_silver_then_gold_v2 (whose silver pool contained ALL 1934 gold_combined ids including the 194 test ids). Here the silver pool has been re-built from the geolocated 96k dataset with every gold id excluded; silver→gold transfer is therefore fair. The silver prior still carries rare-label semantics; altyapi and psikolojik saturate at F1=1.0 on test but the positive support is tiny (3 and 1 rows), so read as qualitative signal.

