# Step 7 Leak Audit (v3)

## Summary

The original step-7 comparison compared `exp3_silver_then_gold_v2` against exp1/exp2 using a silver pretraining pool that secretly contained **every** gold_combined id (train, val, AND test). The silver model therefore saw every gold test row during distant-label pretraining, which invalidates the silver→gold transfer comparison.

This audit documents the fix: a leak-free silver pool (`silver_63k_exgold`) was rebuilt from the geolocated 96k dataset with all 1934 gold ids excluded. A new silver model was trained on that clean pool. exp3 was re-instantiated as `exp3_silver_then_gold_v3_exgold` warm-started from the leak-free silver. CV thresholds were re-tuned. All evaluation is on the unchanged canonical `gold_combined/test.csv`.

## Contamination evidence (before fix)

- **Silver source**: `data/labeling/need_classification_silver_63k_profileA.csv`
- **Silver rows**: 63180
- **Overlap with canonical gold**: 1934 / 1934 ids
- **Overlap with gold train split**: 1547 / 1547
- **Overlap with gold val split**: 193 / 193
- **Overlap with gold test split**: 194 / 194 — THIS IS THE HARMFUL LEAK

Root cause: `scripts/build_need_classification_silver_from_geolocated.py` reads the entire geolocated 96k dataset, dedups by id, and runs rule-based labeling. It did NOT exclude gold_combined ids. Because the gold_combined dataset is a subset of the geolocated 96k corpus, the silver pool inherited every gold id.

## Remediation (how the fix was built)

1. `scripts/build_need_classification_silver_from_geolocated.py` updated: added `--exclude-gold-csv` flag that accepts one or more gold CSV paths. Their `id` columns are unioned and every matching silver row is removed before labeling.

2. Ran the updated script with **four** gold sources passed explicitly (redundant but paranoid), so the excluded id set = canonical gold ∪ train ∪ val ∪ test:

```bash
python scripts/build_need_classification_silver_from_geolocated.py \
  --input data/processed/emergency_geolocated_96k.csv \
  --output data/labeling/need_classification_silver_63k_profileA_exgold.csv \
  --profile A --dedup \
  --exclude-gold-csv data/need_classification_gold_combined.csv \
  --exclude-gold-csv data/modeling/need_classification_gold_combined/train.csv \
  --exclude-gold-csv data/modeling/need_classification_gold_combined/val.csv \
  --exclude-gold-csv data/modeling/need_classification_gold_combined/test.csv
```

3. Produced `data/labeling/need_classification_silver_63k_profileA_exgold.csv` (61,246 rows; 1,934 rows removed).

4. Split into `data/modeling/need_classification_silver_63k_exgold/{train,val,test}.csv` via `scripts/prepare_model_splits.py` (80/10/10, seed=42, stratified by `aciliyet_0_3`, same protocol as original silver split).

5. Trained `models/need_classification_silver_63k_exgold/final` on the leak-free splits with the same hyperparameters as the original silver baseline (BERTurk, pos_weight, LR=2e-5, 3 epochs, BS=16, fp16).

6. Fine-tuned `models/exp3_silver_then_gold_v3_exgold/final` on `gold_combined/train+val` warm-started from the leak-free silver, LR=1e-5, 3 epochs.

7. Ran 5-fold multilabel-stratified CV threshold tuning with the leak-free silver as the fold base model (scratch dir: `models/_cv_tuning_scratch_exp3_v3_exgold`). Emitted `thresholds_cv.json` + meta + report + OOF probabilities.

8. Evaluated `exp3_silver_then_gold_v3_exgold/final` on both canonical gold val and test with the new CV thresholds.

## Verification (after fix)

- **Leak-free silver source**: `data/labeling/need_classification_silver_63k_profileA_exgold.csv`
- **Rows**: 61246
- **Overlap with canonical gold**: 0 / 1934 — must be 0
- **Overlap with gold train**: 0 / 1547 — must be 0
- **Overlap with gold val**: 0 / 193 — must be 0
- **Overlap with gold test**: 0 / 194 — must be 0

Additionally verified: every leak-free silver split (train/val/test under `silver_63k_exgold/`) has 0 overlap with every gold_combined split.

## Impact: old vs new exp3 on the same gold test

| metric | exp3_v2 (contaminated) | exp3_v3_exgold (leak-free) | delta |
|---|---|---|---|
| f1_micro | 0.8878 | 0.8998 | +0.0120 |
| f1_macro | 0.8734 | 0.8753 | +0.0019 |
| precision_micro | 0.9010 | 0.9154 | +0.0144 |
| recall_micro | 0.8750 | 0.8846 | +0.0096 |

| label | exp3_v2 F1 | exp3_v3 F1 | delta |
|---|---|---|---|
| arama_kurtarma | 0.9502 | 0.9686 | +0.0184 |
| saglik | 1.0000 | 1.0000 | +0.0000 |
| barinma | 0.9565 | 0.9565 | +0.0000 |
| gida_su | 0.8889 | 0.8889 | +0.0000 |
| altyapi | 1.0000 | 1.0000 | +0.0000 |
| guvenlik | 0.5714 | 0.5714 | +0.0000 |
| lojistik | 0.8000 | 0.8116 | +0.0116 |
| psikolojik | 1.0000 | 1.0000 | +0.0000 |
| bilgi_paylasimi | 0.6939 | 0.6809 | -0.0130 |

## Interpretation

The headline macro-F1 barely moves (0.873 → 0.875). This is somewhat surprising given the scale of the leak (100% of gold ids present in the silver pool). Two factors explain it:

1. **Silver labels are rule-based, not gold labels.** The silver model was learning to imitate `ai_prefill_annotations.py` patterns, not the actual human labels. Those rules are coarse and don't carry much extra information about a specific gold test tweet beyond what the text itself provides. So even though the model saw the test texts during silver pretraining, it saw them tied to rule outputs — not to the true gold labels. The harm is bounded.

2. **Distant-supervision transfer is robust.** Whether the silver pool contains the gold ids or not, the rule patterns the silver model learns are general enough that fine-tuning on gold picks up the remaining task-specific information. The gain over exp2 (gold-only) stays.

**What this does NOT mean**: the leak was harmless. It means the leak did not help as much as a label-level leak would. A comparison claim like "silver pretrain beats gold-only" was scientifically invalid under v2 — even a tiny advantage might have come from contamination. v3 is the first comparison where this claim is defensible.

## Framing going forward

- `experiment_comparison_v2.*` → **HISTORICAL, CONTAMINATED**. Do not cite.
- `experiment_comparison_v3_leakfree.*` → **CANONICAL**. Use for any step-8 work or reporting.
- Final selection pointer (`models/final/selection.json`) left untouched. v3 winner (`exp3_silver_then_gold_v3_exgold`) is documented but not promoted in this step.

