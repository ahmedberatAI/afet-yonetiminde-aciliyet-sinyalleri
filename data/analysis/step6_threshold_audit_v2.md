# Step 6 Threshold Audit v2

## Canonical strategy
- 5-fold iterative multilabel CV on the current `gold_combined` train+val pool only.
- Thresholds selected on concatenated OOF probabilities with smoothed F1 (alpha=1.0).
- Test set was not used during tuning.

## Stale artifact mismatch
- Old report: `models/need_classification_gold_combined_weighted/threshold_tuning_report.txt`
- Current val split: `data/modeling/need_classification_gold_combined/val.csv`
- Labels with positive-count mismatch:
  - arama_kurtarma: old=98 current=111
  - saglik: old=6 current=7
  - barinma: old=15 current=13
  - gida_su: old=11 current=3
  - altyapi: old=0 current=1
  - guvenlik: old=0 current=7
  - lojistik: old=29 current=23
  - psikolojik: old=0 current=1
  - bilgi_paylasimi: old=1 current=35

## Step 6 outputs
- `models/exp1_gold_v2_bce/thresholds_cv.json`
- `models/exp2_gold_v2_posw/thresholds_cv.json`
- `models/exp3_silver_then_gold_v2/threshold_tuning_cv_blocked.txt`

## exp3 blocker
- `models/need_classification_silver_63k/final` currently has no exported weight file (`model.safetensors` / `pytorch_model.bin`).
- CV tuning for the silver warm-start recipe is intentionally deferred until step 7 repairs or re-exports that model.

## Canonical note
- For step 6 onward, use the new `thresholds_cv.json` artifacts under `models/exp*_v2/` as canonical threshold sources.
