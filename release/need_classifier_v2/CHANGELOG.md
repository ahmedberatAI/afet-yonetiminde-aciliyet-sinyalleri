# CHANGELOG

## v2_final_leakfree_exp3 — 2026-04-21

- **selected_experiment_key**: `exp3_silver_then_gold_v3_exgold`
- **threshold_source**: `cv` (multilabel-stratified 5-fold, smoothed F1, OOF global)
- **leak status**: id-level leak closed (0/1934). Content-level overlap residual risk documented.
- Canonical test (194 rows): f1_micro=0.8998, f1_macro=0.8753.
- Canonical prediction: 96,071 → 63,180 unique rows on `emergency_geolocated_96k.csv`.
- Packaging: pointer-based release; weights not copied.

### Supersedes

- `data/predictions/need_predictions_geolocated_63k.csv` (v0)
- `data/predictions/need_predictions_geolocated_gold_combined_weighted.csv` (v1a)
- `data/predictions/need_predictions_geolocated_v1_gold_combined_weighted.csv` (v1)
- `models/final/selection_historical_v1_gold_combined_weighted.json` (preview v1 selection)

### Known gaps vs. a true production release

- Content-level dedup not applied to silver (step 10+ item).
- `guvenlik` / `bilgi_paylasimi` precision-oriented CV thresholds; recall-oriented production
  thresholds not yet tuned.
- Small canonical test set (194 rows) — rare-label F1 variance is high.
- No web app / API adapter implemented in this repo (step 13 = not applicable).
