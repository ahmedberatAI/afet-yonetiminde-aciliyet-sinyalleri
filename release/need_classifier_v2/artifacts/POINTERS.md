# Artifact pointers

Büyük artifact'lar release dizinine kopyalanmadı; aşağıdaki repo-relative yollardan ulaşın.

## Seçim ve karşılaştırma

| artifact | yol |
|---|---|
| Canonical selection pointer | [../../../models/final/selection.json](../../../models/final/selection.json) |
| Selection rasyonali (doc) | [../../../docs/final_model_selection.md](../../../docs/final_model_selection.md) |
| Canonical experiment comparison | [../../../data/analysis/experiment_comparison_v3_leakfree.md](../../../data/analysis/experiment_comparison_v3_leakfree.md) |
| Same (JSON) | [../../../data/analysis/experiment_comparison_v3_leakfree.json](../../../data/analysis/experiment_comparison_v3_leakfree.json) |
| Same (CSV) | [../../../data/analysis/experiment_comparison_v3_leakfree.csv](../../../data/analysis/experiment_comparison_v3_leakfree.csv) |

## Hata analizi (step 8)

| artifact | yol |
|---|---|
| Error analysis rapor | [../../../data/analysis/error_analysis_v2_leakfree.md](../../../data/analysis/error_analysis_v2_leakfree.md) |
| Same (JSON) | [../../../data/analysis/error_analysis_v2_leakfree.json](../../../data/analysis/error_analysis_v2_leakfree.json) |
| FP detayları | [../../../data/analysis/error_analysis_v2_leakfree.fp.csv](../../../data/analysis/error_analysis_v2_leakfree.fp.csv) |
| FN detayları | [../../../data/analysis/error_analysis_v2_leakfree.fn.csv](../../../data/analysis/error_analysis_v2_leakfree.fn.csv) |
| Slice metrikleri | [../../../data/analysis/error_analysis_v2_leakfree.slices.csv](../../../data/analysis/error_analysis_v2_leakfree.slices.csv) |

## Leak ve overlap auditleri

| artifact | yol |
|---|---|
| id-level leak audit (step 7) | [../../../data/analysis/step7_leak_audit_v3.md](../../../data/analysis/step7_leak_audit_v3.md) |
| Content-overlap audit (step 9) | [../../../data/analysis/content_overlap_audit_v2_leakfree.md](../../../data/analysis/content_overlap_audit_v2_leakfree.md) |
| Same (JSON) | [../../../data/analysis/content_overlap_audit_v2_leakfree.json](../../../data/analysis/content_overlap_audit_v2_leakfree.json) |

## Threshold tuning

| artifact | yol |
|---|---|
| CV-tuned thresholds | [../thresholds/thresholds_cv.json](../thresholds/thresholds_cv.json) |
| Tuning provenance meta | [../thresholds/threshold_tuning_cv_meta.json](../thresholds/threshold_tuning_cv_meta.json) |
| OOF predictions (tuning) | [../../../models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_oof.csv](../../../models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_oof.csv) |
| Human-readable report | [../../../models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_report.txt](../../../models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_report.txt) |

## Canonical tahmin çıktısı

| artifact | yol |
|---|---|
| Tahmin CSV | [../../../data/predictions/need_predictions_geolocated_v2_final.csv](../../../data/predictions/need_predictions_geolocated_v2_final.csv) |
| Tahmin meta | [../../../data/predictions/need_predictions_geolocated_v2_final.meta.json](../../../data/predictions/need_predictions_geolocated_v2_final.meta.json) |
| Tahmin QA (step 10) | [../../../data/analysis/prediction_qa_v2_final.md](../../../data/analysis/prediction_qa_v2_final.md) |
| Tahmin QA (JSON) | [../../../data/analysis/prediction_qa_v2_final.json](../../../data/analysis/prediction_qa_v2_final.json) |
| Label prevalence | [../../../data/analysis/prediction_qa_v2_final.label_prevalence.csv](../../../data/analysis/prediction_qa_v2_final.label_prevalence.csv) |

## Model dosyaları

Boyut nedeniyle release'e kopyalanmadı. Kanonik konum:
[../../../models/exp3_silver_then_gold_v3_exgold/final](../../../models/exp3_silver_then_gold_v3_exgold/final)
(yaklaşık 423 MB safetensors).
