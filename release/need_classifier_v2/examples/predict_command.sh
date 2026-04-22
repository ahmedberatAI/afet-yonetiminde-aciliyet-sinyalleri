#!/usr/bin/env bash
# Canonical batch prediction command for need_classifier_v2.
# Tekrar üretim için repo root'undan çalıştırın.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

python scripts/predict_need_classifier.py \
  --model-dir models/exp3_silver_then_gold_v3_exgold/final \
  --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
  --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
  --input data/processed/emergency_geolocated_96k.csv \
  --output data/predictions/need_predictions_geolocated_v2_final.csv \
  --dedup-by-id \
  --batch-size 128

python scripts/finalize_prediction_metadata.py
python scripts/prediction_qa_v2_final.py

echo "OK - canonical prediction regenerated:"
echo "  data/predictions/need_predictions_geolocated_v2_final.csv"
echo "  data/predictions/need_predictions_geolocated_v2_final.meta.json"
echo "  data/analysis/prediction_qa_v2_final.{md,json}"
