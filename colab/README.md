# Colab Workflow (Need Classification, Multi-Label)

This folder is a copy-paste friendly guide for running training/evaluation in Google Colab.

## 1) Put The Repo In Drive (Recommended Layout)

Create a Drive folder (example): `MyDrive/afetYonetimi_colab/`

Inside it, keep the same structure:

- `requirements_modeling.txt`
- `scripts/`
- `data/modeling/need_classification/` (train/val/test + `training_config.yaml`)

## 2) Colab Cells (Minimal)

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Go to project root:

```bash
%cd /content/drive/MyDrive/afetYonetimi_colab
!ls
!nvidia-smi
```

Install dependencies:

```bash
!pip install -r requirements_modeling.txt
```

Sanity check (no training):

```bash
!python scripts/train_need_classifier.py --prep-only
```

Train (weighted BCE loss via `pos_weight` from config):

```bash
!python scripts/train_need_classifier.py --output-dir models/need_classification
```

Tune per-label thresholds on validation:

```bash
!python scripts/tune_thresholds.py \
  --model-dir models/need_classification/final \
  --csv data/modeling/need_classification/val.csv \
  --out models/need_classification/thresholds.json \
  --report models/need_classification/threshold_tuning_report.txt
```

Evaluate on test with tuned thresholds:

```bash
!python scripts/evaluate_need_classifier.py \
  --model-dir models/need_classification/final \
  --csv data/modeling/need_classification/test.csv \
  --thresholds-json models/need_classification/thresholds.json \
  --out models/need_classification/eval_test_tuned.json
```

## Notes

- Labels with **0 positives** in train (e.g., `psikolojik`) cannot be learned; treat them as data-collection tasks.
- Very rare labels (<10 positives) will be unstable; consider adding targeted samples later.

