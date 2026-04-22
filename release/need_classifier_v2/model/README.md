# model/ — pointer

Bu dizin intentionally boştur — eğitilmiş ağırlıklar **repo içinde** ayrıca tutulur ve release klasörüne kopyalanmaz (safetensors ≈ 423 MB).

## Canonical model konumu

[../../../models/exp3_silver_then_gold_v3_exgold/final](../../../models/exp3_silver_then_gold_v3_exgold/final)

İçerik:

| dosya | boyut | ne |
|---|---|---|
| `config.json` | ~1 KB | HF model config (9 label head) |
| `model.safetensors` | ~423 MB | eğitilmiş ağırlıklar |
| `tokenizer.json` | ~0.7 MB | BERTurk tokenizer |
| `tokenizer_config.json` | ~0.5 KB | tokenizer config |
| `training_args.bin` | ~5 KB | HF Trainer args (referans; yükleme için gerekmez) |

## Yükleme

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

MODEL_DIR = Path("models/exp3_silver_then_gold_v3_exgold/final")  # repo-relative
LABELS_JSON = Path("models/exp3_silver_then_gold_v3_exgold/label_columns.json")
THRESHOLDS_JSON = Path("release/need_classifier_v2/thresholds/thresholds_cv.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
thresholds = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
```

## Neden kopyalanmadı?

- Ağırlık dosyası tekrar üretilebilir (train script'i deterministik, seed=42).
- Release klasörü repo içinde kalıyor; duplicate copy artifact şişkinliği ve versiyon drift riski yaratır.
- İlerideki bir "external release" turunda (örn. Hugging Face Hub push) bu dosya tek bir noktadan yayınlanır.
