# Usage — need_classifier_v2

## 1) Canonical toplu tahmin (repo içinden)

```bash
python scripts/predict_need_classifier.py \
  --model-dir models/exp3_silver_then_gold_v3_exgold/final \
  --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
  --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
  --input data/processed/emergency_geolocated_96k.csv \
  --output data/predictions/need_predictions_geolocated_v2_final.csv \
  --dedup-by-id \
  --postprocess-profile info_v1 \
  --batch-size 128

python scripts/finalize_prediction_metadata.py
python scripts/prediction_qa_v2_final.py
```

Çıktı: canonical `need_predictions_geolocated_v2_final.{csv,meta.json}` dosyaları.

## 2) Programmatic adapter (release dizini pivot)

`examples/adapter_snippet.py` dosyasına bakın. Kısaca:

```python
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).resolve().parents[1]  # release/need_classifier_v2/
REPO = BASE.parents[1]                      # repo root

MODEL_DIR = REPO / "models/exp3_silver_then_gold_v3_exgold/final"
labels = json.loads((BASE / "label_columns.json").read_text(encoding="utf-8"))
thresholds = json.loads((BASE / "thresholds/thresholds_cv.json").read_text(encoding="utf-8"))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

def predict(texts, max_length=192):
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(mdl(**enc).logits).cpu().numpy()
    preds = {lab: [int(probs[i, j] >= thresholds[lab]) for i in range(len(texts))]
             for j, lab in enumerate(labels)}
    return probs, preds
```

## 3) Schema (çıktı CSV)

| kolon grubu | örnek | açıklama |
|---|---|---|
| id | `id` | tweet id |
| metadata | `created_at`, `date`, `time` | orijinal zaman bilgisi |
| text | `tweet`, `tweet_clean` | ham + temizlenmiş metin |
| location | `neighborhood`, `district`, `province` | eşleştirilmiş konum |
| urgency | `urgency_score` | pre-computed acilik sinyali [0,1] |
| probability | `prob_<label>` (×9) | sigmoid probability [0,1] |
| prediction | `pred_<label>` (×9) | 1 if prob_<label> >= thresholds[<label>], else 0 |
| derived | `pred_label_count`, `pred_any_need` | toplam + any flag |

Tam schema ve label↔column mapping: `need_predictions_geolocated_v2_final.meta.json`
(`prediction_columns`, `label_to_pred_column`, `label_to_prob_column` alanları).

## 4) Eşik kullanımı (KRİTİK)

- **`thresholds_cv.json` per-label değerlerini kullanın.**
- `meta.threshold_global_fallback=0.5` değeri **uygulanmadı**; sadece bir fallback parametre adıdır.
- Varsayılan inference katmanı artık `--postprocess-profile info_v1` uygular. Bu profil
  canonical eşikleri değiştirmez; yalnızca güçlü bilgi-paylaşımı dili varsa ve
  `prob_bilgi_paylasimi >= 0.20` ise `pred_bilgi_paylasimi=1` yapar.
- Exact raw-threshold reproduksiyon için `--postprocess-profile none` kullanın.
- Production için daha recall-öncelikli bir ayar isterseniz (özellikle `guvenlik`, `bilgi_paylasimi`),
  `thresholds/threshold_tuning_cv_meta.json` içinde OOF F1/precision/recall eğrisine bakın ve
  ayrı bir `threshold_production.json` türetin — bu release bu varyantı içermez.

## 4.1) Postprocess `info_v1`

`info_v1` bir model ağırlığı değişikliği değildir; inference sonunda uygulanan küçük,
geri alınabilir bir kuraldır. OOF + validation üzerinde birlikte iyileştiği için
release adapter ve predict scriptinde varsayılan açıktır.

Kanıt:
- [postprocess_info_v1_validation_2026_05_17.md](../../data/analysis/postprocess_info_v1_validation_2026_05_17.md)
- [postprocess_info_v1.json](../postprocess_info_v1.json)

## 5) Üretim girdi gereksinimleri

Canonical predict scripti şu kolonları bekler:
- `id` (string; dedup için zorunludur)
- `tweet_clean` (string; model girişi)

Ek metadata kolonları (created_at, date, time, neighborhood, district, province, urgency_score)
tahmine etki etmez — predict script'i sadece `tweet_clean` üstünden çalışır; diğerleri
çıktıya *passthrough* ile kopyalanır.
