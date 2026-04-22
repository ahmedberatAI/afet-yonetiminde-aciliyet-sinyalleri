# need_classifier_v2 — release

**Status:** provisional canonical release
**Version tag:** `v2_final_leakfree_exp3`
**Generated:** 2026-04-21 (repo-local; not a public artifact)

Bu dizin, Turkish disaster-tweet multilabel need sınıflayıcısının (`exp3_silver_then_gold_v3_exgold`) release-ready paketidir. Ağır model dosyaları KOPYALANMAMIŞTIR — yalnızca pointer, config, threshold ve küçük metadata dosyaları bulunur. Eğitilmiş ağırlıklara `model/README.md` üzerinden ulaşın.

## İçerik

```
release/need_classifier_v2/
├── README.md                         — bu dosya
├── VERSION                           — sürüm tag
├── CHANGELOG.md                      — sürüm notları
├── label_columns.json                — 9 need etiketi (sıralı)
├── model/
│   └── README.md                     — ağırlıklara pointer (safetensors yolunu gösterir)
├── thresholds/
│   ├── thresholds_cv.json            — CV-tuned per-label eşikler (uygulanan)
│   └── threshold_tuning_cv_meta.json — tuning provenance
├── docs/
│   ├── metrics_summary.md            — canonical test metrikleri
│   ├── selection_summary.md          — seçim gerekçesi (kısa)
│   ├── usage.md                      — predict nasıl çalıştırılır
│   └── known_limitations.md          — zayıflıklar ve residual risk
├── artifacts/
│   ├── eval_test_tuned.json          — gold test metrikleri (canonical)
│   ├── eval_val_tuned.json           — gold val metrikleri
│   └── POINTERS.md                   — analiz/kaynak artifact pointer'ları
└── examples/
    ├── predict_command.sh            — canonical predict komutu
    └── adapter_snippet.py            — küçük Python adapter örneği
```

## Canonical model

- **Model dizini**: [../../models/exp3_silver_then_gold_v3_exgold/final](../../models/exp3_silver_then_gold_v3_exgold/final) (HuggingFace SequenceClassification)
- **Base**: `dbmdz/bert-base-turkish-cased`
- **Head**: 9-label multilabel (sigmoid)
- **Thresholds**: `thresholds/thresholds_cv.json` (per-label, CV-tuned, OOF global)
- **Selection pointer**: [../../models/final/selection.json](../../models/final/selection.json)

## Canonical test metrikleri

| metrik | değer |
|---|---|
| f1_micro | 0.8998 |
| f1_macro | 0.8753 |
| precision_micro | 0.9154 |
| recall_micro | 0.8846 |

Per-label detaylar ve zayıf noktalar için `docs/metrics_summary.md`.

## Canonical tahmin çıktısı

- CSV: [../../data/predictions/need_predictions_geolocated_v2_final.csv](../../data/predictions/need_predictions_geolocated_v2_final.csv)
- Meta: [../../data/predictions/need_predictions_geolocated_v2_final.meta.json](../../data/predictions/need_predictions_geolocated_v2_final.meta.json)
- QA: [../../data/analysis/prediction_qa_v2_final.md](../../data/analysis/prediction_qa_v2_final.md)

## Status — neden "provisional canonical"

Bu, step 9'un canonical seçimi ve step 10'un QA'sından sonra üretildi. Release blocker yok; yine de iki açık residual risk var (seçim rasyonalinden aynen devralındı):

1. **Content-level overlap** — id-level leak kapalı (0/1934), ama silver havuzunda test satırlarının ~%18'i exact, ~%30'u near-dup olarak tekrar ediyor. Metriğin muhtemel ~+0.01–0.02 üstten şişmesi beklenir.
2. **`guvenlik` (F1=0.57) ve `bilgi_paylasimi` (F1=0.68) zayıflıkları** — production tüketiciler için recall-öncelikli ayrı bir eşik tasarımı gerekebilir.

Bu risklerin büyüklüğü dashboard/API tüketiminin kararlarını etkileyebilir; bu yüzden yayın "provisional" olarak işaretlendi. Gerçek bir kullanıcı anlaşmasına gitmeden önce step 10+ bir silver yeniden-kurma turu ile content-level dedup uygulanmalı.

## Tekrar üretim

`examples/predict_command.sh` çalıştırılabilir komutu içerir. Özet:

```bash
python scripts/predict_need_classifier.py \
  --model-dir models/exp3_silver_then_gold_v3_exgold/final \
  --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
  --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
  --input data/processed/emergency_geolocated_96k.csv \
  --output data/predictions/need_predictions_geolocated_v2_final.csv \
  --dedup-by-id --batch-size 128
python scripts/finalize_prediction_metadata.py
python scripts/prediction_qa_v2_final.py
```
