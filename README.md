# Afet Yönetimi NLP Projesi

Bu proje, deprem dönemlerinde sosyal medyada paylaşılan yardım içerikli tweet'lerden:

- ihtiyaç türlerini (9 etiketli multilabel sınıflandırma),
- temel konum bilgisini (mahalle/ilçe/il),
- aciliyet ve doğrulukla ilişkili yardımcı sinyalleri

çıkarabilen uçtan uca bir veri işleme ve modelleme hattı içerir.

## Canonical sürüm

- **Sürüm tag**: `v2_final_leakfree_exp3`
- **Seçilen model**: `exp3_silver_then_gold_v3_exgold` (silver pretrain → gold fine-tune, leak-free v3 silver)
- **Canonical test** (194 satır, gold): f1_micro=0.8998, f1_macro=0.8753
- **Canonical tahmin**: `data/predictions/need_predictions_geolocated_v2_final.csv` (63,180 satır, 96,071 girdiden dedup sonrası)
- **Release paket**: `release/need_classifier_v2/`

Selection pointer: [`models/final/selection.json`](models/final/selection.json).
Tam rasyonel: [`docs/final_model_selection.md`](docs/final_model_selection.md).

## Klasör yapısı

```
afetYonetimi_colab/
├── data/
│   ├── processed/        # işlenmiş girdi (emergency_geolocated_96k.csv vb.)
│   ├── labeling/         # silver/pseudo etiketli veri (v3 = ex-gold leak-free)
│   ├── modeling/         # train/val/test split'leri (gold_combined canonical)
│   ├── predictions/      # model tahmin çıktıları (v2_final canonical)
│   └── analysis/         # karşılaştırma / hata / leak / overlap / QA raporları
├── scripts/              # preprocess, etiketleme, training, tuning, predict, QA
├── models/               # eğitilmiş modeller, eşik dosyaları, eval metrikleri
│   └── final/            # canonical selection pointer
├── release/
│   └── need_classifier_v2/   # release-ready paket (pointer-based)
├── docs/                 # rasyonel dokümanları, teknik özet
├── colab/                # Colab eğitim rehberi
└── requirements_modeling.txt
```

## Canonical uçtan-uca zincir

Aşağıdaki sıra canonical v2'nin nasıl üretildiğini gösterir. Her adım için script + artifact
verilmiştir; her artifact bir öncekinin çıktısı üstünde çalışır.

| # | Adım | Script | Canonical artifact |
|---|---|---|---|
| 1 | Ön işleme | `preprocess_emergency_data.py` | `data/processed/...` |
| 2 | Acil + konum çıkarım | `extract_emergency_location.py` | `data/processed/emergency_geolocated_96k.csv` |
| 3 | Gold etiketleme + IAA | `scripts/create_need_classification_sample.py`, `scripts/prepare_double_annotation.py`, `scripts/compute_iaa_and_adjudication.py`, `scripts/export_gold_from_adjudication.py` | `data/need_classification_gold.csv` |
| 4 | Gold extension (rare-label annotation pack) | `scripts/build_need_classification_rare_label_pack.py` → human annotation → export | `data/need_classification_gold_extension.csv` → `data/need_classification_gold_combined.csv` |
| 5 | Silver v3 leak-free | `scripts/build_need_classification_silver_from_geolocated.py --exclude-gold` | `data/labeling/need_classification_silver_v3_exgold_*.csv` |
| 6 | Split hazırlama (canonical gold_combined) | `scripts/prepare_model_splits.py` | `data/modeling/need_classification_gold_combined/{train,val,test}.csv` |
| 7 | Eğitim (silver pretrain → gold fine-tune) | `scripts/train_need_classifier.py` (exp3_silver_then_gold_v3_exgold) | `models/exp3_silver_then_gold_v3_exgold/final/` |
| 8 | Threshold tuning (CV OOF) | `scripts/tune_thresholds_cv.py` | `models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json` |
| 9 | Experiment comparison (leak-free) | `scripts/compare_experiments.py` | `data/analysis/experiment_comparison_v3_leakfree.{md,json,csv}` |
| 10 | Hata analizi | `scripts/error_analysis_v2_leakfree.py` | `data/analysis/error_analysis_v2_leakfree.{md,json}` + FP/FN/slices CSV |
| 11 | id-level leak audit | `scripts/leak_audit_v3.py` | `data/analysis/step7_leak_audit_v3.{md,json}` |
| 12 | Content-level overlap audit | `scripts/content_overlap_audit_v2_leakfree.py` | `data/analysis/content_overlap_audit_v2_leakfree.{md,json}` |
| 13 | Final selection | *(el-editli)* | `models/final/selection.json`, `docs/final_model_selection.md` |
| 14 | Toplu tahmin (96k havuz) | `scripts/predict_need_classifier.py` + `scripts/finalize_prediction_metadata.py` | `data/predictions/need_predictions_geolocated_v2_final.{csv,meta.json}` |
| 15 | Prediction QA | `scripts/prediction_qa_v2_final.py` | `data/analysis/prediction_qa_v2_final.{md,json}` |
| 16 | Release paketleme | *(el-editli)* | `release/need_classifier_v2/` |

## Hızlı canonical tekrar-üretim

```powershell
# 0) Bağımlılık
pip install -r requirements_modeling.txt

# 1) Eğitim (silver-then-gold, leak-free silver v3)
python scripts/train_need_classifier.py `
  --config configs/exp3_silver_then_gold_v3_exgold.yaml

# 2) CV-OOF threshold tuning
python scripts/tune_thresholds_cv.py `
  --model-dir models/exp3_silver_then_gold_v3_exgold/final `
  --splits-dir data/modeling/need_classification_gold_combined `
  --out models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json

# 3) Canonical toplu tahmin
python scripts/predict_need_classifier.py `
  --model-dir models/exp3_silver_then_gold_v3_exgold/final `
  --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json `
  --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json `
  --input data/processed/emergency_geolocated_96k.csv `
  --output data/predictions/need_predictions_geolocated_v2_final.csv `
  --dedup-by-id --batch-size 128

# 4) Metadata finalize + QA
python scripts/finalize_prediction_metadata.py
python scripts/prediction_qa_v2_final.py
```

Aynı komutların bash versiyonu: `release/need_classifier_v2/examples/predict_command.sh`.

## Canonical çıktılar

- Tahmin CSV: `data/predictions/need_predictions_geolocated_v2_final.csv` (63,180 satır)
- Tahmin meta: `data/predictions/need_predictions_geolocated_v2_final.meta.json`
- QA raporu: `data/analysis/prediction_qa_v2_final.md`
- Release paket: `release/need_classifier_v2/`

## Historical / non-canonical artifact notu

Eski paralel model dizinleri, contaminated/v1 karşılaştırma raporları ve
superseded tahmin CSV'leri yerel klasörden kaldırıldı. Güncel tüketim ve
dashboard için yalnızca canonical v2 final çıktıları kullanılmalıdır:

- `models/final/selection.json`
- `models/exp3_silver_then_gold_v3_exgold/final/`
- `data/predictions/need_predictions_geolocated_v2_final.csv`
- `data/predictions/need_predictions_geolocated_v2_final.meta.json`

## Önemli notlar ve bilinen sınırlamalar

- **Silver rule-based.** Silver etiketleri `scripts/ai_prefill_annotations.py` kuralları ile
  üretildi; gold insan etiketi değil. Silver baseline'ın yüksek metrikleri gold doğruluğunu
  doğrudan garanti etmez.
- **id-level leak kapalı, content-level overlap açık.** `exp3_v3_exgold` silver pool'unda
  hiçbir gold id bulunmuyor (step 7), ama normalize edilmiş metin overlap'i ~%18 exact /
  ~%30 near-dup düzeyinde. Rapor edilen test F1'inin ~0.01–0.02 üstten tamponlu olabileceği
  makul varsayılmalı; bu residual risk `docs/final_model_selection.md` ve
  `release/need_classifier_v2/docs/known_limitations.md` içinde belgelendi.
- **Zayıf etiketler**: `guvenlik` (F1=0.571), `bilgi_paylasimi` (F1=0.681). CV eşikleri
  production'da recall feda ediyor; ihtiyaç halinde ayrı bir `threshold_production.json`
  tasarlanabilir.
- **Rare-label F1=1.0 değerleri** (`altyapi`, `saglik`, `psikolojik`) küçük test destekli
  (1–7 pozitif) ve kalibre başarı değildir — tek-tahmin saturasyonudur.
- **Havuz prior'ı**: `emergency_geolocated_96k.csv` "acil yardım + konum" filtresinden geçmiş
  tweet'ler içerir; bu yüzden tahmin çıktısında `arama_kurtarma` oranı %62.83 — over-fire değil,
  prior match.
- **Dashboard/API kodu bu modelleme reposunda tutulmaz.** Aktif Streamlit dashboard
  ayrı `afetYonetimi-dashboard` klasöründedir; model tüketicileri için minimal adapter:
  `release/need_classifier_v2/examples/adapter_snippet.py`.

## Dokümantasyon

- Teknik özet (1 sayfa): [`docs/final_technical_summary_v2.md`](docs/final_technical_summary_v2.md)
- Seçim rasyonali: [`docs/final_model_selection.md`](docs/final_model_selection.md)
- Çalışma prensibi (model): [`model_calisma_prensibi.md`](model_calisma_prensibi.md)
- Colab akışı: [`colab/README.md`](colab/README.md)
- Release paket: [`release/need_classifier_v2/README.md`](release/need_classifier_v2/README.md)
