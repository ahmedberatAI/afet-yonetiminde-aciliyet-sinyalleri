# Model Calisma Prensibi

Bu dokuman, projedeki nihai cok etiketli ihtiyac siniflandirma modelinin kod uzerinden nasil calistigini adim adim aciklar.

## 1. Modelin Amaci ve Tipi

Model, Turkce tweet metinlerinden birden fazla ihtiyac etiketini ayni anda tahmin eder (multi-label classification).

Etiketler:
- `arama_kurtarma`
- `saglik`
- `barinma`
- `gida_su`
- `altyapi`
- `guvenlik`
- `lojistik`
- `psikolojik`
- `bilgi_paylasimi`

Temel egitim dosyasi:
- `scripts/train_need_classifier.py`

Konfigurasyon:
- `data/modeling/need_classification_silver_63k/training_config.yaml`

Nihai model cikti dizini:
- `models/need_classification_silver_63k/final`

## 2. Uctan Uca Pipeline

1. Islenmis tweet verisi hazirlanir (`data/processed/emergency_geolocated_96k.csv`).
2. Rule-based pseudo-label ile silver veri uretilir (`scripts/build_need_classification_silver_from_geolocated.py`).
3. Veri train/val/test olarak bolunur (`scripts/prepare_model_splits.py`).
4. BERT tabanli multi-label model egitilir (`scripts/train_need_classifier.py`).
5. Label bazli threshold optimizasyonu yapilir (`scripts/tune_thresholds.py`).
6. Test seti tuned threshold ile degerlendirilir (`scripts/evaluate_need_classifier.py`).
7. Toplu tahmin dosyasi uretilir (`scripts/predict_need_classifier.py`).

## 3. Egitim Mantigi (`train_need_classifier.py`)

### 3.1 Veri Yukleme ve Dogrulama

- CSV dosyalari configten okunur.
- Label kolonlari zorunlu olarak `0/1` formatina cevrilir (`_coerce_binary_0_1`).
- Etiket listesi `label_columns.json` olarak output dizinine yazilir.

### 3.2 Model ve Tokenizer

- Base model: `dbmdz/bert-base-turkish-cased`
- `AutoTokenizer` ile metinler tokenize edilir.
- `AutoModelForSequenceClassification` su sekilde kullanilir:
  - `num_labels = 9`
  - `problem_type = "multi_label_classification"`

### 3.3 Kayip Fonksiyonu ve Dengesiz Veri

Modelin kayip fonksiyonu `BCEWithLogitsLoss`'tur.

Opsiyonel olarak class imbalance icin `pos_weight` kullanilir:
- Her etiket icin train setten `neg/pos` orani hesaplanir.
- Degerler `pos_weight_min` ve `pos_weight_max` araligina clip edilir.
- Sonuc `pos_weight.json` dosyasina yazilir.

### 3.4 Ozel Trainer

Script icinde `_WeightedTrainer` tanimlanmistir.
- `compute_loss` fonksiyonunda `BCEWithLogitsLoss(pos_weight=...)` veya standart `BCEWithLogitsLoss` kullanilir.

### 3.5 Egitim Sirasinda Metrikler

Logit ciktilari sigmoid ile olasiliga cevrilir, threshold ile 0/1 tahmine donusturulur.

Hesaplanan metrikler:
- `f1_micro`
- `f1_macro`
- `precision_micro`
- `recall_micro`
- Label bazli F1 (`f1_<label>`)

### 3.6 Egitim Sonu Uretilen Dosyalar

- `val_metrics.json`
- `test_metrics.json`
- `final/` altinda model ve tokenizer dosyalari

## 4. Threshold Tuning (`tune_thresholds.py`)

Bu adim her etiket icin en uygun threshold'u ayri ayri bulur.

- Grid: `0.01` ile `0.99` arasi (adim: `0.01`).
- Her label icin F1 maksimize eden threshold secilir.
- Sonuc:
  - `thresholds.json`
  - `threshold_tuning_report.txt`

Neden gerekli:
- Multi-label ve dengesiz veri senaryolarinda tek global threshold genelde yetersizdir.

## 5. Degerlendirme (`evaluate_need_classifier.py`)

- Model + tokenizer yuklenir.
- Test CSV uzerinde inference yapilir.
- `--thresholds-json` verilirse label bazli threshold kullanilir.
- Sonuc JSON'a yazilir (or. `eval_test_tuned.json`).

## 6. Tahmin Uretimi (`predict_need_classifier.py`)

Bu script modeli operasyonel ciktiya cevirir.

Girdi:
- Islenmis CSV (`tweet_clean`, yoksa `tweet` fallback)

Islem:
- Opsiyonel `id` bazli dedup
- Model inference
- Her etiket icin:
  - `prob_<label>`
  - `pred_<label>`
- Toplam etiket ve en az bir ihtiyac bilgisi:
  - `pred_label_count`
  - `pred_any_need`

Cikti:
- `data/predictions/need_predictions_geolocated_63k.csv`
- `data/predictions/need_predictions_geolocated_63k.meta.json`

## 7. Karar Mekanizmasi (Matematiksel Ozet)

Her etiket icin model bir logit uretir.

Olasilik:

`p_i = sigmoid(logit_i)`

Karar:

- Eger `p_i >= t_i` ise etiket `1`
- Eger `p_i < t_i` ise etiket `0`

Burada `t_i`, ilgili etiketin tuned threshold degeridir (`thresholds.json`).

## 8. Nihai Modeli Aciklarken Gosterilecek Minimum Dosyalar

1. `scripts/train_need_classifier.py`
2. `scripts/tune_thresholds.py`
3. `scripts/evaluate_need_classifier.py`
4. `scripts/predict_need_classifier.py`
5. `data/modeling/need_classification_silver_63k/training_config.yaml`
6. `models/need_classification_silver_63k/final/config.json`
7. `models/need_classification_silver_63k/thresholds.json`
8. `models/need_classification_silver_63k/eval_test_tuned.json`
9. `data/predictions/need_predictions_geolocated_63k.meta.json`
