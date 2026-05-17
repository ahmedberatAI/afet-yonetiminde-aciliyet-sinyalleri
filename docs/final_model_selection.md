# Final Model Selection (v2, leak-free)

Bu doküman step 9'un canonical çıktısıdır. Önceki v1 (F1-macro=0.309) seçim tarihsel bir öngösterim idi; bu v2, leak-free step 7 karşılaştırmasının ve step 8 hata analizinin üstüne inşa edilmiştir.

## Karar

- **Seçilen experiment key**: `exp3_silver_then_gold_v3_exgold`
- **Model dizini**: [models/exp3_silver_then_gold_v3_exgold/final](../models/exp3_silver_then_gold_v3_exgold/final)
- **Threshold kaynağı**: [models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json](../models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json)
- **Threshold türü**: `cv` / `oof_global` (multilabel-stratified 5-fold, smoothed F1 α=1.0, grid [0.05, 0.95] adım 0.01)
- **Canonical test**: [data/modeling/need_classification_gold_combined/test.csv](../data/modeling/need_classification_gold_combined/test.csv) — 194 satır
- **Seçim pointer**: [models/final/selection.json](../models/final/selection.json) (v2, bu dokümanla senkron)

## Kanıt kaynakları

| Kanıt | Dosya |
|---|---|
| Canonical karşılaştırma | [experiment_comparison_v3_leakfree.md](../data/analysis/experiment_comparison_v3_leakfree.md) |
| Canonical karşılaştırma (JSON) | [experiment_comparison_v3_leakfree.json](../data/analysis/experiment_comparison_v3_leakfree.json) |
| Hata analizi (step 8) | [error_analysis_v2_leakfree.md](../data/analysis/error_analysis_v2_leakfree.md) |
| Leak audit (step 7) | [step7_leak_audit_v3.md](../data/analysis/step7_leak_audit_v3.md) |
| Content-overlap audit (step 9) | [content_overlap_audit_v2_leakfree.md](../data/analysis/content_overlap_audit_v2_leakfree.md) |

## Test metrikleri (CV-tuned thresholds)

| metrik | değer |
|---|---|
| f1_micro | 0.8998 |
| f1_macro | 0.8753 |
| precision_micro | 0.9154 |
| recall_micro | 0.8846 |

### Per-label F1 (CV thresholds)

| label | test pozitif | F1 | thr |
|---|---|---|---|
| arama_kurtarma | 111 | 0.969 | 0.60 |
| saglik | 7 | 1.000 | 0.90 |
| barinma | 12 | 0.957 | 0.93 |
| gida_su | 10 | 0.889 | 0.14 |
| altyapi | 3 | 1.000 | 0.73 |
| guvenlik | 4 | **0.571** | 0.74 |
| lojistik | 35 | 0.812 | 0.91 |
| psikolojik | 1 | 1.000 | 0.86 |
| bilgi_paylasimi | 25 | **0.681** | 0.87 |

## Seçim rasyoneli (dürüst)

1. **Aggregate metrikler tutarlı biçimde öndedir.** Canonical v3 karşılaştırmasında `exp3_silver_then_gold_v3_exgold` hem f1_micro (0.8998 vs exp1=0.7325, exp2=0.7298) hem f1_macro (0.8753 vs exp1=0.4137, exp2=0.5560) sıralamasında birinci. Aynı zamanda precision_micro (0.9154) ve recall_micro (0.8846) başlıklarında da birinci — kazanç tek boyutta değil.

2. **Silver pretrain'in rare-label faydası gerçek ama kalibre değil.** Silver prior olmadan `altyapi`, `saglik`, `psikolojik` test setinde F1=0 ile çöküyordu (exp1 ve çoğu için exp2). Silver ile bu etiketler tahmin edilebilir hâle geldi, ama test pozitif sayıları çok küçük (altyapi=3, saglik=7, psikolojik=1) — F1=1.0 skorları **tek tahmin saturasyonu** seviyesinde, kalibre başarı olarak okunamaz. Rare-label kazanımının sinyal olduğuna inanıyoruz, metriğin kendisine değil.

3. **`guvenlik` gerçek bir zayıflık.** Test pozitifi=4, F1=0.571 (TP=2, FP=1, FN=2). Yağmacılık / asayiş çağrıları için model sinyali zayıf; step 10'da veya yeni bir etiketleme turunda ele alınmalı. Winner olması `guvenlik`'teki zayıflığı yumuşatmaz.

4. **`bilgi_paylasimi` canonical winner'ın açık zayıf noktasıdır.** F1=0.681, bu etikette **exp1'in (F1=0.76) altında**. CV eşiği 0.87 çok tutucu; 9 FN'in 4'ü model "arama_kurtarma" dediği için oluşmuş. "Haber alamıyoruz, gören var mı" gibi bilgi çağrıları sistematik olarak `arama_kurtarma` olarak gidiyor. Bu bir etiket-tanımı / threshold meselesi; production için ayrı bir recall-öncelikli eşik değerlendirilebilir (step 9 kapsamı dışında).

5. **Content-overlap kalıntı riski.** Step 7 v3 id-level leak'i kapattı (1934/1934 → 0/1934 id overlap). Ancak bu step 9'da koşulan içerik-seviyesi audit gösterdi ki:
   - Test setinin **~%18'inde** (35/194) silver havuzunda normalize-edilmiş **aynı metin** başka bir id ile mevcut.
   - **~%30'unda** (59/194) 5-char shingle Jaccard ≥ 0.85 seviyesinde near-dup.
   - Aynı oranlar train (~%21 exact, ~%32 near-dup) ve val (~%22, ~%33) için de geçerli.

   Bu bir residual risktir: silver, aynı olayla ilgili retweet/alıntı metinleri içeriyor. Selection'ı durdurmuyor çünkü:
   - v2 (100% id-overlap) → v3 (0% id-overlap) geçişinde f1_micro sadece +0.0120 oynadı; content-level overlap'in dramatik metrik şişirmesi için güçlü kanıt yok.
   - exp1/exp2 silver kullanmadığı için sıralama *olgusal olarak* etkilenmiyor — hangi çözüm en iyisi sorusu cevabı değişmiyor.
   - Silver etiketleri rule-based, gold etiketleri değil — gördüğü metin ≠ gördüğü etiket.

   Yine de bu overlap'in metriklerin üstünde bir ~+0.01–0.02 tampon üretmiş olabileceğini varsaymak makul; gerçek genelleme metriği muhtemelen şu rapordaki rakamların birazcık altındadır. Step 10 veya sonraki bir silver yeniden-kurma turunda content-level dedup uygulanmalı.

## Bilinen sınırlamalar

- **Küçük test seti (194 satır)**. Aggregate metrikler geniş CI'lere sahiptir; rare etiketler 1–4 pozitif ile tek tahmin flip'i yapabilir.
- **Silver sinyal rule-based**. Silver baseline'ın val/test f1_micro ≈ 0.99 değerleri gold yerine `scripts/ai_prefill_annotations.py` kurallarıyla uyumu ölçer.
- **Content-level overlap**. (bkz. selection rationale madde 5.)
- **`guvenlik` / `bilgi_paylasimi` zayıflıkları**. (bkz. madde 3–4.)

## Tarihsel / non-canonical artifact notu

Eski paralel model dizinleri, contaminated/v1 karşılaştırma raporları ve
superseded prediction pas'ları yerel klasörden kaldırıldı. Bu seçim notu için
canonical olan tek aktif prediction çıktısı:

Canonical prediction çıktısı **sadece**:

- `data/predictions/need_predictions_geolocated_v2_final.csv`
- `data/predictions/need_predictions_geolocated_v2_final.meta.json`

## Tekrar üretim

```bash
python scripts/predict_need_classifier.py \
  --model-dir models/exp3_silver_then_gold_v3_exgold/final \
  --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
  --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
  --input data/processed/emergency_geolocated_96k.csv \
  --output data/predictions/need_predictions_geolocated_v2_final.csv \
  --dedup-by-id \
  --batch-size 128
```

Metadata zenginleştirme (threshold_source, artifact link'leri, timestamp) `scripts/finalize_prediction_metadata.py` tarafından `selection.json` okunup eklenir.
