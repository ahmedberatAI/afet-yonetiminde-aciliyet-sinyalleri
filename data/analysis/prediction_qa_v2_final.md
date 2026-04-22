# Prediction QA — v2 final (leak-free)

Step 10 çıktısı. Canonical tahmin dosyası üstünde bütünlük, şema netliği, etiket prevalansı, dilimleme ve anomali taraması.

## Kaynaklar

- CSV: `data/predictions/need_predictions_geolocated_v2_final.csv` (63,180 satır)
- Meta: `data/predictions/need_predictions_geolocated_v2_final.meta.json`
- Model: `models/exp3_silver_then_gold_v3_exgold/final`
- Threshold source: `cv` / `oof_global`

## 1) Bütünlük

- CSV satır sayısı: **63,180**, meta.rows_after: **63,180**, meta.rows_before: **96,071**
- Tekrarlanan id: **0** (beklenen: 0)
- Eksik sütun: **0** (beklenen: 0)
- NaN olasılık hücresi: **0** (beklenen: 0)
- [0,1] dışı olasılık: **0** (beklenen: 0)
- `pred_positives` meta ile eşleşme: **EVET**
- Eşik/pred tutarsızlığı: **0** (hepsi tutarlı)

## 2) Şema netliği (meta'ya eklenen alanlar)

Aşağıdaki alanlar `need_predictions_geolocated_v2_final.meta.json` içine eklendi:

- `prediction_columns`, `probability_columns`
- `label_to_pred_column`, `label_to_prob_column`
- `auxiliary_columns` (`pred_label_count`, `pred_any_need`)
- `identifier_columns`, `metadata_columns`, `text_columns`, `location_columns`, `urgency_columns`
- `row_count`, `schema_note`, `schema_finalized_{by,at}`

Amaç: tüketiciler (dashboard, API adapter, notebook) `prob_*` ve `pred_*` eşleşmelerini ve `threshold_global_fallback=0.5`'in uygulanmadığını tek bakışta görebilsin.

## 3) Etiket prevalansı

| label | threshold | pozitif | oran | prob_mean | prob_p95 |
|---|---|---|---|---|---|
| arama_kurtarma | 0.60 | 39,694 | 62.83% | 0.643 | 0.999 |
| saglik | 0.90 | 2,030 | 3.21% | 0.037 | 0.027 |
| barinma | 0.93 | 3,869 | 6.12% | 0.076 | 0.999 |
| gida_su | 0.14 | 3,469 | 5.49% | 0.055 | 0.923 |
| altyapi | 0.73 | 850 | 1.35% | 0.018 | 0.013 |
| guvenlik | 0.74 | 214 | 0.34% | 0.029 | 0.079 |
| lojistik | 0.91 | 9,246 | 14.63% | 0.162 | 0.999 |
| psikolojik | 0.86 | 268 | 0.42% | 0.017 | 0.043 |
| bilgi_paylasimi | 0.87 | 3,374 | 5.34% | 0.146 | 0.882 |

Not: `arama_kurtarma` oranı havuzun %62.83'ü — bu beklenen, çünkü girdi kümesi zaten 'acil yardım + konum' filtresinden geçmiş tweet'ler. Yine de tüketiciler için bu *prior* net belirtilmeli (known_limitation).

## 4) pred_label_count / pred_any_need dağılımı

| label_count | satır |
|---|---|
| 0 | 14,207 |
| 1 | 35,998 |
| 2 | 12,036 |
| 3 | 819 |
| 4 | 114 |
| 5 | 5 |
| 6 | 1 |

- `pred_any_need=1`: **48,973** (77.51%)
- `pred_any_need=0`: **14,207** (22.49%)

## 5) Dilimleme

### 5.1 Metin uzunluğu (tweet_clean karakter sayısı)

| bucket | satır | arama_kurtarma | lojistik | barinma | gida_su | bilgi_paylasimi | any_need |
|---|---|---|---|---|---|---|---|
| 221+ | 35,481 | 67.0% | 20.3% | 7.2% | 6.9% | 5.8% | 83.5% |
| 141-220 | 22,312 | 59.6% | 8.3% | 5.0% | 3.9% | 5.4% | 72.8% |
| 81-140 | 5,130 | 49.6% | 3.6% | 3.7% | 2.5% | 2.3% | 58.9% |
| 41-80 | 256 | 26.6% | 2.0% | 3.9% | 2.3% | 0.4% | 35.9% |
| 00-40 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### 5.2 Urgency bucket

| bucket | satır | arama_kurtarma | saglik | lojistik | any_need |
|---|---|---|---|---|---|
| 0.8-1.0 | 56,168 | 61.9% | 3.5% | 14.8% | 77.3% |
| 0.0-0.2 | 7,012 | 70.4% | 1.1% | 13.3% | 79.6% |

### 5.3 Province (top 15 by satır)

Tam tablo: `prediction_qa_v2_final.slices_province.csv`. Özet:

| province | satır | arama_kurtarma | lojistik | bilgi_paylasimi | any_need |
|---|---|---|---|---|---|
| Hatay | 26,124 | 66.1% | 15.9% | 5.8% | 78.0% |
| Adana | 11,222 | 63.7% | 13.2% | 5.2% | 75.9% |
| nan | 8,107 | 49.2% | 12.5% | 4.5% | 76.4% |
| Unknown | 5,051 | 66.3% | 16.4% | 5.3% | 78.5% |
| Adıyaman | 4,978 | 66.7% | 15.5% | 4.9% | 79.1% |
| Kahramanmaraş | 3,034 | 59.9% | 14.9% | 6.5% | 74.4% |
| Gaziantep | 2,234 | 68.3% | 10.3% | 3.7% | 82.9% |
| Diyarbakır | 1,074 | 57.9% | 16.5% | 5.8% | 75.6% |
| Malatya | 866 | 47.2% | 10.9% | 3.7% | 80.9% |
| Osmaniye | 251 | 55.4% | 10.4% | 9.6% | 68.9% |
| Şanlıurfa | 202 | 51.5% | 6.9% | 4.0% | 74.8% |
| Kilis | 37 | 16.2% | 16.2% | 5.4% | 78.4% |

## 6) Etiket-özel gözlemler

- **arama_kurtarma** (62.83%, 39,694 satır). Havuz zaten 'acil yardım' filtresinden geçtiği için bu oran beklenen üst sınırda. Gold test F1=0.969; production'da over-fire değil, *prior match*.

- **guvenlik** (0.34%, 214 satır). Step 8'de belgelenen zayıflık burada da görünüyor: 63k havuzdaki gerçek yağmacılık/asayiş sinyallerinin büyük kısmı muhtemelen `altyapi` veya `arama_kurtarma` olarak etiketleniyor. Tüketiciler `guvenlik` için recall-öncelikli ayrı eşik kullanabilir (`threshold_production.json` önerisi, step 10 kapsamı dışında).

- **psikolojik** (0.42%, 268 satır). Gold test F1=1.0 ama test pozitifi=1; havuzda bu oran kalibre bir başarı değil, rare-label saturasyonu. Düşük recall bekleyin.

- **bilgi_paylasimi** (5.34%, 3,374 satır). CV eşiği 0.87 tutucu; step 8'de gösterildi ki 'haber alamıyoruz' tarzı ifadeler sistematik olarak `arama_kurtarma`'ya kayıyor. Gerçek prevalans burada görünenin muhtemelen üstünde.

## 7) Bulguların sınıflandırması

### release_blocker (0)

_Yok._

### warning (0)

_Yok._

### known_limitation (1)

- **over_firing_labels** — Labels with positive rate > 0.30 on the 63k pool.
  - detay: `[{'label': 'arama_kurtarma', 'rate': 0.6282684393795505}]`

## 8) Sonuç

**Release blocker yok.** Canonical tahmin çıktısı (`v2_final`) release paketlenmeye uygun. Mevcut uyarılar ve known-limitation'lar selection rationale ile tutarlı (content-overlap residual risk, guvenlik/bilgi_paylasimi zayıflıkları, rare-label saturasyonu, havuzun arama_kurtarma-ağırlıklı prior'ı). Step 11 (release packaging) ve step 14 (teknik özet) bu uyarılara atıf verecek.
