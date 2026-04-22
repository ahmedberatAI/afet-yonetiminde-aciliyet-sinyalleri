# Teknik Özet — need_classifier_v2 (1 sayfa)

**Sürüm**: `v2_final_leakfree_exp3` · **Tarih**: 2026-04-21 · **Durum**: provisional canonical

## 1. Projenin amacı

Deprem dönemlerinde Türkçe sosyal medya tweet'lerinden **9 ihtiyaç etiketini** (arama_kurtarma,
saglik, barinma, gida_su, altyapi, guvenlik, lojistik, psikolojik, bilgi_paylasimi) çok-etiketli
sınıflandırma + basit konum çıkarımı + aciliyet skoru üreten uçtan uca bir pipeline.

## 2. Veri akışı

`raw tweets` → `preprocess_emergency_data.py` (temizleme, urgency_score) →
`extract_emergency_location.py` (acil + konum filtresi) → `emergency_geolocated_96k.csv`
(model girdisi). Gold etiketleme IAA + adjudication ile 1,934 satır gold üretir
(`need_classification_gold_combined`, 194 satır canonical test). Silver etiket havuzu
gold id'lerini hariç tutarak (ex-gold) kural-tabanlı üretildi (`silver_v3_exgold`).

## 3. Modelleme

Base: `dbmdz/bert-base-turkish-cased`, 9-label multilabel (sigmoid head).
Kayıp: `BCEWithLogitsLoss` + `pos_weight = clip(neg/pos, 1, 50)`. Max length 192,
batch 128, fp16, RTX 5090 (CUDA 12.8, torch 2.10.0+cu128), deterministik
(seed=42, `torch.use_deterministic_algorithms`).

## 4. Deneyler (leak-free v3 karşılaştırması)

| rank | experiment_key | strateji | f1_micro | f1_macro |
|---|---|---|---|---|
| **1** | `exp3_silver_then_gold_v3_exgold` | silver pretrain → gold fine-tune | **0.8998** | **0.8753** |
| 2 | `exp2_gold_v2_posw` | gold + pos_weight | 0.7298 | 0.5560 |
| 3 | `exp1_gold_v2_bce` | gold + vanilla BCE | 0.7325 | 0.4137 |

(Kaynak: `data/analysis/experiment_comparison_v3_leakfree.*`. v1/v2 historical karşılaştırmalar
contaminated; canonical **değil** — README'de açıkça işaretlendi.)

## 5. Seçilen winner

`exp3_silver_then_gold_v3_exgold`, CV-OOF tuned thresholds (smoothed F1 α=1.0, 5-fold
multilabel-stratified). Per-label F1: arama_kurtarma 0.969 · saglik 1.000 · barinma 0.957 ·
gida_su 0.889 · altyapi 1.000 · **guvenlik 0.571** · lojistik 0.812 · psikolojik 1.000 ·
**bilgi_paylasimi 0.681**. Seçim pointer: `models/final/selection.json`.
Tam rasyonel: `docs/final_model_selection.md`.

## 6. Canonical tahmin çıktısı

- Girdi: `data/processed/emergency_geolocated_96k.csv` (96,071 satır)
- Dedup-by-id sonrası: **63,180 satır**, 32,891 duplicate atıldı
- Çıktı: `data/predictions/need_predictions_geolocated_v2_final.{csv,meta.json}`
- `pred_any_need=1`: 48,973 satır (%77.51)
- Per-label pozitifler: arama_kurtarma 39,694 · lojistik 9,246 · barinma 3,869 ·
  bilgi_paylasimi 3,374 · gida_su 3,469 · saglik 2,030 · altyapi 850 · psikolojik 268 ·
  guvenlik 214

## 7. Kalite (step 10 QA)

- **Release blocker: 0.** Row count, id uniqueness, NaN prob, OOR prob, threshold-pred
  tutarlılık — hepsi temiz.
- **Warning: 0.**
- **Known limitation: 1** — arama_kurtarma pozitif oranı %62.83, ama havuz zaten
  "acil yardım + konum" filtresinden geçtiği için bu **prior match**, over-fire değil.

## 8. Açık riskler (dürüst)

1. **Content-level overlap residual.** id-level leak kapalı (0/1934) ama test'in ~%18'i
   silver havuzunda exact, ~%30'u near-dup (J>=0.85). Gerçek genelleme F1'i
   muhtemelen rapor edilen rakamların **~0.01–0.02 altında**. v2 (id-contaminated) → v3
   (id-clean) geçişinde f1_micro yalnızca +0.0120 oynadığı için dramatik şişme kanıtı yok,
   ama tampon varsayılmalı.
2. **`guvenlik` F1=0.571** — yağmacılık/asayiş sinyali zayıf (test TP=2, FP=1, FN=2).
   Production tüketiciler için ayrı recall-öncelikli eşik önerilir.
3. **`bilgi_paylasimi` F1=0.681** — CV eşiği 0.87 tutucu; "haber alamıyoruz" tarzı
   ifadeler sistematik olarak `arama_kurtarma`'ya kayıyor. Etiket-tanımı + threshold sorunu.
4. **Rare-label F1=1.0 saturation** — saglik (pos=7), altyapi (pos=3), psikolojik (pos=1)
   test setinde F1=1.0 ama bu kalibre başarı değil, tek-tahmin saturasyonu. Silver prior
   olmadan F1=0 kollapsından kurtardı — nitel sinyal değerli, metrik değil.
5. **Küçük canonical test (194 satır)** — rare etiketlerde 1 tahmin flip'i ±0.2 skor
   salınabilir.
6. **Web app/API yok** — step 13 `not_applicable`. `requirements_dashboard.txt` intent
   sinyali ama consumer kodu yok.

## 9. Release

`release/need_classifier_v2/` — pointer-based paket (ağırlıklar kopyalanmadı).
Thresholds, label_columns, eval metrics, usage docs, adapter snippet, pointer'lar dahil.
**Status: provisional canonical** — blocker yok, ama #1 ve #2-3 giderilmeden production
anlaşmasına girilmemeli.

## 10. Önerilen sonraki hamle

**En mantıklı tek hamle**: **silver v4 build with content-level dedup** (SHA1 + shingle
Jaccard ≥ 0.85). Bu tek değişiklik (1) content-overlap residual risk'ini kapatır, (2)
metriklerin üst-tampon tahminini sıfırlar ve (3) mevcut pipeline'a minimum dokunuşla
entegre olur. Eşik yeniden tune edilir, v3-v4 karşılaştırılır; tampon gerçekten varsa
burada görünür. Bu olmadan `guvenlik`/`bilgi_paylasimi` için recall-öncelikli
threshold tasarımı *yanlış tabana* kurulmuş olur.

---

**Referans artifact'lar** (tam liste: `release/need_classifier_v2/artifacts/POINTERS.md`):

- Selection: `models/final/selection.json` · `docs/final_model_selection.md`
- Comparison: `data/analysis/experiment_comparison_v3_leakfree.{md,json,csv}`
- Error analysis: `data/analysis/error_analysis_v2_leakfree.{md,json}` + FP/FN/slices CSV
- Leak + overlap: `data/analysis/step7_leak_audit_v3.md` · `content_overlap_audit_v2_leakfree.md`
- Tahmin QA: `data/analysis/prediction_qa_v2_final.{md,json}`
- Web app status: `docs/web_app_integration_status.md`
