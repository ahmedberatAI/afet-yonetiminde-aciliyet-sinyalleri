# Metrics summary — v2 final (leak-free)

## Kaynak

- Canonical test set: [data/modeling/need_classification_gold_combined/test.csv](../../../data/modeling/need_classification_gold_combined/test.csv) — 194 satır
- Eşik kaynağı: `thresholds/thresholds_cv.json` (CV-tuned, OOF global, multilabel-stratified 5-fold)
- Eval artifact: [artifacts/eval_test_tuned.json](../artifacts/eval_test_tuned.json)

## Aggregate (CV-tuned thresholds)

| metrik | değer |
|---|---|
| f1_micro | 0.8998 |
| f1_macro | 0.8753 |
| precision_micro | 0.9154 |
| recall_micro | 0.8846 |

## Per-label F1

| label | test pozitif | threshold | F1 | not |
|---|---|---|---|---|
| arama_kurtarma | 111 | 0.60 | 0.969 | güçlü |
| saglik | 7 | 0.90 | 1.000 | **rare-label saturation** (küçük destek) |
| barinma | 12 | 0.93 | 0.957 | güçlü |
| gida_su | 10 | 0.14 | 0.889 | güçlü |
| altyapi | 3 | 0.73 | 1.000 | **rare-label saturation** (küçük destek) |
| guvenlik | 4 | 0.74 | **0.571** | **zayıf** (TP=2, FP=1, FN=2) |
| lojistik | 35 | 0.91 | 0.812 | iyi |
| psikolojik | 1 | 0.86 | 1.000 | **rare-label saturation** (tek pozitif) |
| bilgi_paylasimi | 25 | 0.87 | **0.681** | **zayıf** (CV eşiği çok tutucu) |

## Inference-layer postprocess (`info_v1`)

`info_v1` canonical eşiği değiştirmez; güçlü bilgi-paylaşımı dili yakalandığında ve
`prob_bilgi_paylasimi >= 0.20` olduğunda `bilgi_paylasimi` etiketini ekler. Seçim OOF +
validation üzerinde yapıldı; test yalnızca regresyon kontrolü olarak raporlandı.

| dataset | base micro F1 | info_v1 micro F1 | base bilgi F1 | info_v1 bilgi F1 |
|---|---:|---:|---:|---:|
| OOF | 0.9087 | 0.9136 | 0.8062 | 0.8525 |
| validation | 0.9040 | 0.9204 | 0.8438 | 0.9429 |
| test (regression check) | 0.8998 | 0.9082 | 0.6809 | 0.7692 |

Kanıt: [postprocess_info_v1_validation_2026_05_17.md](../../../data/analysis/postprocess_info_v1_validation_2026_05_17.md).
Exact baseline için `--postprocess-profile none` kullanılabilir.

## Ranking (v3 leak-free comparison)

| rank | experiment_key | f1_micro | f1_macro |
|---|---|---|---|
| 1 | `exp3_silver_then_gold_v3_exgold` | 0.8998 | 0.8753 |
| 2 | `exp2_gold_v2_posw` | 0.7298 | 0.5560 |
| 3 | `exp1_gold_v2_bce` | 0.7325 | 0.4137 |

Tüm ranking'ler leak-free step 7 karşılaştırmasından gelir
([data/analysis/experiment_comparison_v3_leakfree.md](../../../data/analysis/experiment_comparison_v3_leakfree.md)).

## Generalization caveat

İçerik-seviyesi overlap auditi
([data/analysis/content_overlap_audit_v2_leakfree.md](../../../data/analysis/content_overlap_audit_v2_leakfree.md))
test setinin ~%18'inin silver havuzunda exact, ~%30'unun near-dup olarak tekrar ettiğini gösterdi.
Bu, gerçek genelleme F1'inin buradaki rakamların ~0.01–0.02 altında olabileceği anlamına gelir.
Seçim kararını etkilemez (exp1/exp2 silver kullanmaz; ranking olgusal olarak sağlam), ama
dashboard/API tüketicileri için rapor edilmelidir.
