# Selection summary — v2 final (leak-free)

Tam rasyoneli için [docs/final_model_selection.md](../../../docs/final_model_selection.md) veya
[models/final/selection.json](../../../models/final/selection.json).

## Kısa gerekçe (4 madde)

1. **Aggregate liderlik tutarlı.** `exp3_silver_then_gold_v3_exgold` canonical leak-free
   karşılaştırmada (experiment_comparison_v3_leakfree.*) hem f1_micro (0.8998) hem f1_macro
   (0.8753) sıralamasında birinci; ayrıca precision_micro (0.9154) ve recall_micro (0.8846)
   başlıklarında da önde. Kazanç tek boyutta değil.

2. **Silver pretrain'in rare-label faydası gerçek.** Silver prior olmadan `altyapi`, `saglik`,
   `psikolojik` test setinde F1=0 ile çöküyor (exp1/çoğu exp2 konfigürasyonu). Silver ile bu
   etiketler tahmin edilebilir hâle geldi. Ancak test pozitifleri çok küçük (1–7) olduğundan
   F1=1.0 skorları **kalibre başarı değil**, tek-tahmin saturasyonu seviyesinde sinyal.

3. **`guvenlik` ve `bilgi_paylasimi` açık zayıf noktalar.** F1=0.571 ve F1=0.681 sırasıyla;
   `bilgi_paylasimi` exp1'in (F1=0.76) altında. Winner statusunu değiştirmezler ama
   production tüketiciler bilmeli.

4. **Content-level overlap residual risk.** id-level leak kapalı, ama silver havuzunda
   test satırlarının ~%18'i exact, ~%30'u near-dup olarak tekrar ediyor. v2→v3 (0% id overlap)
   geçişinde f1_micro yalnızca +0.0120 oynadığı için bu overlap'in dramatik metrik şişirmesine
   dair güçlü kanıt yok; ancak ~0.01–0.02 tampon varsayılabilir. Step 10+ bir silver
   yeniden-kurma turunda content-level dedup uygulanmalı.

## Bilinen sınırlamalar

- Küçük canonical test seti (194 satır) — rare-label F1 yüksek varyansa sahiptir.
- Silver etiketleri rule-based (`scripts/ai_prefill_annotations.py`); gold değil.
- Havuzun kendisi 'acil yardım + konum' filtresinden geçmiş — `arama_kurtarma` prevalansının
  63k üzerinde %62.83 olması bu prior'dan kaynaklanır, over-fire değil.

## Neden provisional, canonical olmasına rağmen?

Bu release canonical çünkü step 9 seçimi ve step 10 QA'sı bu sürümü işaretledi. Yine de
"provisional" etiketi açık, çünkü production-grade bir anlaşmaya gitmeden önce content-level
dedup (#4) ve `guvenlik`/`bilgi_paylasimi` için ikinci tur eşik tasarımı (#3) önerilir.
