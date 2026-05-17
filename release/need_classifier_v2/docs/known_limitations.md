# Known limitations — v2 final

Bu liste seçim kararında da (step 9) ve tahmin QA'sında da (step 10) belgelendi.
Release tüketicileri karar vermeden önce okumalı.

## 1) Content-level overlap residual risk

- id-level leak kapalı: 0/1934 gold id silver havuzunda.
- Ama normalize-edilmiş metin overlap'i devam ediyor:
  - test: 35/194 exact (~18%), 59/194 near-dup Jaccard>=0.85 (~30%)
  - val : 42/193 exact (~22%), 63/193 near-dup (~33%)
  - train: 330/1547 exact (~21%), 499/1547 near-dup (~32%)
- v2 (id-contaminated) → v3 (id-clean) geçişinde f1_micro yalnızca +0.0120 oynadı;
  bu yüzden content-overlap'in dramatik metrik şişirmesine dair güçlü kanıt yok.
- Yine de gerçek genelleme F1'inin buradaki rakamların **~0.01–0.02 altında** olabileceğini
  varsaymak makul.
- Fix plan: step 10+ bir silver yeniden-kurma turunda SHA1 + shingle-Jaccard tabanlı
  content-level dedup uygula, tekrar eğit ve compare et.

## 2) `guvenlik` F1=0.571

- Test pozitif=4 (TP=2, FP=1, FN=2).
- Yağmacılık/asayiş çağrılarında model sinyali zayıf.
- 63k tahmin havuzunda pozitif oran %0.34 — muhtemelen gerçek oranın altında.
- Production önerisi: `guvenlik` için recall-öncelikli ayrı bir eşik.

## 3) `bilgi_paylasimi` F1=0.681

- CV eşiği 0.87 çok tutucu.
- 9 FN'den 4'ü model "arama_kurtarma" dediği için oluştu.
- "Haber alamıyoruz, gören var mı" tarzı ifadeler sistematik olarak `arama_kurtarma`'ya kayıyor.
- `info_v1` postprocess profili bu zayıflığı kısmen azaltır: canonical eşikleri korur,
  güçlü bilgi-paylaşımı dilinde ve `prob_bilgi_paylasimi >= 0.20` iken etiketi ekler.
- Bu yine de veri/guideline problemini çözmez; "haber alamıyoruz / gören duyan var mı"
  alt tipi yeni gold örneklerle güçlendirilmeli.

## 4) Rare-label F1=1.0 saturation

- `saglik` (pos=7), `altyapi` (pos=3), `psikolojik` (pos=1) test setinde F1=1.0.
- Bu **kalibre başarı değil**, tek-tahmin saturasyon seviyesi.
- Gerçek prevalans üzerinde precision/recall çok daha yüksek varyansla dolanır.
- Silver pretrain bu etiketleri "tahmin edilebilir" hâle getirdi (silver-siz modellerde F1=0
  kollapsından kurtardı) — bu nitel sinyal değerli, metrik değeri değil.

## 5) Küçük canonical test seti (194 satır)

- Aggregate metrikler geniş CI'lere sahip.
- Rare etiketlerde 1–4 pozitif tek tahmin flip'i ile ±0.2 skor salınabilir.

## 6) Silver sinyal rule-based

- Silver etiketleri `scripts/ai_prefill_annotations.py` kurallarıyla üretildi; gold değil.
- Silver baseline'ın val/test f1_micro ≈ 0.99 değerleri *silver etiketleriyle* tutarlı
  olduğunu gösterir, gold'a uyumu ölçmez.
- Gold'daki gerçek dağılıma karşı rule-fit bias riski vardır; step 8 hata analizi bu biasın
  asimilasyonlarını tartışır.

## 7) Havuz prior'ı (arama_kurtarma)

- 63k tahmin havuzu `emergency_geolocated_96k.csv` üzerinde `dedup-by-id` ile üretildi.
- Girdi kendisi "acil yardım + konum" filtresinden geçmiş olduğu için `arama_kurtarma`
  oranı %62.83 — bu over-fire değil, **prior match**.
- Tüketiciler bu havuz dışında bir kümede çok daha düşük `arama_kurtarma` oranı beklemeli.

## 8) Web app / API adapter bu release'de YOK

- Step 13 bu repoda `not_applicable` — tamamlanmış bir dashboard/API kodu bulunmadı
  (`requirements_dashboard.txt` mevcut ama karşılığı yok).
- Tüketici tarafı kendi adapter'ını yazmak zorunda; `examples/adapter_snippet.py` minimal
  referans verir.

## Fix-it önerileri (kısa)

| # | öneri | etki |
|---|---|---|
| 1 | Silver'da content-level dedup + retrain | genelleme tamponunu kapatır |
| 2 | `bilgi_paylasimi` için `info_v1` postprocess + yeni gold örnek | production coverage |
| 3 | Rare-label için hedefli annotation turu (güvenlik +50 pozitif gibi) | metriği kalibre eder |
| 4 | Test set genişletme (194 → 500+ satır) | CI daraltır |
