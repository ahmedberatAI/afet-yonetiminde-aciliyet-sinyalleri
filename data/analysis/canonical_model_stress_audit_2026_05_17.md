# Canonical Model Stress Audit - 2026-05-17

Bu rapor canonical modeli yeniden yükleyip test setini tekrar koşturur; model, eşik ve canonical prediction artefaktlarını değiştirmez.

## 1. Baseline doğrulama

- Selection experiment: `exp3_silver_then_gold_v3_exgold`
- Model dir: `C:\Users\omen\Desktop\afetYonetimi_colab\models\exp3_silver_then_gold_v3_exgold\final`
- Test CSV: `C:\Users\omen\Desktop\afetYonetimi_colab\data\modeling\need_classification_gold_combined\test.csv`
- Runtime: `cpu`, torch `2.12.0+cpu`
- Reproduced metrics: F1 micro=0.8998, F1 macro=0.8753, P micro=0.9154, R micro=0.8846.
- Selection delta: micro=+0.00000000, macro=+0.00000000.
- Canonical prediction audit: exists=True, rows=63180, meta canonical=True, experiment=`exp3_silver_then_gold_v3_exgold`.
- Dashboard meta uyumu: labels=True, thresholds=True.

## 2. Etiket bazlı sonuç

| label | support | threshold | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arama_kurtarma` | 111 | 0.60 | 0.964 | 0.973 | 0.969 | 108 | 4 | 3 |
| `saglik` | 7 | 0.90 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| `barinma` | 12 | 0.93 | 1.000 | 0.917 | 0.957 | 11 | 0 | 1 |
| `gida_su` | 10 | 0.14 | 1.000 | 0.800 | 0.889 | 8 | 0 | 2 |
| `altyapi` | 3 | 0.73 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
|  **`guvenlik`** | 4 | 0.74 | 0.667 | 0.500 | 0.571 | 2 | 1 | 2 |
| `lojistik` | 35 | 0.91 | 0.824 | 0.800 | 0.812 | 28 | 6 | 7 |
| `psikolojik` | 1 | 0.86 | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
|  **`bilgi_paylasimi`** | 25 | 0.87 | 0.727 | 0.640 | 0.681 | 16 | 6 | 9 |

## 3. Zayıf etiketler

### `guvenlik`
- Test desteği 4 pozitif; TP=2, FP=1, FN=2, P=0.667, R=0.500, F1=0.571.
- CV eşiği 0.74. Test üzerinde tanısal en iyi eşik 0.39 ile F1=0.667 görünüyor; bu değer seçim için kullanılmamalı, sadece hassasiyet sinyali.
- FN'ler eşiğe çok yakın değil; salt threshold düşürmek bütün hatayı açıklamıyor.
- FN cofire: `arama_kurtarma` x1, `barinma` x1, `gida_su` x1.
- FP satırlarında gerçek diğer etiketler: `arama_kurtarma` x1, `bilgi_paylasimi` x1.
- Anahtar kelime kapsaması: gold pozitif 4/4 satırda yakalandı; FN içinde keyword=2, keyword yok=0.
- Kritik FN örnekleri:
  - id=1623462499705126912 p=0.113, gold=arama_kurtarma,barinma,gida_su,guvenlik, pred=arama_kurtarma,barinma,gida_su - ‼️ ACİL GIDA YİYECEK ÇADIR ‼️ 40 kişi mahsur kalmış durumda. Yardım ulaşamamış aç susuz ve evsizler lütfen yardım edin Turunçlu mah. çağla sk no13 Hatay-Defne Hırsızlar dadanmış, asayiş yok! Süleyman Ertuğrul: 05364801765 @BabalaTv @barisatay @haluklevent @OguzhanUgur #deprem
  - id=1623636721182154752 p=0.399, gold=guvenlik, pred=- - İl dışı, deprem bölgesine yakın çalıştığım bir Mah. aradılar, sizi de merak ettik iyi misiniz diye. Millet olarak gerçekten vicdanlı ve sag duyuluyuz. İçimizden yağmacı, dalgaci, fırsatçı üç beş kötü çıksa da, memleketimiz insanı genel olarak iyi kalpli ve yardımseverdir.
- Kritik FP örnekleri:
  - id=1622435007158419456 p=0.871, gold=arama_kurtarma,bilgi_paylasimi, pred=arama_kurtarma,guvenlik - Malatya dogansehir kurucaova mahallesi haber alamiyorum. Yurtdisindayim. Teyzelerime ulasamiyorum. Lutfen haber alan varsa bildirsin. Orada yangin ve enkaz oldugu yaziliyor. Haber alan var mi lutfen #deprem #malatya

### `bilgi_paylasimi`
- Test desteği 25 pozitif; TP=16, FP=6, FN=9, P=0.727, R=0.640, F1=0.681.
- CV eşiği 0.87. Test üzerinde tanısal en iyi eşik 0.75 ile F1=0.704 görünüyor; bu değer seçim için kullanılmamalı, sadece hassasiyet sinyali.
- FN'lerin 1 tanesi eşiğin 0.10 altında; threshold oynatması bazı kaçırmaları geri alabilir.
- FN cofire: `arama_kurtarma` x4, `altyapi` x2, `barinma` x1, `guvenlik` x1.
- FP satırlarında gerçek diğer etiketler: `arama_kurtarma` x1.
- Anahtar kelime kapsaması: gold pozitif 22/25 satırda yakalandı; FN içinde keyword=6, keyword yok=3.
- Kritik FN örnekleri:
  - id=1624930670869880833 p=0.026, gold=barinma,bilgi_paylasimi, pred=barinma - 🚨🚨🚨 ÖNEMLİ! Habertürk canlı yayından iki önemli duyuru: * İstanbul'a gelen tüm depremzedelere gitmek istedikleri yer için ücretsiz transfer: İRTİBAT: 0532 795 75 10 Kahramanmaraş Necip Fazıl Mah. Acil Çadır ihtiyacı. İRTİBAT: Betül Öğretmen 0546 811 22 46 #deprem https://t.co/3a...
  - id=1622438179130195969 p=0.027, gold=bilgi_paylasimi, pred=- - atatürk mahallesi Atatürk caddesi no.94 samandağ hatay 🚨🚨 #DEPREMOLDU #deprem #acil #afad #hatay
  - id=1622706650443382799 p=0.030, gold=bilgi_paylasimi, pred=- - Adıyaman Gölbaşı Fatih mahallesi meriç caddesi acil destek lütfen @salihro @OguzhanUgur @haluklevent @ahbap @DepremDairesi #deprem #adiyamandeprem #adıyamanafad
  - id=1622706574450982913 p=0.185, gold=arama_kurtarma,altyapi,bilgi_paylasimi, pred=arama_kurtarma,altyapi - @gutitoli ELEKTRİK MAH. AKTAŞ SK. NO: 17 İÇ KAPI NO: 2 DEFNE / HATAY KADİR GÖZTEPE ARKADAŞIMIZDAN VE AİLESİNDEN HABER ALAMIYORUZ LÜTFEN YARDİMCİ OLUN İLETİŞİM BİLGİSİ 0541 494 0618 LÜTFEN GOREN DUYAN YAKİN OLAN YARDİMCİ OLSUN #Hatay #hataydepremi #ENKAZALTİNDAYİM #EnkazBilgisi #...
- Kritik FP örnekleri:
  - id=1622479252552617984 p=0.974, gold=-, pred=bilgi_paylasimi - Esenlik mahallesi hasan sokak mehdican apt. Antakya / HATAY “HOROZ AİLESİNDEN HABER ALINAMIYOR ORDA OLUP GÖREN VARSA BURAYA YANITLAYABİLİR Mİ ARKADAŞLAR ASKER ARKADAŞIMIN AİLESİ HABER ALAMIYORUZ” #hatay #deprem #AFAD #afad #ACİL #antakya #deprem #iskenderun
  - id=1622880337230499841 p=0.967, gold=arama_kurtarma, pred=arama_kurtarma,bilgi_paylasimi - Akevler mahallesi 424. sokak Melis apartmanı a blok kat 3 daire 7 Antakya/Hatay *Kaan Yılmaz ve ailesi, göçük altında olduğu düşünülüyor. Bilgi edinebilenler lütfen ulaşsın. #deprem #depremhatay @AFADBaskanlik @DepremDairesi @AFADHatay @sjw_erlik @OguzhanUgur @cerrahpasaIi https...
  - id=1622443888425275394 p=0.964, gold=-, pred=bilgi_paylasimi - Urgenpasa mahallesi 19. Sokak Hatay Antakya haber alamıyorum lütfen yardımcı olur musunuz bilgisi olan yazabilir mi #Hatay #deprem #ACİL
  - id=1623565010927603712 p=0.943, gold=-, pred=bilgi_paylasimi - Huzurevleri mahallesi 77201 sokak ihsan bayram sitesi Çukurova/Adana Yonca Baycal Ağahan Baycal Esila Baycal (6 yaşında) Azra Baycal (4 aylık) Bir haber olursa lütfen #deprem #ACILYARDIM #seferberlik @haluklevent @OguzhanUgur @Ahbap_Adana @AFAD_Adana

## 4. Threshold hassasiyeti

| label | current_thr | current_F1 | diagnostic_best_thr | diagnostic_best_F1 | current_FN | best_FN | best_FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `arama_kurtarma` | 0.60 | 0.969 | 0.60 | 0.969 | 3 | 3 | 4 |
| `saglik` | 0.90 | 1.000 | 0.90 | 1.000 | 0 | 0 | 0 |
| `barinma` | 0.93 | 0.957 | 0.93 | 0.957 | 1 | 1 | 0 |
| `gida_su` | 0.14 | 0.889 | 0.14 | 0.889 | 2 | 2 | 0 |
| `altyapi` | 0.73 | 1.000 | 0.73 | 1.000 | 0 | 0 | 0 |
| `guvenlik` | 0.74 | 0.571 | 0.39 | 0.667 | 2 | 1 | 2 |
| `lojistik` | 0.91 | 0.812 | 0.91 | 0.812 | 7 | 7 | 6 |
| `psikolojik` | 0.86 | 1.000 | 0.86 | 1.000 | 0 | 0 | 0 |
| `bilgi_paylasimi` | 0.87 | 0.681 | 0.75 | 0.704 | 9 | 6 | 10 |

> Not: `diagnostic_best_thr` test setinden hesaplandığı için production eşiği olarak önerilmez; sadece mevcut eşiğin ne kadar hassas olduğunu gösterir.

## 5. Bucket analizi

### text_length
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| short(<120) | 6 | 2 | 0.333 |
| medium(120-219) | 70 | 16 | 0.229 |
| long(>=220) | 118 | 20 | 0.169 |

### aciliyet_0_3
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| 0 | 7 | 2 | 0.286 |
| 2 | 19 | 5 | 0.263 |
| 1 | 56 | 12 | 0.214 |
| 3 | 112 | 19 | 0.170 |

### urgency_score
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| 0 | 7 | 3 | 0.429 |
| 1 | 16 | 6 | 0.375 |
| 2 | 8 | 3 | 0.375 |
| 4 | 31 | 10 | 0.323 |
| 6 | 11 | 2 | 0.182 |
| 5 | 35 | 6 | 0.171 |
| 3 | 24 | 3 | 0.125 |
| 7 | 49 | 5 | 0.102 |
| 9 | 8 | 0 | 0.000 |
| 10 | 2 | 0 | 0.000 |

### gold_label_count
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| 3+ | 4 | 3 | 0.750 |
| 2 | 45 | 12 | 0.267 |
| 0 | 40 | 8 | 0.200 |
| 1 | 105 | 15 | 0.143 |

### province_min3
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| Adıyaman | 17 | 7 | 0.412 |
| Unknown | 13 | 3 | 0.231 |
| Hatay | 85 | 18 | 0.212 |
| Adana | 33 | 6 | 0.182 |
| (missing) | 28 | 4 | 0.143 |
| Kahramanmaraş | 6 | 0 | 0.000 |
| Gaziantep | 5 | 0 | 0.000 |
| Osmaniye | 4 | 0 | 0.000 |

### province_district_min3
| group | rows | error_rows | error_rate |
|---|---:|---:|---:|
| Adıyaman / Gölbaşı | 15 | 6 | 0.400 |
| Hatay / İskenderun | 5 | 2 | 0.400 |
| Hatay / Altınözü | 56 | 14 | 0.250 |
| Unknown / Unknown | 13 | 3 | 0.231 |
| Adana / Aladağ | 33 | 6 | 0.182 |
|  /  | 28 | 4 | 0.143 |
| Hatay / Antakya | 24 | 2 | 0.083 |
| Kahramanmaraş / Dulkadiroğlu | 4 | 0 | 0.000 |
| Osmaniye / Düziçi | 4 | 0 | 0.000 |
| Gaziantep / Karkamış | 3 | 0 | 0.000 |

## 6. Belirsiz örnekler

- Herhangi bir eşiğe 0.05 yakın satır: 9 / 194.
- Hatalı olup herhangi bir eşiğe 0.05 yakın satır: 4.
- Eşiğe en yakın hatalı örnekler:
  - `bilgi_paylasimi` p=0.877, thr=0.870, gold=-, pred=bilgi_paylasimi - Hatayda hâla kendilerinden haber alamıyoruz lütfen yardım edin (09.02.2023) Odabaşı mah 1022 sokak dağ apartmanı hatay/antakya @AKUT_Dernegi @DepremDairesi @AFADBaskanlik #deprem #AFADhatay #YardimEdin
  - `arama_kurtarma` p=0.620, thr=0.600, gold=-, pred=arama_kurtarma - Ses var ; 1 kişiden ses var Odabaşı mahallesi Gürses sokak Prestij asi apartmanı Hatay / Antakya 05321511455 Uğur yaşar TAM ADRESİ AŞŞAĞIDA #deprem #afad #sesvar #ahbap #ahbapacil #hatay https://t.co/rHBMySVJJV
  - `bilgi_paylasimi` p=0.840, thr=0.870, gold=arama_kurtarma,bilgi_paylasimi, pred=arama_kurtarma,guvenlik - Malatya dogansehir kurucaova mahallesi haber alamiyorum. Yurtdisindayim. Teyzelerime ulasamiyorum. Lutfen haber alan varsa bildirsin. Orada yangin ve enkaz oldugu yaziliyor. Haber alan var mi lutfen #deprem #malatya
  - `bilgi_paylasimi` p=0.912, thr=0.870, gold=-, pred=bilgi_paylasimi - @nediyorsunyasen @DepremDairesi @AFAD_gonullu ❗️ACİLLL❗️ BERAT UZUNESER, KAHRAMANMARAŞ TÜRKOĞLU ŞEKEROBA MAHALLESI saatlerdir haber almaya çalışıyoruz bu çevrede gören bilen olursa bana yazabilir mi? 05530164977 #deprem #acil #Kahramanmaras @DepremDairesi @AFAD_gonullu
  - `barinma` p=0.999, thr=0.930, gold=barinma,bilgi_paylasimi, pred=barinma - 🚨🚨🚨 ÖNEMLİ! Habertürk canlı yayından iki önemli duyuru: * İstanbul'a gelen tüm depremzedelere gitmek istedikleri yer için ücretsiz transfer: İRTİBAT: 0532 795 75 10 Kahramanmaraş Necip Fazıl Mah. Acil Çadır ihtiyacı. İRTİBAT: Betül Öğretmen 0546 811 22 46 #deprem https://t.co/3a...
  - `barinma` p=0.999, thr=0.930, gold=arama_kurtarma,barinma,gida_su,guvenlik, pred=arama_kurtarma,barinma,gida_su - ‼️ ACİL GIDA YİYECEK ÇADIR ‼️ 40 kişi mahsur kalmış durumda. Yardım ulaşamamış aç susuz ve evsizler lütfen yardım edin Turunçlu mah. çağla sk no13 Hatay-Defne Hırsızlar dadanmış, asayiş yok! Süleyman Ertuğrul: 05364801765 @BabalaTv @barisatay @haluklevent @OguzhanUgur #deprem
  - `bilgi_paylasimi` p=0.943, thr=0.870, gold=-, pred=bilgi_paylasimi - Huzurevleri mahallesi 77201 sokak ihsan bayram sitesi Çukurova/Adana Yonca Baycal Ağahan Baycal Esila Baycal (6 yaşında) Azra Baycal (4 aylık) Bir haber olursa lütfen #deprem #ACILYARDIM #seferberlik @haluklevent @OguzhanUgur @Ahbap_Adana @AFAD_Adana
  - `lojistik` p=0.986, thr=0.910, gold=altyapi,lojistik, pred=arama_kurtarma,altyapi,lojistik - Hatay Antakya Atatürk caddesi ürgen paşa mahallesi özköse apartmanı yanında doğalgaz kaçağından patlama oldu ACİL ekip gerekiyor ‼️ @AFADHatay @AFADBaskanlik @ahbap @haluklevent #deprem #enkazaltındayım #hataydeprem

## 7. Challenge set

Bu bölüm sentetik/elle yazılmış nitel problardır; skor olarak okunmamalı.
| probe | expected | predicted | top probs |
|---|---|---|---|
| `security_direct_looting` | guvenlik | guvenlik,psikolojik | guvenlik=0.99, psikolojik=0.98, bilgi_paylasimi=0.82 |
| `security_plus_basic_needs` | barinma,gida_su,guvenlik | barinma,gida_su,guvenlik,psikolojik | barinma=1.00, guvenlik=0.97, psikolojik=0.95 |
| `security_fire_ambiguous` | guvenlik,altyapi | arama_kurtarma,guvenlik,lojistik,psikolojik | lojistik=1.00, guvenlik=0.98, psikolojik=0.97 |
| `info_only_missing_relative` | bilgi_paylasimi | bilgi_paylasimi | bilgi_paylasimi=1.00, guvenlik=0.09, arama_kurtarma=0.08 |
| `info_plus_rescue` | arama_kurtarma,bilgi_paylasimi | arama_kurtarma,bilgi_paylasimi | bilgi_paylasimi=0.99, arama_kurtarma=0.64, guvenlik=0.05 |
| `info_address_only` | bilgi_paylasimi | - | bilgi_paylasimi=0.67, arama_kurtarma=0.02, guvenlik=0.02 |
| `logistics_rescue_machine` | arama_kurtarma,lojistik | arama_kurtarma,lojistik | lojistik=1.00, arama_kurtarma=0.99, guvenlik=0.01 |
| `basic_needs_no_rescue` | barinma,gida_su | barinma,gida_su | gida_su=1.00, barinma=1.00, guvenlik=0.03 |
| `announcement_distribution` | bilgi_paylasimi,gida_su | gida_su | gida_su=1.00, barinma=0.12, lojistik=0.07 |
| `psychological_support` | psikolojik | guvenlik,psikolojik | psikolojik=1.00, barinma=0.81, guvenlik=0.76 |

## 8. Sonuç ve öneriler

- Baseline korunmalı: canonical model ve CV eşikleri yeniden üretildi; bu rapor model seçimini değiştirmiyor.
- `bilgi_paylasimi` için en hızlı güvenli deney, production amaçlı ayrı bir recall-senaryosu eşiği çalışmak: CV eşiği 0.87, mevcut recall=0.64; test üstündeki tanısal en iyi eşik 0.75. Bu eşik doğrudan seçilmemeli, OOF/validation ile doğrulanmalı.
- `bilgi_paylasimi` FN'lerinde `arama_kurtarma` cofire belirgin; guideline'a 'haber alamıyoruz/gören duyan var mı' ifadeleri arama-kurtarma ile birlikte de `bilgi_paylasimi` alır notu eklenmeli.
- `guvenlik` için skor çok küçük desteğe dayanıyor (test pozitif=4, pool pozitif=39); hızlı iyileştirme threshold değil, keyword/active-learning adaylarından çift etiketlemeli yeni pozitif örnek toplamak.
- `psikolojik` testte kusursuz görünüyor ama challenge problarında beklenmeyen ateşlemeler var (security_direct_looting, security_plus_basic_needs, security_fire_ambiguous); bu rare-label skorunu kalibre başarı olarak okumamak gerekir.
- `bilgi_paylasimi` challenge problarında kısa/adres ağırlıklı bilgi çağrılarını kaçırabiliyor (info_address_only, announcement_distribution); guideline ve veri genişletme planında bu alt tip ayrı kovalanmalı.
- Büyük iyileştirme için silver havuzu içerik düzeyinde dedup edilip `guvenlik` ve `bilgi_paylasimi` odaklı 50-100 yeni gold pozitif eklenmeli; ardından threshold CV tekrar koşulmalı.
- Dashboard tarafında değişiklik gerekmedi: bundled label/threshold metadata canonical ile eşleşiyor ve legacy 63k fallback geri getirilmedi.
