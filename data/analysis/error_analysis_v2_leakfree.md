# Step 8 — Error Analysis (v2, leak-free)

Canonical winner'ın ([step 7 v3](experiment_comparison_v3_leakfree.md)) leak-free test setinde hata örüntüleri. Model seçimi, tahmin üretimi ve final pointer güncelleme bu adımda yapılmaz.

## 1. Kullanılan kaynaklar

- **Model**: `C:/Users/omen/Desktop/afetYonetimi_colab/models/exp3_silver_then_gold_v3_exgold/final`
- **Test CSV**: `C:/Users/omen/Desktop/afetYonetimi_colab/data/modeling/need_classification_gold_combined/test.csv` (194 satır)
- **Thresholds**: `C:/Users/omen/Desktop/afetYonetimi_colab/models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json` (type=`cv`, strategy=`oof_global`, k=5)
- **Comparison source**: `C:/Users/omen/Desktop/afetYonetimi_colab/data/analysis/experiment_comparison_v3_leakfree.json`
- **GPU**: NVIDIA GeForce RTX 5090 Laptop GPU (CUDA required; CPU fallback disabled)
- **Seed**: 42 — deterministic inference

## 2. Genel metrik bağlamı

- **f1_micro** = 0.8998, **f1_macro** = 0.8753, P_micro = 0.9154, R_micro = 0.8846.

### Per-label F1 özet

| label | thr | TP | FP | FN | TN | P | R | F1 | test_pozitif | pool_pozitif |
|---|---|---|---|---|---|---|---|---|---|---|
| `arama_kurtarma` | 0.60 | 108 | 4 | 3 | 79 | 0.96 | 0.97 | **0.969** | 111 | 1101 |
| `saglik` | 0.90 | 7 | 0 | 0 | 187 | 1.00 | 1.00 | **1.000** | 7 | 50 |
| `barinma` | 0.93 | 11 | 0 | 1 | 182 | 1.00 | 0.92 | **0.957** | 12 | 173 |
| `gida_su` | 0.14 | 8 | 0 | 2 | 184 | 1.00 | 0.80 | **0.889** | 10 | 110 |
| `altyapi` | 0.73 | 3 | 0 | 0 | 191 | 1.00 | 1.00 | **1.000** | 3 | 23 |
| `guvenlik` | 0.74 | 2 | 1 | 2 | 189 | 0.67 | 0.50 | **0.571** | 4 | 39 |
| `lojistik` | 0.91 | 28 | 6 | 7 | 153 | 0.82 | 0.80 | **0.812** | 35 | 300 |
| `psikolojik` | 0.86 | 1 | 0 | 0 | 193 | 1.00 | 1.00 | **1.000** | 1 | 11 |
| `bilgi_paylasimi` | 0.87 | 16 | 6 | 9 | 163 | 0.73 | 0.64 | **0.681** | 25 | 260 |

### Vurgulanan etiketler

- **`altyapi`** → F1=1.000 (thr=0.73, TP=3, FP=0, FN=0, test pozitif=3)
- **`guvenlik`** → F1=0.571 (thr=0.74, TP=2, FP=1, FN=2, test pozitif=4)
- **`psikolojik`** → F1=1.000 (thr=0.86, TP=1, FP=0, FN=0, test pozitif=1)
- **`bilgi_paylasimi`** → F1=0.681 (thr=0.87, TP=16, FP=6, FN=9, test pozitif=25)

## 3. Per-label error breakdown

### `arama_kurtarma` (thr=0.60)

- TP=108, FP=4, FN=3, TN=79 — P=0.964, R=0.973, F1=0.969
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1623418967464325123` p=0.041 aciliyet=3 — gold=`arama_kurtarma` | pred=`-`
    - Merkez/Adıyaman bozbey yolu alitaşı mahallesi 1223.sokak Bahçekent sitesi B blok zemin kat yardım gitmemiş, 100 kişi olduğu söyleniyor. @AFADAdiyaman @AFADBaskanlik @ahbap @haluklevent @BabalaTv @OguzhanUgur #deprem #adıyaman #AdiyamanYardimBekliyor #acildeprem https://t.co/qfVU5
  - `id=1624482947301535747` p=0.089 aciliyet=3 — gold=`arama_kurtarma,lojistik` | pred=`-`
    - ANTAKYA-madencilere ihtiyac var ACİL Ekinci Mahallesi İnönü Bulvarı Alya Uçar Apt. No:128 Binanın altında Tutku Past. No: 0532 744 96 46 ALBAY YUSUF YONCA #deprem #acil
  - `id=1622910084291362818` p=0.171 aciliyet=3 — gold=`arama_kurtarma` | pred=`-`
    - @uberkuloz #sadıkmurat #hatayyardimbekliyor Antakya Cebrail Mahallesi Emlak Bank Evleri 1. Kisim 6f blok SADIK MURAT IPEK ses veriyormus cevresindekiler teyit etti nolur kardesimize yardim edin sıkısmıs ama gun isigini goruyormus ses veriyormus #deprem @haluklevent @AFADBaskanlik
- **En kritik FP örnekleri** (en yüksek prob):
  - `id=1624092271275651095` p=0.999 aciliyet=1 — gold=`lojistik` | pred=`arama_kurtarma`
    - ACİLLLL SES DİNLEME ALETİNE İHTİYAÇ VAR ACİLLLLL!!!!!!!!! Antakya Ekinci mahallesi , güneş caddesi , İbrahim elif apartmanı , no18 Zehra karatoyuk #hatayantakya #deprem #afad #SESVARACiL #sesvarEnkaz #ahbab
  - `id=1622675566666125330` p=0.998 aciliyet=1 — gold=`-` | pred=`arama_kurtarma`
    - İnsanların umudu tükenmişiz Antakya Gazi Mahallesi. Acilll! #deprem #ENKAZALTİNDAYİM #Hatay https://t.co/bsCchr2zbH
  - `id=1622648265995845660` p=0.970 aciliyet=1 — gold=`altyapi,lojistik` | pred=`arama_kurtarma,altyapi,lojistik`
    - Hatay Antakya Atatürk caddesi ürgen paşa mahallesi özköse apartmanı yanında doğalgaz kaçağından patlama oldu ACİL ekip gerekiyor ‼️ @AFADHatay @AFADBaskanlik @ahbap @haluklevent #deprem #enkazaltındayım #hataydeprem
  - `id=1623351862396059648` p=0.620 aciliyet=0 — gold=`-` | pred=`arama_kurtarma`
    - Ses var ; 1 kişiden ses var Odabaşı mahallesi Gürses sokak Prestij asi apartmanı Hatay / Antakya 05321511455 Uğur yaşar TAM ADRESİ AŞŞAĞIDA #deprem #afad #sesvar #ahbap #ahbapacil #hatay https://t.co/rHBMySVJJV

### `saglik` (thr=0.90)

- TP=7, FP=0, FN=0, TN=187 — P=1.000, R=1.000, F1=1.000
- FN yok.
- FP yok.

### `barinma` (thr=0.93)

- TP=11, FP=0, FN=1, TN=182 — P=1.000, R=0.917, F1=0.957
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1623643555645325312` p=0.171 aciliyet=1 — gold=`barinma` | pred=`-`
    - Ailem akrabalarımla dışarıda kalıyor, çok kalabalıklar. 50'den fazla kişiler aralarında 3 bebek, bir yaşlı ve 1 çocuk da var. İlk günden beri yardım alamadılar. Adres: Hatay/ Antakya karaali mahallesi birinci cadde 4.c sokak. İletişim: 05352028207 05538651417 #deprem #Yardim
- FP yok.

### `gida_su` (thr=0.14)

- TP=8, FP=0, FN=2, TN=184 — P=1.000, R=0.800, F1=0.889
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1622878618564034561` p=0.002 aciliyet=3 — gold=`arama_kurtarma,gida_su,lojistik` | pred=`arama_kurtarma,lojistik`
    - @ProfDemirtas @ProfDemirtas Mustafa Kemal mah 575/3 sokak iskenderun/Hatay.Arkadasimin ailesi gocuk altinda.Bina tamamen cokmus.Kurtulanlarin yiyecek ve su gocuk altinda kalanlarin kurtarilmasi icin ise ekipman ve insan gucune ihtiyac var.Lutfen sesimiz olun.Deprem anindan beri y
  - `id=1623341986517229570` p=0.002 aciliyet=2 — gold=`gida_su` | pred=`-`
    - BÜYÜKDALYAN MERKEZ MAHALLESİ'NE ACİL YARDIM!! YİYECEK SADECE SU VE BİSKÜVİ VARMIŞ, ULAŞIM İMKANSIZ OLDUĞU İÇİN MERKEZE GELEN HİÇ BİR YARDIM BURAYA GELMEMİŞ!! ULAŞILACAK TELEFON 05327313313 Hatay @AFADBaskanlik @haluklevent #acil #deprem
- FP yok.

### `altyapi` (thr=0.73)

- TP=3, FP=0, FN=0, TN=191 — P=1.000, R=1.000, F1=1.000
- FN yok.
- FP yok.

### `guvenlik` (thr=0.74)

- TP=2, FP=1, FN=2, TN=189 — P=0.667, R=0.500, F1=0.571
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1623462499705126912` p=0.113 aciliyet=3 — gold=`arama_kurtarma,barinma,gida_su,guvenlik` | pred=`arama_kurtarma,barinma,gida_su`
    - ‼️ ACİL GIDA YİYECEK ÇADIR ‼️ 40 kişi mahsur kalmış durumda. Yardım ulaşamamış aç susuz ve evsizler lütfen yardım edin Turunçlu mah. çağla sk no13 Hatay-Defne Hırsızlar dadanmış, asayiş yok! Süleyman Ertuğrul: 05364801765 @BabalaTv @barisatay @haluklevent @OguzhanUgur #deprem
  - `id=1623636721182154752` p=0.399 aciliyet=2 — gold=`guvenlik` | pred=`-`
    - İl dışı, deprem bölgesine yakın çalıştığım bir Mah. aradılar, sizi de merak ettik iyi misiniz diye. Millet olarak gerçekten vicdanlı ve sag duyuluyuz. İçimizden yağmacı, dalgaci, fırsatçı üç beş kötü çıksa da, memleketimiz insanı genel olarak iyi kalpli ve yardımseverdir.
- **En kritik FP örnekleri** (en yüksek prob):
  - `id=1622435007158419456` p=0.871 aciliyet=3 — gold=`arama_kurtarma,bilgi_paylasimi` | pred=`arama_kurtarma,guvenlik`
    - Malatya dogansehir kurucaova mahallesi haber alamiyorum. Yurtdisindayim. Teyzelerime ulasamiyorum. Lutfen haber alan varsa bildirsin. Orada yangin ve enkaz oldugu yaziliyor. Haber alan var mi lutfen #deprem #malatya

### `lojistik` (thr=0.91)

- TP=28, FP=6, FN=7, TN=153 — P=0.824, R=0.800, F1=0.812
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1623376743997313041` p=0.007 aciliyet=1 — gold=`lojistik` | pred=`-`
    - ANTAKYA - FARKLI YAŞAM RENDE SİTESİ Odabaşı Mah. Bektaş Sokak ACİL IŞIKLANDIRMA LAZIM! #hatay #antakyahatay #antakya #deprem #antakyadeprem #antakyadepremi #DepremAcil #ahbaphatay
  - `id=1624482947301535747` p=0.010 aciliyet=3 — gold=`arama_kurtarma,lojistik` | pred=`-`
    - ANTAKYA-madencilere ihtiyac var ACİL Ekinci Mahallesi İnönü Bulvarı Alya Uçar Apt. No:128 Binanın altında Tutku Past. No: 0532 744 96 46 ALBAY YUSUF YONCA #deprem #acil
  - `id=1622436763951157248` p=0.013 aciliyet=3 — gold=`arama_kurtarma,lojistik` | pred=`arama_kurtarma`
    - ACİİİLLLL!!!!!! Hatay Antakya kuzeytepe mahallesi 13. Sokak no 10 bir aile sıkıştı çıkamıyor itfaiye kepce viryler ara ENKAZ ALTINDA KALANLAR ACİİİLLL YARDIM MÜDAHALE EDİLMESİ GEREKİYOR HATAY #deprem #hatay
  - `id=1622677283684159507` p=0.014 aciliyet=3 — gold=`arama_kurtarma,lojistik` | pred=`arama_kurtarma`
    - Hatay İskenderun sanayi Sakarya Mah.288/19. Sok. Özincikent sit. Arkadaşlarım enkazda Hiç kimse yok Siviller kendi imkanları ile çıkarıyor Hayatlarından endişeliyiz arkadaşımın Cocuklarida enkazda🥺 Lütfen adresi dağıtın lütfen 🥺 Bir kardeşimizin isteği😔#deprem #seferberlik
  - `id=1624092271275651095` p=0.019 aciliyet=1 — gold=`lojistik` | pred=`arama_kurtarma`
    - ACİLLLL SES DİNLEME ALETİNE İHTİYAÇ VAR ACİLLLLL!!!!!!!!! Antakya Ekinci mahallesi , güneş caddesi , İbrahim elif apartmanı , no18 Zehra karatoyuk #hatayantakya #deprem #afad #SESVARACiL #sesvarEnkaz #ahbab
- **En kritik FP örnekleri** (en yüksek prob):
  - `id=1622681485168857093` p=0.999 aciliyet=3 — gold=`arama_kurtarma` | pred=`arama_kurtarma,lojistik`
    - #İskenderun Mustafa Kemal Mah. 544 ve 550.sokak kesişiminde 36,5671860, 36,1506040 koordinatlarında göçük altında anne Yeşim Parlak, baba Bülent Parlak ve çocukları Öykü Parlak göçük altında hala yardım beklemektedir. #Teyitli #ACİL #HATAY #DEPREMOLDU #deprem https://t.co/cvZsrXl
  - `id=1622641240968658947` p=0.998 aciliyet=3 — gold=`arama_kurtarma` | pred=`arama_kurtarma,lojistik`
    - #deprem Turgut Reis Mahallesi Sakarya Caddesi No: 57 Adıyaman / Merkez Enkaz altında en az 5 kişi var acil yardıma ihtiyaç var AFAD ekipleri yok.
  - `id=1622901642835132418` p=0.998 aciliyet=3 — gold=`arama_kurtarma` | pred=`arama_kurtarma,lojistik`
    - @alniopulesitivt Sümerevler mahallesi 1709. Sk. 10B Adıyaman Merkez Bozdoğan oteli arkası, hayal cafe üstü ARKADAŞLARIM GÖÇÜK ALTINDALAR. ACİL İŞ MAKİNASI GEREKİYOR. LÜTFEN YARDIM EDİN! @AFADTurkiye @AFADAdiyaman @ahbap #Deprem
  - `id=1623296929483431937` p=0.997 aciliyet=3 — gold=`arama_kurtarma` | pred=`arama_kurtarma,lojistik`
    - eşi ve eşinin annesi babası da var Hatay armutlu mahallesine lütfen kurtarma ekipleri bir an önce yardım etsin bina adını bilmiyoruz ARMUTLU PAZAR YERİ karşısı 3 çocuklu aile hala enkaz altında dün sesleri duyulmuş bugün ses yok lütfen yardım ulaştırın acil şekilde #deprem
  - `id=1623951000062025728` p=0.993 aciliyet=1 — gold=`-` | pred=`lojistik`
    - Anahtarı olmayan araba için acil çekici ihtiyacımız bulunuyor, yardım edebilecek birini bulabilir miyiz? Konum: Antakya - Ürgenpaşa Mahallesi @AFADBaskanlik @AFADTurkiye @ahbap #deprem #yardım #arabaçekici

### `psikolojik` (thr=0.86)

- TP=1, FP=0, FN=0, TN=193 — P=1.000, R=1.000, F1=1.000
- FN yok.
- FP yok.

### `bilgi_paylasimi` (thr=0.87)

- TP=16, FP=6, FN=9, TN=163 — P=0.727, R=0.640, F1=0.681
- **En kritik FN örnekleri** (en düşük prob):
  - `id=1624930670869880833` p=0.026 aciliyet=2 — gold=`barinma,bilgi_paylasimi` | pred=`barinma`
    - 🚨🚨🚨 ÖNEMLİ! Habertürk canlı yayından iki önemli duyuru: * İstanbul'a gelen tüm depremzedelere gitmek istedikleri yer için ücretsiz transfer: İRTİBAT: 0532 795 75 10 Kahramanmaraş Necip Fazıl Mah. Acil Çadır ihtiyacı. İRTİBAT: Betül Öğretmen 0546 811 22 46 #deprem https://t.co/3az
  - `id=1622438179130195969` p=0.027 aciliyet=2 — gold=`bilgi_paylasimi` | pred=`-`
    - atatürk mahallesi Atatürk caddesi no.94 samandağ hatay 🚨🚨 #DEPREMOLDU #deprem #acil #afad #hatay
  - `id=1622706650443382799` p=0.030 aciliyet=2 — gold=`bilgi_paylasimi` | pred=`-`
    - Adıyaman Gölbaşı Fatih mahallesi meriç caddesi acil destek lütfen @salihro @OguzhanUgur @haluklevent @ahbap @DepremDairesi #deprem #adiyamandeprem #adıyamanafad
  - `id=1622706574450982913` p=0.185 aciliyet=3 — gold=`arama_kurtarma,altyapi,bilgi_paylasimi` | pred=`arama_kurtarma,altyapi`
    - @gutitoli ELEKTRİK MAH. AKTAŞ SK. NO: 17 İÇ KAPI NO: 2 DEFNE / HATAY KADİR GÖZTEPE ARKADAŞIMIZDAN VE AİLESİNDEN HABER ALAMIYORUZ LÜTFEN YARDİMCİ OLUN İLETİŞİM BİLGİSİ 0541 494 0618 LÜTFEN GOREN DUYAN YAKİN OLAN YARDİMCİ OLSUN #Hatay #hataydepremi #ENKAZALTİNDAYİM #EnkazBilgisi #d
  - `id=1622415491233992708` p=0.418 aciliyet=3 — gold=`arama_kurtarma,bilgi_paylasimi` | pred=`arama_kurtarma`
    - ARKADAŞIMIN AİLESİNDEN HABER ALAMIYORUZ GÖÇÜK ALTINDA OLABİLİRLER!! Numune mahallesi ibrahim karaoğlan caddesi Güleryüz apartmanı E5 karayolu üzeri Hatay/İskenderun İLETİŞİM: 05331371908 #DEPREMOLDU #deprem
- **En kritik FP örnekleri** (en yüksek prob):
  - `id=1622479252552617984` p=0.974 aciliyet=1 — gold=`-` | pred=`bilgi_paylasimi`
    - Esenlik mahallesi hasan sokak mehdican apt. Antakya / HATAY “HOROZ AİLESİNDEN HABER ALINAMIYOR ORDA OLUP GÖREN VARSA BURAYA YANITLAYABİLİR Mİ ARKADAŞLAR ASKER ARKADAŞIMIN AİLESİ HABER ALAMIYORUZ” #hatay #deprem #AFAD #afad #ACİL #antakya #deprem #iskenderun
  - `id=1622880337230499841` p=0.967 aciliyet=3 — gold=`arama_kurtarma` | pred=`arama_kurtarma,bilgi_paylasimi`
    - Akevler mahallesi 424. sokak Melis apartmanı a blok kat 3 daire 7 Antakya/Hatay *Kaan Yılmaz ve ailesi, göçük altında olduğu düşünülüyor. Bilgi edinebilenler lütfen ulaşsın. #deprem #depremhatay @AFADBaskanlik @DepremDairesi @AFADHatay @sjw_erlik @OguzhanUgur @cerrahpasaIi https:
  - `id=1622443888425275394` p=0.964 aciliyet=1 — gold=`-` | pred=`bilgi_paylasimi`
    - Urgenpasa mahallesi 19. Sokak Hatay Antakya haber alamıyorum lütfen yardımcı olur musunuz bilgisi olan yazabilir mi #Hatay #deprem #ACİL
  - `id=1623565010927603712` p=0.943 aciliyet=0 — gold=`-` | pred=`bilgi_paylasimi`
    - Huzurevleri mahallesi 77201 sokak ihsan bayram sitesi Çukurova/Adana Yonca Baycal Ağahan Baycal Esila Baycal (6 yaşında) Azra Baycal (4 aylık) Bir haber olursa lütfen #deprem #ACILYARDIM #seferberlik @haluklevent @OguzhanUgur @Ahbap_Adana @AFAD_Adana
  - `id=1622708620742852622` p=0.912 aciliyet=1 — gold=`-` | pred=`bilgi_paylasimi`
    - @nediyorsunyasen @DepremDairesi @AFAD_gonullu ❗️ACİLLL❗️ BERAT UZUNESER, KAHRAMANMARAŞ TÜRKOĞLU ŞEKEROBA MAHALLESI saatlerdir haber almaya çalışıyoruz bu çevrede gören bilen olursa bana yazabilir mi? 05530164977 #deprem #acil #Kahramanmaras @DepremDairesi @AFAD_gonullu

## 4. Pattern analysis (örüntü yorumu)

- **`altyapi`: ince destekte F1 ~1.0** — `altyapi` testte yalnızca 3 pozitifle F1=1.000 veriyor (TP=3, FP=0, FN=0). Tek tahmin bu sayıyı oynatabilir; kalibre bir skor olarak değil, nitel sinyal olarak oku.
- **`psikolojik`: ince destekte F1 ~1.0** — `psikolojik` testte yalnızca 1 pozitifle F1=1.000 veriyor (TP=1, FP=0, FN=0). Tek tahmin bu sayıyı oynatabilir; kalibre bir skor olarak değil, nitel sinyal olarak oku.
- **Yüksek eşikler (≥0.85) — tutucu etiketler** — CV şu etiketlere çok yüksek eşik seçti: `barinma`=0.93, `lojistik`=0.91, `saglik`=0.90, `bilgi_paylasimi`=0.87, `psikolojik`=0.86. Bu etiketler yalnızca model çok eminken ateşliyor; bu precision'ı korur ama belirsiz pozitifleri kaçırır (FN riski).
- **Düşük eşikler (≤0.25) — agresif etiketler** — CV şu etiketlere çok düşük eşik seçti: `gida_su`=0.14. Bu etiketler zayıf sinyalde bile ateşliyor; recall yüksek, precision düşük (FP riski).
- **`lojistik` kaçırıldığında model sıkça `arama_kurtarma` diyor** — `lojistik` için 5/7 FN satırında `arama_kurtarma` predicted=1 — iki etiket arasında sınır bulanık.
- **`bilgi_paylasimi` kaçırıldığında model sıkça `arama_kurtarma` diyor** — `bilgi_paylasimi` için 4/9 FN satırında `arama_kurtarma` predicted=1 — iki etiket arasında sınır bulanık.
- **`arama_kurtarma` yanlış ateşlediğinde gold'da sıkça `lojistik` var** — `arama_kurtarma` için 2/4 FP satırında gerçek etiket `lojistik` idi — model `lojistik` benzeri dili `arama_kurtarma` olarak etiketliyor.
- **`lojistik` yanlış ateşlediğinde gold'da sıkça `arama_kurtarma` var** — `lojistik` için 5/6 FP satırında gerçek etiket `arama_kurtarma` idi — model `arama_kurtarma` benzeri dili `lojistik` olarak etiketliyor.

## 5. Confusion & co-occurrence

### FN cofire (etiket kaçırılırken hangi başka etiket ateşledi?)

- `arama_kurtarma`: (FN yok veya cofire yok)
- `saglik`: (FN yok veya cofire yok)
- `barinma`: (FN yok veya cofire yok)
- `gida_su`: `arama_kurtarma`×1, `lojistik`×1
- `altyapi`: (FN yok veya cofire yok)
- `guvenlik`: `arama_kurtarma`×1, `barinma`×1, `gida_su`×1
- `lojistik`: `arama_kurtarma`×5
- `psikolojik`: (FN yok veya cofire yok)
- `bilgi_paylasimi`: `arama_kurtarma`×4, `altyapi`×2, `barinma`×1, `guvenlik`×1

### FP cofire (yanlış ateş eden etiketin satırında gerçekten hangi etiketler vardı?)

- `arama_kurtarma`: `lojistik`×2, `altyapi`×1
- `saglik`: (FP yok veya cofire yok)
- `barinma`: (FP yok veya cofire yok)
- `gida_su`: (FP yok veya cofire yok)
- `altyapi`: (FP yok veya cofire yok)
- `guvenlik`: `arama_kurtarma`×1, `bilgi_paylasimi`×1
- `lojistik`: `arama_kurtarma`×5, `saglik`×1
- `psikolojik`: (FP yok veya cofire yok)
- `bilgi_paylasimi`: `arama_kurtarma`×1

### Gold-level co-occurrence (en sık birlikte gelen etiket çiftleri)

| a | b | count |
|---|---|---|
| `arama_kurtarma` | `lojistik` | 25 |
| `arama_kurtarma` | `bilgi_paylasimi` | 9 |
| `arama_kurtarma` | `saglik` | 6 |
| `barinma` | `gida_su` | 4 |
| `arama_kurtarma` | `gida_su` | 3 |
| `gida_su` | `lojistik` | 3 |
| `altyapi` | `bilgi_paylasimi` | 2 |
| `arama_kurtarma` | `barinma` | 1 |
| `arama_kurtarma` | `altyapi` | 1 |
| `arama_kurtarma` | `guvenlik` | 1 |

### Multi-label: her iki etiket gold'da var iken modelin tahmini

| a | b | gold_ikisi | pred_ikisi | sadece_a | sadece_b | hiçbiri |
|---|---|---|---|---|---|---|
| `arama_kurtarma` | `lojistik` | 25 | 20 | 4 | 0 | 1 |
| `arama_kurtarma` | `bilgi_paylasimi` | 9 | 5 | 4 | 0 | 0 |
| `altyapi` | `bilgi_paylasimi` | 2 | 0 | 2 | 0 | 0 |
| `arama_kurtarma` | `gida_su` | 3 | 2 | 1 | 0 | 0 |
| `arama_kurtarma` | `guvenlik` | 1 | 0 | 1 | 0 | 0 |
| `barinma` | `guvenlik` | 1 | 0 | 1 | 0 | 0 |
| `barinma` | `bilgi_paylasimi` | 1 | 0 | 1 | 0 | 0 |
| `gida_su` | `guvenlik` | 1 | 0 | 1 | 0 | 0 |
| `gida_su` | `lojistik` | 3 | 2 | 0 | 1 | 0 |
| `arama_kurtarma` | `saglik` | 6 | 6 | 0 | 0 | 0 |

## 6. Slice analysis

### Aciliyet (0-3)

| group | rows | with_any_error | error_rate |
|---|---|---|---|
| 0 | 7 | 2 | 0.286 |
| 2 | 19 | 5 | 0.263 |
| 1 | 56 | 12 | 0.214 |
| 3 | 112 | 19 | 0.170 |

### Metin uzunluğu

| group | rows | with_any_error | error_rate |
|---|---|---|---|
| medium(60-139) | 15 | 3 | 0.200 |
| long(>=140) | 179 | 35 | 0.196 |

### Gold etiket sayısı

| group | rows | with_any_error | error_rate |
|---|---|---|---|
| 3+ | 4 | 3 | 0.750 |
| 2 | 45 | 12 | 0.267 |
| 0 | 40 | 8 | 0.200 |
| 1 | 105 | 15 | 0.143 |

### Metin türü (heuristic)

| group | rows | with_any_error | error_rate |
|---|---|---|---|
| other | 8 | 4 | 0.500 |
| call_for_help | 185 | 34 | 0.184 |
| info_request | 1 | 0 | 0.000 |

## 7. Leak-free caveat

- Step 7 v3'te id-level silver→gold leakage kapatıldı (canonical gold'un 1934 id'sinin tamamı silver havuzundan çıkarıldı).
- **Ancak content-level risk hâlâ tamamen elenmiş değildir.** Silver kaynağı (`data/processed/emergency_geolocated_96k.csv`), gold ile aynı tweet havuzundan türedi; gold id'leri çıkarılsa bile, retweet / alıntı / near-duplicate metinler silver'da kalmış olabilir.
- Bu nedenle özellikle F1=1.0 saturasyonları (ör. `altyapi`, `psikolojik`) gerçek genelleme değil, **dar desen ezberi** sonucu da olabilir. Başarı örneklerini yorumlarken bu sınırı unutmayın.

## 8. Son bölüm

### Modelin en çok zorlandığı 5 örüntü

1. `bilgi_paylasimi` kaçırıldığında modelin en sık önerdiği etiket `arama_kurtarma` — 4/9 FN satırında `arama_kurtarma` ateşledi; bu iki etiket arasında sınır bulanık.
2. `bilgi_paylasimi` için CV eşiği çok yüksek (thr=0.87); recall=0.64 ile FN=9 — threshold kalibrasyonu tutucu, eşik düşürmek FN'i azaltabilir ama FP riski var.
3. `lojistik` kaçırıldığında modelin en sık önerdiği etiket `arama_kurtarma` — 5/7 FN satırında `arama_kurtarma` ateşledi; bu iki etiket arasında sınır bulanık.
4. `lojistik` yanlış ateşlediğinde gold'da en sık `arama_kurtarma` bulunuyor — 5/6 FP satırında `arama_kurtarma` gerçek pozitifti; model `arama_kurtarma`-benzeri dili `lojistik` olarak etiketliyor.
5. `arama_kurtarma` yanlış ateşlediğinde gold'da en sık `lojistik` bulunuyor — 2/4 FP satırında `lojistik` gerçek pozitifti; model `lojistik`-benzeri dili `arama_kurtarma` olarak etiketliyor.

### Step 9 için öneriler (bu adımda uygulanmaz)

- Step 9 (seçim/çıkarım) öncesinde, `bilgi_paylasimi` ve `guvenlik` için eşik duyarlılığını sensitivity-plot ile dökümante et (CV thresholds sabit kalsa da production'da neyi feda ettiğimizi bilelim).
- `bilgi_paylasimi`: eşik 0.87 ile recall=0.64 — step 9'da ayrı bir `threshold_production.json` üretip recall-öncelikli senaryoda daha düşük eşik dene (ama bu scoreboard'u değiştirmez).
- Rare etiketler (`altyapi`, `psikolojik`) için step 9 raporuna confidence interval ekleyerek tek-tahmin saturasyonunu açıkça belirt.
- Step 7 leak-free kapsamı id-seviyesiydi; step 9'dan önce silver ↔ gold **içerik** örtüşmesini (char n-gram veya normalize metin hash) ölç ve raporla; near-duplicate kalıntı varsa dokümante et.
- `bilgi_paylasimi` F1 exp3'te exp1'in altında kaldı — step 9'da etiket tanımı review et (arama çağrısı vs bilgi çağrısı sınırı) ve gerekirse annotation guideline'a küçük bir not düş.
- FN örneklerini gözle inceleyip potansiyel annotation hatalarını işaretle; tespit edilenleri bir `gold_review_candidates.csv` olarak saklayıp step 10/11 için bekleme listesine koy.

> Bu adımda model seçimi / tahmin üretimi / `models/final/selection.json` güncellemesi yapılmadı. Step 9 kararları için bu rapor girdi; aksiyonu onay sonrası.
