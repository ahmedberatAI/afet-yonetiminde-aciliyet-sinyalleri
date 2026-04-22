# Content-overlap audit (v2, leak-free silver vs gold splits)

Bu audit step-9 selection rationale'ı destekler. Step 7 v3 id-seviyesindeki silver→gold leakage'ı kapattı (1934/1934 gold id silver havuzundan çıkarıldı). Burada id ayıklamasından sonra **içerik** (cleaned-text) seviyesinde kalıntı örtüşme olup olmadığını ölçüyoruz.

## Kaynaklar

- Silver (leak-free): `C:/Users/omen/Desktop/afetYonetimi_colab/data/labeling/need_classification_silver_63k_profileA_exgold.csv` — 61246 satır
- Gold splits:
  - `C:/Users/omen/Desktop/afetYonetimi_colab/data/modeling/need_classification_gold_combined/train.csv` — 1547 satır
  - `C:/Users/omen/Desktop/afetYonetimi_colab/data/modeling/need_classification_gold_combined/val.csv` — 193 satır
  - `C:/Users/omen/Desktop/afetYonetimi_colab/data/modeling/need_classification_gold_combined/test.csv` — 194 satır
- Normalizasyon: lowercase + URL/mention/`#` ayıklama + ASCII-dışı dengeleme + whitespace squash.
- Near-dup metriği: 5-char shingle Jaccard ≥ 0.85

## Özet tablo

| split | gold satır | exact overlap | near-dup (J≥thr) | ratio_exact | ratio_near |
|---|---|---|---|---|---|
| train | 1547 | 330 | 499 | 0.2133 | 0.3226 |
| val | 193 | 42 | 63 | 0.2176 | 0.3264 |
| test | 194 | 35 | 59 | 0.1804 | 0.3041 |

## train — örnekler

### Exact overlap (ilk 10)

- `id=1623786537526632451` — kahramanmaraş onikişubat şazibey mahallesi elifbike apartmanı için acil termal kamera gerekmektedir destek bekliyoruz deprem
- `id=1623432995490365441` — şu an ses geliyormuş enkaz altından adres besni betaş 15 temmuz mahallesi petrol arkası i letişim 0501 352 62 50 çağla saha alanında kendisi şarjı var halukleve
- `id=1622670035977216007` — arkadaşımız ömer yiğit e hala ulaşamadık bir haber alamıyoruz emek mahallesi 10 sok no 31 antakya hatay emin olmamakla birlikte hatay enkazaltındayım enkazalti 
- `id=1623423283210207232` — deprem adiyamandeprem arkadaşımn anne ve babası şu adreste göçük altında ve maalesef saatlerdir haber alınamıyor ve 3 gundur ne vinc geldi ne enkaz kaldirma eki
- `id=1623408013817020418` — enkaz altında olanlar cansu 6 aylık hamile ve mustafa kavlak 0531 861 21 91 ekinci mah i nönü blv no 124 elit apt kat 5 d 25 antakya hatay dışarıda bekleyen ert
- `id=1623458431670661125` — cebrail mahallesi 1 bahceli sokak aknur apartmanı d blok antakya adres burası içinde canlı insan lar var antakyahatay depremsondakika deprem
- `id=1622457173191041024` — şazibey mahallesi stad caddesi no 6 iç kapı no 28 onikişubat kahramanmaraş adresinde arkadaşımız ve ailesi enkaz altında kalmıştır arama kurtarma ekibi gerekiyo
- `id=1623494681064513536` — kahramanmaraş afşi n i lçesi aritaş mahallesi nde aci l olarak çok sayida isitici çadir bebek mamasi ve bezi i hti yaci bulunmakta çocuklar yaşlilar peri şan ol
- `id=1622867041198780420` — lütfen destek olalım gaziantep i slahiye ilçesinde cumhuriyet mahallesi şehit i brahim aslan sokak kat 1 5 yaşındaki çocuk annesi göçük altında acil yardım bekl
- `id=1623408743361675267` — allah aşkina bi ri artik sila yi kurtarsin ses geli yor hatay cebrail mahallesi 1 bahçeli sokak aknur apartmanı d blok hatay antakya sila yüzgüleç sesi geliyor 

### Near-dup (ilk 10, J≥0.85)

- J=1.000 `gold_id=1623786537526632451` ↔ `silver_id=1623789260229820421`
  - gold : kahramanmaraş onikişubat şazibey mahallesi elifbike apartmanı için acil termal kamera gerekmektedir destek bekliyoruz deprem
  - silver: kahramanmaraş onikişubat şazibey mahallesi elifbike apartmanı için acil termal kamera gerekmektedir destek bekliyoruz deprem
- J=1.000 `gold_id=1623432995490365441` ↔ `silver_id=1623433692122324992`
  - gold : şu an ses geliyormuş enkaz altından adres besni betaş 15 temmuz mahallesi petrol arkası i letişim 0501 352 62 50 çağla saha alanında kendisi şarjı var halukleve
  - silver: şu an ses geliyormuş enkaz altından adres besni betaş 15 temmuz mahallesi petrol arkası i letişim 0501 352 62 50 çağla saha alanında kendisi şarjı var halukleve
- J=1.000 `gold_id=1622670035977216007` ↔ `silver_id=1622670169284780037`
  - gold : arkadaşımız ömer yiğit e hala ulaşamadık bir haber alamıyoruz emek mahallesi 10 sok no 31 antakya hatay emin olmamakla birlikte hatay enkazaltındayım enkazalti 
  - silver: arkadaşımız ömer yiğit e hala ulaşamadık bir haber alamıyoruz emek mahallesi 10 sok no 31 antakya hatay emin olmamakla birlikte hatay enkazaltındayım enkazalti 
- J=1.000 `gold_id=1623423283210207232` ↔ `silver_id=1623423183947800578`
  - gold : deprem adiyamandeprem arkadaşımn anne ve babası şu adreste göçük altında ve maalesef saatlerdir haber alınamıyor ve 3 gundur ne vinc geldi ne enkaz kaldirma eki
  - silver: deprem adiyamandeprem arkadaşımn anne ve babası şu adreste göçük altında ve maalesef saatlerdir haber alınamıyor ve 3 gundur ne vinc geldi ne enkaz kaldirma eki
- J=1.000 `gold_id=1623408013817020418` ↔ `silver_id=1623405614192787458`
  - gold : enkaz altında olanlar cansu 6 aylık hamile ve mustafa kavlak 0531 861 21 91 ekinci mah i nönü blv no 124 elit apt kat 5 d 25 antakya hatay dışarıda bekleyen ert
  - silver: enkaz altında olanlar cansu 6 aylık hamile ve mustafa kavlak 0531 861 21 91 ekinci mah i nönü blv no 124 elit apt kat 5 d 25 antakya hatay dışarıda bekleyen ert
- J=1.000 `gold_id=1623458431670661125` ↔ `silver_id=1623460215885647873`
  - gold : cebrail mahallesi 1 bahceli sokak aknur apartmanı d blok antakya adres burası içinde canlı insan lar var antakyahatay depremsondakika deprem
  - silver: cebrail mahallesi 1 bahceli sokak aknur apartmanı d blok antakya adres burası içinde canlı insan lar var antakyahatay depremsondakika deprem
- J=1.000 `gold_id=1622457173191041024` ↔ `silver_id=1622454690368655360`
  - gold : şazibey mahallesi stad caddesi no 6 iç kapı no 28 onikişubat kahramanmaraş adresinde arkadaşımız ve ailesi enkaz altında kalmıştır arama kurtarma ekibi gerekiyo
  - silver: şazibey mahallesi stad caddesi no 6 iç kapı no 28 onikişubat kahramanmaraş adresinde arkadaşımız ve ailesi enkaz altında kalmıştır arama kurtarma ekibi gerekiyo
- J=1.000 `gold_id=1623494681064513536` ↔ `silver_id=1623556992509054980`
  - gold : kahramanmaraş afşi n i lçesi aritaş mahallesi nde aci l olarak çok sayida isitici çadir bebek mamasi ve bezi i hti yaci bulunmakta çocuklar yaşlilar peri şan ol
  - silver: kahramanmaraş afşi n i lçesi aritaş mahallesi nde aci l olarak çok sayida isitici çadir bebek mamasi ve bezi i hti yaci bulunmakta çocuklar yaşlilar peri şan ol
- J=1.000 `gold_id=1622867041198780420` ↔ `silver_id=1622704703963275264`
  - gold : lütfen destek olalım gaziantep i slahiye ilçesinde cumhuriyet mahallesi şehit i brahim aslan sokak kat 1 5 yaşındaki çocuk annesi göçük altında acil yardım bekl
  - silver: lütfen destek olalım gaziantep i slahiye ilçesinde cumhuriyet mahallesi şehit i brahim aslan sokak kat 1 5 yaşındaki çocuk annesi göçük altında acil yardım bekl
- J=1.000 `gold_id=1623408743361675267` ↔ `silver_id=1623405956506812416`
  - gold : allah aşkina bi ri artik sila yi kurtarsin ses geli yor hatay cebrail mahallesi 1 bahçeli sokak aknur apartmanı d blok hatay antakya sila yüzgüleç sesi geliyor 
  - silver: allah aşkina bi ri artik sila yi kurtarsin ses geli yor hatay cebrail mahallesi 1 bahçeli sokak aknur apartmanı d blok hatay antakya sila yüzgüleç sesi geliyor 

## val — örnekler

### Exact overlap (ilk 10)

- `id=1622463117669056512` — aci llll hatay iskenderun deprem aci l sondakika i skenderun hatay numune mahallesi i brahim karaoglanoglu caddesi demirkent ticaret üstü apartman yıkılmış aile
- `id=1622701698354479114` — şan ailesinden deprem olduğundan beri haber alamıyoruz hiçbir acil kurtarma ekibi enkaz bölgesine intikal etmemiş karamanlı mahallesi doktor sadullah caddesi çu
- `id=1622471374194237442` — i brahim karaoğlanoğlu caddesi 63a 31230 mustafa kemal mahallesi i skenderun hatay müzeyyen ezgi 05537431881 göçük altında olan bu arkadaşımızın acil yardıma ih
- `id=1622430644021116929` — yardi m edi n deprem i çeride insanlar var dışarı çıkamıyorlar yardım edin adres yavuz aslan apartmanı a blok kat 7 no 23 akevker mahallesi antakya hatay 432 so
- `id=1623628789661024257` — profesyonel ekipler çalışıyor fakat aci l vi nç lazim ve gaz maskesi ihtiyacı var lütfen yardımcı olun güncel bilgi 9 şubat 8 00 76 saat oldu merve haydar mursa
- `id=1622485523733282818` — enkaz altinda aci l deprem depremhatay artik bittim buraya kadarmis digerlerimi kurtarin sesler geliyor mustafa kemal mah 547 sokak a22 apartman atatürk düşünce
- `id=1622653070705401866` — dumlupınar mahallesi sait zarifoğlu bulvarı no 89 oni ki şubat kahramanmaraş fatma vişne göçük altında lütfen yardım edin deprem kahramanmaras
- `id=1623674460501954561` — vi nç lazim aci l vi nç hayrullah mahallesi mali k ejder caddesi no 36 a blok oni ki şubat kahramanmaraş deprem vinç yardım kahramanmaraş
- `id=1624498805335814145` — hatay altınözü köyü mayadalı mahallesine acil çadır gerekiyor lütfen yayalım hatayyardimbekliyor deprem aci l cadirihtiyacı çadır depremyardim
- `id=1622909982814482432` — lütfen paylaşalim berk emre ekenek çocuk ömer ekenek bebek necdet ekenek bedia ekenek i brahim ekenek akevler mahallesi 432 sokak torunlar apartmanı antakya hat

### Near-dup (ilk 10, J≥0.85)

- J=1.000 `gold_id=1622463117669056512` ↔ `silver_id=1622462551488331776`
  - gold : aci llll hatay iskenderun deprem aci l sondakika i skenderun hatay numune mahallesi i brahim karaoglanoglu caddesi demirkent ticaret üstü apartman yıkılmış aile
  - silver: aci llll hatay iskenderun deprem aci l sondakika i skenderun hatay numune mahallesi i brahim karaoglanoglu caddesi demirkent ticaret üstü apartman yıkılmış aile
- J=1.000 `gold_id=1622701698354479114` ↔ `silver_id=1622732847705387009`
  - gold : şan ailesinden deprem olduğundan beri haber alamıyoruz hiçbir acil kurtarma ekibi enkaz bölgesine intikal etmemiş karamanlı mahallesi doktor sadullah caddesi çu
  - silver: şan ailesinden deprem olduğundan beri haber alamıyoruz hiçbir acil kurtarma ekibi enkaz bölgesine intikal etmemiş karamanlı mahallesi doktor sadullah caddesi çu
- J=1.000 `gold_id=1622471374194237442` ↔ `silver_id=1622470958538805249`
  - gold : i brahim karaoğlanoğlu caddesi 63a 31230 mustafa kemal mahallesi i skenderun hatay müzeyyen ezgi 05537431881 göçük altında olan bu arkadaşımızın acil yardıma ih
  - silver: i brahim karaoğlanoğlu caddesi 63a 31230 mustafa kemal mahallesi i skenderun hatay müzeyyen ezgi 05537431881 göçük altında olan bu arkadaşımızın acil yardıma ih
- J=1.000 `gold_id=1622430644021116929` ↔ `silver_id=1622430185193701376`
  - gold : yardi m edi n deprem i çeride insanlar var dışarı çıkamıyorlar yardım edin adres yavuz aslan apartmanı a blok kat 7 no 23 akevker mahallesi antakya hatay 432 so
  - silver: yardi m edi n deprem i çeride insanlar var dışarı çıkamıyorlar yardım edin adres yavuz aslan apartmanı a blok kat 7 no 23 akevker mahallesi antakya hatay 432 so
- J=1.000 `gold_id=1623628789661024257` ↔ `silver_id=1623628410588217346`
  - gold : profesyonel ekipler çalışıyor fakat aci l vi nç lazim ve gaz maskesi ihtiyacı var lütfen yardımcı olun güncel bilgi 9 şubat 8 00 76 saat oldu merve haydar mursa
  - silver: profesyonel ekipler çalışıyor fakat aci l vi nç lazim ve gaz maskesi ihtiyacı var lütfen yardımcı olun güncel bilgi 9 şubat 8 00 76 saat oldu merve haydar mursa
- J=1.000 `gold_id=1622485523733282818` ↔ `silver_id=1622485901661069312`
  - gold : enkaz altinda aci l deprem depremhatay artik bittim buraya kadarmis digerlerimi kurtarin sesler geliyor mustafa kemal mah 547 sokak a22 apartman atatürk düşünce
  - silver: enkaz altinda aci l deprem depremhatay artik bittim buraya kadarmis digerlerimi kurtarin sesler geliyor mustafa kemal mah 547 sokak a22 apartman atatürk düşünce
- J=1.000 `gold_id=1622653070705401866` ↔ `silver_id=1622651026149867544`
  - gold : dumlupınar mahallesi sait zarifoğlu bulvarı no 89 oni ki şubat kahramanmaraş fatma vişne göçük altında lütfen yardım edin deprem kahramanmaras
  - silver: dumlupınar mahallesi sait zarifoğlu bulvarı no 89 oni ki şubat kahramanmaraş fatma vişne göçük altında lütfen yardım edin deprem kahramanmaras
- J=1.000 `gold_id=1623674460501954561` ↔ `silver_id=1623672643483234310`
  - gold : vi nç lazim aci l vi nç hayrullah mahallesi mali k ejder caddesi no 36 a blok oni ki şubat kahramanmaraş deprem vinç yardım kahramanmaraş
  - silver: vi nç lazim aci l vi nç hayrullah mahallesi mali k ejder caddesi no 36 a blok oni ki şubat kahramanmaraş deprem vinç yardım kahramanmaraş
- J=1.000 `gold_id=1624498805335814145` ↔ `silver_id=1624501536817569796`
  - gold : hatay altınözü köyü mayadalı mahallesine acil çadır gerekiyor lütfen yayalım hatayyardimbekliyor deprem aci l cadirihtiyacı çadır depremyardim
  - silver: hatay altınözü köyü mayadalı mahallesine acil çadır gerekiyor lütfen yayalım hatayyardimbekliyor deprem aci l cadirihtiyacı çadır depremyardim
- J=1.000 `gold_id=1622909982814482432` ↔ `silver_id=1622909123854495745`
  - gold : lütfen paylaşalim berk emre ekenek çocuk ömer ekenek bebek necdet ekenek bedia ekenek i brahim ekenek akevler mahallesi 432 sokak torunlar apartmanı antakya hat
  - silver: lütfen paylaşalim berk emre ekenek çocuk ömer ekenek bebek necdet ekenek bedia ekenek i brahim ekenek akevler mahallesi 432 sokak torunlar apartmanı antakya hat

## test — örnekler

### Exact overlap (ilk 10)

- `id=1622660204335706112` — arkadasimizin ailesine daha yardım gitmemiş göçük altindalar şazibey mah stad caddesi elif bike apartmanı kahramanmaraş merkez 4 kişiler necati kurt leyla kurt 
- `id=1622465320878329857` — hatay deprem afad depremde yıkılan binada akrabalarım mahsur kaldıı lütfen acil yardım yollayın hatay antakya adnan menderes caddesi cumhuriyet mahallesi tuba a
- `id=1622679827395346432` — adıyaman merkez yeni sanayi mahallesi 2815 sokak barbaros sitesi 4 bina tamamen çökmüş durumda enkazın altında olanlar için acil yetkililerin ulaşması gerek dep
- `id=1623361162610937876` — dervişpaşa mah molla hasan cad kardelen apt i slahiye gaziantep 27 yaşındaki fatih selçuk ve eşi fatmanur akşan selçuk tan deprem saatinden beri haber alamıyoru
- `id=1622487399988084738` — 5 yardim çağrisi sarayken mahallesi yeni akkent royal sitesi b blok 10 kat antakya hatay çocuklarla göçük altındayım 6 yardim çağrisi adiyaman gölbaşi özteki n 
- `id=1623393193403326464` — kahramanmaraş dulkadi roğlu trabzon bulvari menderes mahallesi ağaoğlu si tesi b blok no 127 arkadasim furkan celik e hala ulasamiyoruz calismalar durdurulmuş p
- `id=1622880337230499841` — akevler mahallesi 424 sokak melis apartmanı a blok kat 3 daire 7 antakya hatay kaan yılmaz ve ailesi göçük altında olduğu düşünülüyor bilgi edinebilenler lütfen
- `id=1623302770928353282` — enkazdalar ses geliyor burcu zubari anil zubari yardim bekliyorlar adres cumhuriyet mahallesi ingiliz okulu caddesi mithat kuk apartmani shell yani hatay samand
- `id=1622458917056917510` — acil deprem müzeler bolgesi beko mağazasi üstü küçük dalyan mahallesi hatay şu an enkaz altında yaşıyorlar az önce irtibata geçildi ferhat biner gülsüm biner ve
- `id=1622890376766840835` — deprem esenevler mah 15536 sok palmiye apartmanı no 1 6 merkez osmaniye yusuf konur depremden beri ulaşamıyorum yardım edin acil

### Near-dup (ilk 10, J≥0.85)

- J=1.000 `gold_id=1622660204335706112` ↔ `silver_id=1622660869942386688`
  - gold : arkadasimizin ailesine daha yardım gitmemiş göçük altindalar şazibey mah stad caddesi elif bike apartmanı kahramanmaraş merkez 4 kişiler necati kurt leyla kurt 
  - silver: arkadasimizin ailesine daha yardım gitmemiş göçük altindalar şazibey mah stad caddesi elif bike apartmanı kahramanmaraş merkez 4 kişiler necati kurt leyla kurt 
- J=1.000 `gold_id=1622465320878329857` ↔ `silver_id=1622458062362619905`
  - gold : hatay deprem afad depremde yıkılan binada akrabalarım mahsur kaldıı lütfen acil yardım yollayın hatay antakya adnan menderes caddesi cumhuriyet mahallesi tuba a
  - silver: hatay deprem afad depremde yıkılan binada akrabalarım mahsur kaldıı lütfen acil yardım yollayın hatay antakya adnan menderes caddesi cumhuriyet mahallesi tuba a
- J=1.000 `gold_id=1622679827395346432` ↔ `silver_id=1622680306284212224`
  - gold : adıyaman merkez yeni sanayi mahallesi 2815 sokak barbaros sitesi 4 bina tamamen çökmüş durumda enkazın altında olanlar için acil yetkililerin ulaşması gerek dep
  - silver: adıyaman merkez yeni sanayi mahallesi 2815 sokak barbaros sitesi 4 bina tamamen çökmüş durumda enkazın altında olanlar için acil yetkililerin ulaşması gerek dep
- J=1.000 `gold_id=1623361162610937876` ↔ `silver_id=1623357123869519874`
  - gold : dervişpaşa mah molla hasan cad kardelen apt i slahiye gaziantep 27 yaşındaki fatih selçuk ve eşi fatmanur akşan selçuk tan deprem saatinden beri haber alamıyoru
  - silver: dervişpaşa mah molla hasan cad kardelen apt i slahiye gaziantep 27 yaşındaki fatih selçuk ve eşi fatmanur akşan selçuk tan deprem saatinden beri haber alamıyoru
- J=1.000 `gold_id=1622487399988084738` ↔ `silver_id=1622469283631493120`
  - gold : 5 yardim çağrisi sarayken mahallesi yeni akkent royal sitesi b blok 10 kat antakya hatay çocuklarla göçük altındayım 6 yardim çağrisi adiyaman gölbaşi özteki n 
  - silver: 5 yardim çağrisi sarayken mahallesi yeni akkent royal sitesi b blok 10 kat antakya hatay çocuklarla göçük altındayım 6 yardim çağrisi adiyaman gölbaşi özteki n 
- J=1.000 `gold_id=1623393193403326464` ↔ `silver_id=1623380280315678721`
  - gold : kahramanmaraş dulkadi roğlu trabzon bulvari menderes mahallesi ağaoğlu si tesi b blok no 127 arkadasim furkan celik e hala ulasamiyoruz calismalar durdurulmuş p
  - silver: kahramanmaraş dulkadi roğlu trabzon bulvari menderes mahallesi ağaoğlu si tesi b blok no 127 arkadasim furkan celik e hala ulasamiyoruz calismalar durdurulmuş p
- J=1.000 `gold_id=1622880337230499841` ↔ `silver_id=1622881458367889408`
  - gold : akevler mahallesi 424 sokak melis apartmanı a blok kat 3 daire 7 antakya hatay kaan yılmaz ve ailesi göçük altında olduğu düşünülüyor bilgi edinebilenler lütfen
  - silver: akevler mahallesi 424 sokak melis apartmanı a blok kat 3 daire 7 antakya hatay kaan yılmaz ve ailesi göçük altında olduğu düşünülüyor bilgi edinebilenler lütfen
- J=1.000 `gold_id=1623302770928353282` ↔ `silver_id=1623302860044701697`
  - gold : enkazdalar ses geliyor burcu zubari anil zubari yardim bekliyorlar adres cumhuriyet mahallesi ingiliz okulu caddesi mithat kuk apartmani shell yani hatay samand
  - silver: enkazdalar ses geliyor burcu zubari anil zubari yardim bekliyorlar adres cumhuriyet mahallesi ingiliz okulu caddesi mithat kuk apartmani shell yani hatay samand
- J=1.000 `gold_id=1622458917056917510` ↔ `silver_id=1622455784108867584`
  - gold : acil deprem müzeler bolgesi beko mağazasi üstü küçük dalyan mahallesi hatay şu an enkaz altında yaşıyorlar az önce irtibata geçildi ferhat biner gülsüm biner ve
  - silver: acil deprem müzeler bolgesi beko mağazasi üstü küçük dalyan mahallesi hatay şu an enkaz altında yaşıyorlar az önce irtibata geçildi ferhat biner gülsüm biner ve
- J=1.000 `gold_id=1622890376766840835` ↔ `silver_id=1622890861070544897`
  - gold : deprem esenevler mah 15536 sok palmiye apartmanı no 1 6 merkez osmaniye yusuf konur depremden beri ulaşamıyorum yardım edin acil
  - silver: deprem esenevler mah 15536 sok palmiye apartmanı no 1 6 merkez osmaniye yusuf konur depremden beri ulaşamıyorum yardım edin acil

## Yorum

UYARI: id-ayıklamasına rağmen 407/1934 gold satır silver'da normalize-eşdeğer metinle kaldı (muhtemelen aynı tweet'in farklı id'si). Bu sinyal selection'ı DURDURACAK düzeyde değilse (örn. yalnızca gold_train'i etkiliyorsa) 'known residual risk' olarak doküman et ve step 10'da kapsamlı bir dedup yap.
