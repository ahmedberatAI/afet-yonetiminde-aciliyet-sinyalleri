# Web app / API integration — step 13 status

**Status:** `not_applicable`
**Tarih:** 2026-04-21
**Karar verici:** step 10–14 one-parcel kapanış

## Kanıt

Bu repoda bir web app, API servis katmanı veya tamamlanmış dashboard **bulunmadı**.
Aşağıdaki tarama yapıldı:

1. Repo root altında `app/`, `api/`, `web/`, `frontend/`, `dashboard/` dizinleri **yok**.
2. `scripts/` altında `app*.py`, `api*.py`, `serve*.py`, `endpoint*.py`, `dashboard*.py`
   **yok**.
3. Tüm Python dosyaları (recursive) içinde
   `streamlit`, `fastapi`, `flask`, `dash`, `gradio`, `uvicorn`
   import'u arandı — **hiçbir eşleşme yok**.
4. `requirements_dashboard.txt` mevcut:
   ```
   streamlit
   pandas
   numpy
   ```
   Bu sadece bir *niyet sinyali*; buna karşılık gelen bir `*.py` consumer kodu repoda yok.
   Root README'de de bu açıkça belirtilmişti: "bu repoda tamamlanmış bir dashboard
   uygulama kodu bulunmamaktadır."

## Karar

Step 13 bu release pass'inde `not_applicable`. Sahte bir adapter / mock FastAPI server
yazıp "tamamlandı" işaretlemek iki sebepten doğru değil:

1. **Kontrat yok.** Hangi endpoint, hangi payload, hangi latency bütçesi, hangi auth?
   Hiçbiri mevcut değil. Tahmin edip uydurmak yanlış karara zemin olur.
2. **Tüketici yok.** Ne dashboard, ne notebook, ne dış servis bu çıktıyı tüketiyor.
   Adapter kimsenin kullanmayacağı sürüklenen bir artifact olur.

## Yine de tüketiciye destek

Gerçek bir dashboard veya API için başlangıç noktası olarak:

- **Shape ve schema**: `data/predictions/need_predictions_geolocated_v2_final.meta.json`
  (schema_note + prediction_columns + label_to_prob_column tam adaptasyon bilgisi içerir).
- **Programmatic adapter**: `release/need_classifier_v2/examples/adapter_snippet.py`
  (HuggingFace tokenizer + model, per-label threshold uygulamalı minimal örnek).
- **Usage doc**: `release/need_classifier_v2/docs/usage.md` (canonical predict komutu +
  schema tablosu).

Gerçek bir dashboard implement edilmek istenirse minimum gereksinimler:

1. Girdi CSV schema: en azından `id`, `tweet_clean` (tahmin için); diğer kolonlar passthrough.
2. Çıktı: `prob_<label>`, `pred_<label>` (thresholds_cv.json ile); tüketici kendi eşiğini
   uygulayacaksa sadece `prob_*` döndürmek yeterli.
3. Threshold kaynağı: `release/need_classifier_v2/thresholds/thresholds_cv.json` (repo içi
   kanonik pointer).
4. Model yükleme: `release/need_classifier_v2/model/README.md` içindeki pointer'a göre
   safetensors bir kereye yüklenmeli (lazy singleton).

Bu gereksinimler canonical release içinde belgelendi; yeni bir tüketici eklendiğinde
bu doküman güncellenmeli.
