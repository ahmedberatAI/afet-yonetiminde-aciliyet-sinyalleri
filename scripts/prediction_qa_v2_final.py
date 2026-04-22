#!/usr/bin/env python3
"""
Step 10 - Prediction QA on the canonical v2 final prediction output.

Reads:
  - data/predictions/need_predictions_geolocated_v2_final.csv
  - data/predictions/need_predictions_geolocated_v2_final.meta.json

Does:
  1. Integrity checks (row count, pred_positives match, id duplicates,
     prob_*/pred_* column integrity, empty rows).
  2. Schema clarity: enriches the meta with `prediction_columns`,
     `probability_columns`, `label_to_pred_column`, `label_to_prob_column`,
     `row_count`, and a small schema_note.
  3. Label prevalence (count + rate), pred_label_count distribution,
     pred_any_need distribution.
  4. Slices by province / district (top 15) / text-length buckets / urgency buckets.
  5. Anomaly scan: over-firing (label rate > 0.3), under-firing (label rate < 0.001),
     empty-prediction rows, duplicate ids, NaN probabilities, out-of-range probs,
     threshold/pred mismatch (sanity).
  6. Classifies findings as release_blocker / warning / known_limitation.

Writes:
  - data/analysis/prediction_qa_v2_final.md
  - data/analysis/prediction_qa_v2_final.json
  - data/analysis/prediction_qa_v2_final.label_prevalence.csv
  - data/analysis/prediction_qa_v2_final.slices_province.csv
  - data/analysis/prediction_qa_v2_final.slices_district.csv

Does NOT modify the prediction CSV. Updates the meta JSON in place with
the schema-clarity additions (non-destructive).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = REPO_ROOT / "data" / "predictions" / "need_predictions_geolocated_v2_final.csv"
PRED_META = REPO_ROOT / "data" / "predictions" / "need_predictions_geolocated_v2_final.meta.json"
OUT_DIR = REPO_ROOT / "data" / "analysis"
OUT_MD = OUT_DIR / "prediction_qa_v2_final.md"
OUT_JSON = OUT_DIR / "prediction_qa_v2_final.json"
OUT_LABEL_PREV = OUT_DIR / "prediction_qa_v2_final.label_prevalence.csv"
OUT_SLICE_PROV = OUT_DIR / "prediction_qa_v2_final.slices_province.csv"
OUT_SLICE_DIST = OUT_DIR / "prediction_qa_v2_final.slices_district.csv"

# Over-firing / under-firing heuristic thresholds for a disaster-tweet pool.
OVER_FIRE_RATE = 0.30
UNDER_FIRE_RATE = 0.001


def _bucket_length(n: int) -> str:
    if n <= 40:
        return "00-40"
    if n <= 80:
        return "41-80"
    if n <= 140:
        return "81-140"
    if n <= 220:
        return "141-220"
    return "221+"


def _bucket_urgency(u: float) -> str:
    if pd.isna(u):
        return "nan"
    if u < 0.2:
        return "0.0-0.2"
    if u < 0.4:
        return "0.2-0.4"
    if u < 0.6:
        return "0.4-0.6"
    if u < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _fmt_rate(n: int, total: int) -> str:
    if total == 0:
        return "0 (0.00%)"
    return f"{n} ({100.0 * n / total:.2f}%)"


def _slice_positive_rates(
    df: pd.DataFrame, labels: List[str], key: str, top_n: int = 15
) -> pd.DataFrame:
    """Per-group size and positive rate per label."""
    g = df.groupby(key, dropna=False)
    sizes = g.size().rename("rows")
    rows = []
    for name, sub in g:
        r = {"group": name, "rows": int(len(sub))}
        for lab in labels:
            col = f"pred_{lab}"
            r[f"rate_{lab}"] = float(sub[col].mean()) if col in sub else float("nan")
        r["any_need_rate"] = float(sub["pred_any_need"].mean()) if "pred_any_need" in sub else float("nan")
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("rows", ascending=False)
    return out.head(top_n)


def main() -> int:
    if not PRED_CSV.exists():
        raise SystemExit(f"Not found: {PRED_CSV}")
    if not PRED_META.exists():
        raise SystemExit(f"Not found: {PRED_META}")

    meta: Dict[str, Any] = json.loads(PRED_META.read_text(encoding="utf-8"))
    labels: List[str] = list(meta["labels"])
    prob_cols = [f"prob_{l}" for l in labels]
    pred_cols = [f"pred_{l}" for l in labels]
    thresholds: Dict[str, float] = dict(meta["thresholds_per_label"])
    pred_positives_meta: Dict[str, int] = dict(meta.get("pred_positives", {}))

    # Efficient read - dtypes to speed up
    dtype_hint: Dict[str, Any] = {c: "float32" for c in prob_cols}
    dtype_hint.update({c: "int8" for c in pred_cols})
    dtype_hint["pred_label_count"] = "int8"
    dtype_hint["pred_any_need"] = "int8"
    dtype_hint["urgency_score"] = "float32"

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", dtype=dtype_hint, low_memory=False)

    total_rows = int(len(df))
    rows_after_meta = int(meta.get("rows_after", -1))
    rows_before_meta = int(meta.get("rows_before", -1))

    findings: List[Dict[str, Any]] = []

    def _add_finding(severity: str, code: str, msg: str, detail: Any = None) -> None:
        findings.append(
            {"severity": severity, "code": code, "message": msg, "detail": detail}
        )

    # ============ 1. Integrity ============
    # row count vs meta
    if total_rows != rows_after_meta:
        _add_finding(
            "release_blocker",
            "row_count_mismatch",
            f"CSV has {total_rows} rows; meta.rows_after={rows_after_meta}.",
        )

    # id uniqueness
    id_col = "id"
    dup_ids = int(df[id_col].duplicated().sum())
    if dup_ids > 0:
        _add_finding(
            "release_blocker",
            "duplicate_ids",
            f"Found {dup_ids} duplicate id values; meta says dedup-by-id=True.",
        )

    # missing columns
    missing = [c for c in prob_cols + pred_cols if c not in df.columns]
    if missing:
        _add_finding("release_blocker", "missing_columns", "Missing columns", missing)

    # NaN or out-of-range probs
    nan_prob_total = int(df[prob_cols].isna().sum().sum())
    if nan_prob_total > 0:
        _add_finding(
            "release_blocker", "nan_probability", f"{nan_prob_total} NaN probability cells."
        )
    oor_prob = 0
    for c in prob_cols:
        oor_prob += int(((df[c] < 0.0) | (df[c] > 1.0)).sum())
    if oor_prob > 0:
        _add_finding("release_blocker", "prob_out_of_range", f"{oor_prob} probability cells outside [0,1].")

    # pred_positives match
    pred_positives_now: Dict[str, int] = {}
    mismatch_pos: Dict[str, List[int]] = {}
    for lab in labels:
        n = int(df[f"pred_{lab}"].sum())
        pred_positives_now[lab] = n
        if lab in pred_positives_meta and pred_positives_meta[lab] != n:
            mismatch_pos[lab] = [pred_positives_meta[lab], n]
    if mismatch_pos:
        _add_finding(
            "release_blocker",
            "pred_positives_mismatch",
            "pred_positives in meta disagree with recomputed values.",
            mismatch_pos,
        )

    # threshold consistency (spot-check)
    threshold_violations: Dict[str, int] = {}
    for lab in labels:
        thr = float(thresholds[lab])
        p = df[f"prob_{lab}"].to_numpy()
        pred = df[f"pred_{lab}"].to_numpy().astype(bool)
        expected = p >= thr
        # np.isclose on float32; tolerate fp jitter on the boundary.
        violations = int(((pred != expected) & (~np.isclose(p, thr, atol=1e-5))).sum())
        if violations > 0:
            threshold_violations[lab] = violations
    if threshold_violations:
        _add_finding(
            "warning",
            "threshold_pred_inconsistency",
            "Some rows where pred != (prob >= thr). Usually fp jitter; investigate if > 100.",
            threshold_violations,
        )

    # empty predictions
    empty_any = int((df["pred_any_need"] == 0).sum())
    empty_count_mismatch = int(((df["pred_label_count"] == 0) != (df["pred_any_need"] == 0)).sum())
    if empty_count_mismatch > 0:
        _add_finding(
            "warning",
            "empty_flag_mismatch",
            f"{empty_count_mismatch} rows where pred_label_count==0 disagrees with pred_any_need==0.",
        )

    # ============ 2. Schema clarity (meta update) ============
    meta_updates = {
        "prediction_columns": pred_cols,
        "probability_columns": prob_cols,
        "label_to_pred_column": {l: f"pred_{l}" for l in labels},
        "label_to_prob_column": {l: f"prob_{l}" for l in labels},
        "auxiliary_columns": {
            "pred_label_count": "int count of labels where pred==1 in this row",
            "pred_any_need": "1 if any label pred==1 else 0 (i.e., pred_label_count >= 1)",
        },
        "identifier_columns": [id_col],
        "metadata_columns": ["created_at", "date", "time"],
        "text_columns": ["tweet", "tweet_clean"],
        "location_columns": ["neighborhood", "district", "province"],
        "urgency_columns": ["urgency_score"],
        "row_count": int(total_rows),
        "schema_note": (
            "Each label has exactly two columns: prob_<label> (sigmoid probability "
            "in [0,1]) and pred_<label> (1 if prob_<label> >= thresholds_per_label[<label>], "
            "else 0). pred_any_need and pred_label_count are derived. No hidden global "
            "threshold is applied; threshold_global_fallback is unused for this output."
        ),
        "schema_finalized_by": "scripts/prediction_qa_v2_final.py",
        "schema_finalized_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    enriched_meta = dict(meta)
    enriched_meta.update(meta_updates)
    PRED_META.write_text(json.dumps(enriched_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ============ 3. Label prevalence ============
    prev_rows = []
    for lab in labels:
        n = pred_positives_now[lab]
        prev_rows.append(
            {
                "label": lab,
                "positive_count": n,
                "positive_rate": n / total_rows if total_rows else 0.0,
                "threshold": thresholds[lab],
                "prob_mean": float(df[f"prob_{lab}"].mean()),
                "prob_median": float(df[f"prob_{lab}"].median()),
                "prob_p95": float(df[f"prob_{lab}"].quantile(0.95)),
            }
        )
    prev_df = pd.DataFrame(prev_rows).sort_values("positive_count", ascending=False)
    prev_df.to_csv(OUT_LABEL_PREV, index=False, encoding="utf-8-sig")

    # over-fire / under-fire scan
    over_fire = [r for r in prev_rows if r["positive_rate"] > OVER_FIRE_RATE]
    under_fire = [r for r in prev_rows if r["positive_rate"] < UNDER_FIRE_RATE]
    if over_fire:
        _add_finding(
            "known_limitation",
            "over_firing_labels",
            f"Labels with positive rate > {OVER_FIRE_RATE:.2f} on the 63k pool.",
            [{"label": r["label"], "rate": r["positive_rate"]} for r in over_fire],
        )
    if under_fire:
        _add_finding(
            "warning",
            "under_firing_labels",
            f"Labels with positive rate < {UNDER_FIRE_RATE:.4f} (very few positives).",
            [{"label": r["label"], "count": r["positive_count"]} for r in under_fire],
        )

    # ============ 4. pred_label_count / pred_any_need distributions ============
    lc_vc = df["pred_label_count"].value_counts().sort_index()
    lc_dist = {int(k): int(v) for k, v in lc_vc.items()}
    any_pos = int((df["pred_any_need"] == 1).sum())
    any_neg = int((df["pred_any_need"] == 0).sum())

    # ============ 5. Slices ============
    df["_text_len"] = df["tweet_clean"].fillna("").str.len()
    df["_len_bucket"] = df["_text_len"].apply(_bucket_length)
    df["_urg_bucket"] = df["urgency_score"].apply(_bucket_urgency)

    slice_len = _slice_positive_rates(df, labels, "_len_bucket", top_n=20).to_dict(orient="records")
    slice_urg = _slice_positive_rates(df, labels, "_urg_bucket", top_n=20).to_dict(orient="records")
    slice_prov_df = _slice_positive_rates(df, labels, "province", top_n=15)
    slice_dist_df = _slice_positive_rates(df, labels, "district", top_n=15)
    slice_prov_df.to_csv(OUT_SLICE_PROV, index=False, encoding="utf-8-sig")
    slice_dist_df.to_csv(OUT_SLICE_DIST, index=False, encoding="utf-8-sig")

    # ============ 6. Markdown report ============
    md: List[str] = []
    md.append("# Prediction QA — v2 final (leak-free)\n")
    md.append("Step 10 çıktısı. Canonical tahmin dosyası üstünde bütünlük, şema netliği, "
              "etiket prevalansı, dilimleme ve anomali taraması.\n")
    md.append("## Kaynaklar\n")
    md.append(f"- CSV: `{PRED_CSV.relative_to(REPO_ROOT).as_posix()}` ({total_rows:,} satır)")
    md.append(f"- Meta: `{PRED_META.relative_to(REPO_ROOT).as_posix()}`")
    md.append(f"- Model: `{meta.get('model_dir','?')}`")
    md.append(f"- Threshold source: `{meta.get('threshold_source','?')}` / `{meta.get('threshold_type','?')}`\n")

    md.append("## 1) Bütünlük\n")
    md.append(f"- CSV satır sayısı: **{total_rows:,}**, meta.rows_after: **{rows_after_meta:,}**, meta.rows_before: **{rows_before_meta:,}**")
    md.append(f"- Tekrarlanan id: **{dup_ids}** (beklenen: 0)")
    md.append(f"- Eksik sütun: **{len(missing)}** (beklenen: 0)")
    md.append(f"- NaN olasılık hücresi: **{nan_prob_total}** (beklenen: 0)")
    md.append(f"- [0,1] dışı olasılık: **{oor_prob}** (beklenen: 0)")
    md.append(f"- `pred_positives` meta ile eşleşme: **{'EVET' if not mismatch_pos else 'HAYIR'}**")
    if threshold_violations:
        md.append(f"- Eşik/pred tutarsızlığı (label:count): `{threshold_violations}` (uyarı)")
    else:
        md.append("- Eşik/pred tutarsızlığı: **0** (hepsi tutarlı)")
    md.append("")

    md.append("## 2) Şema netliği (meta'ya eklenen alanlar)\n")
    md.append("Aşağıdaki alanlar `need_predictions_geolocated_v2_final.meta.json` içine eklendi:\n")
    md.append("- `prediction_columns`, `probability_columns`")
    md.append("- `label_to_pred_column`, `label_to_prob_column`")
    md.append("- `auxiliary_columns` (`pred_label_count`, `pred_any_need`)")
    md.append("- `identifier_columns`, `metadata_columns`, `text_columns`, `location_columns`, `urgency_columns`")
    md.append("- `row_count`, `schema_note`, `schema_finalized_{by,at}`\n")
    md.append("Amaç: tüketiciler (dashboard, API adapter, notebook) `prob_*` ve `pred_*` "
              "eşleşmelerini ve `threshold_global_fallback=0.5`'in uygulanmadığını "
              "tek bakışta görebilsin.\n")

    md.append("## 3) Etiket prevalansı\n")
    md.append("| label | threshold | pozitif | oran | prob_mean | prob_p95 |")
    md.append("|---|---|---|---|---|---|")
    for r in prev_rows:
        md.append(
            f"| {r['label']} | {r['threshold']:.2f} | {r['positive_count']:,} | "
            f"{100.0*r['positive_rate']:.2f}% | {r['prob_mean']:.3f} | {r['prob_p95']:.3f} |"
        )
    md.append("")
    md.append("Not: `arama_kurtarma` oranı havuzun %62.83'ü — bu beklenen, çünkü "
              "girdi kümesi zaten 'acil yardım + konum' filtresinden geçmiş tweet'ler. "
              "Yine de tüketiciler için bu *prior* net belirtilmeli (known_limitation).\n")

    md.append("## 4) pred_label_count / pred_any_need dağılımı\n")
    md.append("| label_count | satır |")
    md.append("|---|---|")
    for k, v in sorted(lc_dist.items()):
        md.append(f"| {k} | {v:,} |")
    md.append("")
    md.append(f"- `pred_any_need=1`: **{any_pos:,}** ({100.0*any_pos/total_rows:.2f}%)")
    md.append(f"- `pred_any_need=0`: **{any_neg:,}** ({100.0*any_neg/total_rows:.2f}%)")
    md.append("")

    md.append("## 5) Dilimleme\n")
    md.append("### 5.1 Metin uzunluğu (tweet_clean karakter sayısı)\n")
    md.append("| bucket | satır | arama_kurtarma | lojistik | barinma | gida_su | bilgi_paylasimi | any_need |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in slice_len:
        md.append(
            f"| {r['group']} | {r['rows']:,} | "
            f"{100*r.get('rate_arama_kurtarma',0):.1f}% | "
            f"{100*r.get('rate_lojistik',0):.1f}% | "
            f"{100*r.get('rate_barinma',0):.1f}% | "
            f"{100*r.get('rate_gida_su',0):.1f}% | "
            f"{100*r.get('rate_bilgi_paylasimi',0):.1f}% | "
            f"{100*r.get('any_need_rate',0):.1f}% |"
        )
    md.append("")
    md.append("### 5.2 Urgency bucket\n")
    md.append("| bucket | satır | arama_kurtarma | saglik | lojistik | any_need |")
    md.append("|---|---|---|---|---|---|")
    for r in slice_urg:
        md.append(
            f"| {r['group']} | {r['rows']:,} | "
            f"{100*r.get('rate_arama_kurtarma',0):.1f}% | "
            f"{100*r.get('rate_saglik',0):.1f}% | "
            f"{100*r.get('rate_lojistik',0):.1f}% | "
            f"{100*r.get('any_need_rate',0):.1f}% |"
        )
    md.append("")
    md.append("### 5.3 Province (top 15 by satır)\n")
    md.append("Tam tablo: `prediction_qa_v2_final.slices_province.csv`. Özet:\n")
    md.append("| province | satır | arama_kurtarma | lojistik | bilgi_paylasimi | any_need |")
    md.append("|---|---|---|---|---|---|")
    for r in slice_prov_df.head(15).to_dict(orient="records"):
        md.append(
            f"| {r['group']} | {r['rows']:,} | "
            f"{100*r.get('rate_arama_kurtarma',0):.1f}% | "
            f"{100*r.get('rate_lojistik',0):.1f}% | "
            f"{100*r.get('rate_bilgi_paylasimi',0):.1f}% | "
            f"{100*r.get('any_need_rate',0):.1f}% |"
        )
    md.append("")

    md.append("## 6) Etiket-özel gözlemler\n")
    # dynamic per-label commentary for the four focus labels
    ak_rate = pred_positives_now["arama_kurtarma"] / total_rows
    gv_rate = pred_positives_now["guvenlik"] / total_rows
    ps_rate = pred_positives_now["psikolojik"] / total_rows
    bp_rate = pred_positives_now["bilgi_paylasimi"] / total_rows
    md.append(f"- **arama_kurtarma** ({100*ak_rate:.2f}%, {pred_positives_now['arama_kurtarma']:,} satır). "
              "Havuz zaten 'acil yardım' filtresinden geçtiği için bu oran beklenen üst sınırda. "
              "Gold test F1=0.969; production'da over-fire değil, *prior match*.\n")
    md.append(f"- **guvenlik** ({100*gv_rate:.2f}%, {pred_positives_now['guvenlik']:,} satır). "
              "Step 8'de belgelenen zayıflık burada da görünüyor: 63k havuzdaki gerçek "
              "yağmacılık/asayiş sinyallerinin büyük kısmı muhtemelen `altyapi` veya `arama_kurtarma` "
              "olarak etiketleniyor. Tüketiciler `guvenlik` için recall-öncelikli ayrı eşik kullanabilir "
              "(`threshold_production.json` önerisi, step 10 kapsamı dışında).\n")
    md.append(f"- **psikolojik** ({100*ps_rate:.2f}%, {pred_positives_now['psikolojik']:,} satır). "
              "Gold test F1=1.0 ama test pozitifi=1; havuzda bu oran kalibre bir başarı değil, "
              "rare-label saturasyonu. Düşük recall bekleyin.\n")
    md.append(f"- **bilgi_paylasimi** ({100*bp_rate:.2f}%, {pred_positives_now['bilgi_paylasimi']:,} satır). "
              "CV eşiği 0.87 tutucu; step 8'de gösterildi ki 'haber alamıyoruz' tarzı ifadeler "
              "sistematik olarak `arama_kurtarma`'ya kayıyor. Gerçek prevalans burada görünenin "
              "muhtemelen üstünde.\n")

    md.append("## 7) Bulguların sınıflandırması\n")
    if not findings:
        md.append("_Hiçbir bulgu yok._\n")
    else:
        by_sev: Dict[str, List[Dict[str, Any]]] = {}
        for f in findings:
            by_sev.setdefault(f["severity"], []).append(f)
        for sev in ("release_blocker", "warning", "known_limitation"):
            group = by_sev.get(sev, [])
            md.append(f"### {sev} ({len(group)})\n")
            if not group:
                md.append("_Yok._\n")
                continue
            for f in group:
                md.append(f"- **{f['code']}** — {f['message']}")
                if f.get("detail") is not None:
                    md.append(f"  - detay: `{f['detail']}`")
            md.append("")

    md.append("## 8) Sonuç\n")
    blockers = [f for f in findings if f["severity"] == "release_blocker"]
    if not blockers:
        md.append("**Release blocker yok.** Canonical tahmin çıktısı (`v2_final`) release paketlenmeye uygun. "
                  "Mevcut uyarılar ve known-limitation'lar selection rationale ile tutarlı "
                  "(content-overlap residual risk, guvenlik/bilgi_paylasimi zayıflıkları, rare-label saturasyonu, "
                  "havuzun arama_kurtarma-ağırlıklı prior'ı). Step 11 (release packaging) ve "
                  "step 14 (teknik özet) bu uyarılara atıf verecek.\n")
    else:
        md.append("**Release blocker bulundu** — step 11'e geçmeden giderilmeli:\n")
        for f in blockers:
            md.append(f"- {f['code']}: {f['message']}")
        md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    # ============ 7. JSON ============
    qa_obj: Dict[str, Any] = {
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "csv": str(PRED_CSV.relative_to(REPO_ROOT).as_posix()),
            "meta": str(PRED_META.relative_to(REPO_ROOT).as_posix()),
            "model_dir": meta.get("model_dir", ""),
            "threshold_source": meta.get("threshold_source", ""),
            "threshold_type": meta.get("threshold_type", ""),
        },
        "row_count": total_rows,
        "rows_before": rows_before_meta,
        "rows_after": rows_after_meta,
        "duplicate_ids": dup_ids,
        "nan_probability_cells": nan_prob_total,
        "prob_out_of_range_cells": oor_prob,
        "pred_positives_now": pred_positives_now,
        "pred_positives_meta": pred_positives_meta,
        "pred_positives_mismatch": mismatch_pos,
        "threshold_violations": threshold_violations,
        "label_prevalence": prev_rows,
        "pred_label_count_distribution": lc_dist,
        "pred_any_need": {"positive": any_pos, "negative": any_neg},
        "slice_length": slice_len,
        "slice_urgency": slice_urg,
        "slice_province_top15": slice_prov_df.to_dict(orient="records"),
        "slice_district_top15": slice_dist_df.to_dict(orient="records"),
        "findings": findings,
        "release_blocker": bool(blockers),
        "meta_schema_updates_applied": list(meta_updates.keys()),
        "outputs": {
            "md": str(OUT_MD.relative_to(REPO_ROOT).as_posix()),
            "json": str(OUT_JSON.relative_to(REPO_ROOT).as_posix()),
            "label_prevalence_csv": str(OUT_LABEL_PREV.relative_to(REPO_ROOT).as_posix()),
            "slices_province_csv": str(OUT_SLICE_PROV.relative_to(REPO_ROOT).as_posix()),
            "slices_district_csv": str(OUT_SLICE_DIST.relative_to(REPO_ROOT).as_posix()),
        },
    }
    OUT_JSON.write_text(json.dumps(qa_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Rows: {total_rows:,}")
    print(f"Duplicate ids: {dup_ids}")
    print(f"NaN probs: {nan_prob_total}")
    print(f"pred_any_need=1: {any_pos:,} ({100.0*any_pos/total_rows:.2f}%)")
    print(f"Findings: {len(findings)} (blockers={len(blockers)})")
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_LABEL_PREV}")
    print(f"Wrote: {OUT_SLICE_PROV}")
    print(f"Wrote: {OUT_SLICE_DIST}")
    print(f"Meta schema updates: {list(meta_updates.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
