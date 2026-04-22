#!/usr/bin/env python3
"""
Audit the combined gold need-classification dataset for schema, completeness,
duplicate ids, label balance, overlap conflicts between old/new annotations,
and a few practical risk signals for downstream modeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


NEED_LABEL_COLS: List[str] = [
    "arama_kurtarma",
    "saglik",
    "barinma",
    "gida_su",
    "altyapi",
    "guvenlik",
    "lojistik",
    "psikolojik",
    "bilgi_paylasimi",
]

BASE_COLS: List[str] = [
    "id",
    "created_at",
    "date",
    "time",
    "neighborhood",
    "district",
    "province",
    "urgency_score",
    "tweet",
    "tweet_clean",
]

ACILIYET_COL = "aciliyet_0_3"
VERACITY_COL = "veracity_label"
NOTES_COL = "notes"

EXPECTED_COLUMNS: List[str] = [
    *BASE_COLS,
    *NEED_LABEL_COLS,
    ACILIYET_COL,
    VERACITY_COL,
    NOTES_COL,
]

COMPARE_COLUMNS: List[str] = [*NEED_LABEL_COLS, ACILIYET_COL, VERACITY_COL, NOTES_COL]
CRITICAL_TEXT_COLS: List[str] = ["id", "tweet", "tweet_clean", "created_at", ACILIYET_COL, VERACITY_COL]
PLACEHOLDER_VALUES = {"unknown", "bilinmiyor", "none", "nan", "null", "n/a"}


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype="string")


def _normalize_string_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _blank_count(s: pd.Series) -> int:
    normalized = _normalize_string_series(s)
    return int(normalized.eq("").sum())


def _placeholder_count(s: pd.Series) -> int:
    normalized = _normalize_string_series(s).str.lower()
    return int(normalized.isin(PLACEHOLDER_VALUES).sum())


def _schema_info(df: pd.DataFrame) -> Dict[str, object]:
    cols = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    same_order = cols == EXPECTED_COLUMNS
    return {
        "rows": int(len(df)),
        "cols": int(len(cols)),
        "missing_columns": missing,
        "extra_columns": extra,
        "matches_expected_order": same_order,
    }


def _coerce_binary(df: pd.DataFrame, col: str) -> pd.Series:
    s = _normalize_string_series(df[col]).replace({"": "0"})
    return pd.to_numeric(s, errors="coerce")


def _label_distribution(df: pd.DataFrame) -> Dict[str, object]:
    label_df = pd.DataFrame({c: _coerce_binary(df, c).fillna(0).astype(int) for c in NEED_LABEL_COLS})
    total_rows = int(len(df))
    positives = {c: int(label_df[c].sum()) for c in NEED_LABEL_COLS}
    prevalence = {
        c: round((positives[c] / total_rows) * 100.0, 3) if total_rows else 0.0 for c in NEED_LABEL_COLS
    }
    label_cardinality = label_df.sum(axis=1)
    return {
        "positives": positives,
        "prevalence_pct": prevalence,
        "rows_with_any_need": int((label_cardinality > 0).sum()),
        "rows_with_no_need": int((label_cardinality == 0).sum()),
        "label_cardinality_distribution": {
            str(int(k)): int(v) for k, v in label_cardinality.value_counts().sort_index().to_dict().items()
        },
    }


def _validation_info(df: pd.DataFrame) -> Dict[str, object]:
    binary_invalid: Dict[str, int] = {}
    for col in NEED_LABEL_COLS:
        numeric = _coerce_binary(df, col)
        binary_invalid[col] = int((~numeric.isin([0, 1]) & numeric.notna()).sum())

    ac_numeric = pd.to_numeric(_normalize_string_series(df[ACILIYET_COL]).replace({"": "0"}), errors="coerce")
    ac_invalid = int((~ac_numeric.isin([0, 1, 2, 3]) & ac_numeric.notna()).sum())

    veracity = _normalize_string_series(df[VERACITY_COL]).replace({"": "<BLANK>"})
    veracity_counts = {str(k): int(v) for k, v in veracity.value_counts().to_dict().items()}
    expected_veracity = {"supheli", "dogrulanmis", "asilsiz"}
    unexpected_veracity = sorted(
        {v for v in veracity.unique().tolist() if v not in expected_veracity and v != "<BLANK>"}
    )

    return {
        "binary_label_invalid_rows": binary_invalid,
        "aciliyet_distribution": {
            str(int(k)): int(v) for k, v in ac_numeric.fillna(-1).astype(int).value_counts().sort_index().to_dict().items()
        },
        "aciliyet_invalid_rows": ac_invalid,
        "veracity_distribution": veracity_counts,
        "unexpected_veracity_values": unexpected_veracity,
    }


def _blank_info(df: pd.DataFrame) -> Dict[str, object]:
    counts: Dict[str, Dict[str, int]] = {}
    for col in EXPECTED_COLUMNS:
        counts[col] = {
            "blank": _blank_count(df[col]),
            "placeholder": _placeholder_count(df[col]),
        }
    return {
        "by_column": counts,
        "critical_blank_columns": {c: counts[c]["blank"] for c in CRITICAL_TEXT_COLS},
        "location_incomplete_rows": {
            "district_blank_or_placeholder": counts["district"]["blank"] + counts["district"]["placeholder"],
            "province_blank_or_placeholder": counts["province"]["blank"] + counts["province"]["placeholder"],
        },
    }


def _duplicate_info(df: pd.DataFrame) -> Dict[str, object]:
    ids = _normalize_string_series(df["id"])
    duplicate_ids = ids[ids.duplicated(keep=False)].tolist()

    tweet_clean = _normalize_string_series(df["tweet_clean"])
    dup_mask = tweet_clean.ne("") & tweet_clean.duplicated(keep=False)
    dup_df = df.loc[dup_mask, ["id", "tweet_clean"]].copy()

    label_conflicts: List[Dict[str, object]] = []
    if not dup_df.empty:
        full_df = df.assign(tweet_clean_norm=tweet_clean)
        for text, sub in full_df.groupby("tweet_clean_norm", dropna=False):
            if len(sub) < 2 or not str(text).strip():
                continue
            changed = [
                c
                for c in [*NEED_LABEL_COLS, ACILIYET_COL, VERACITY_COL]
                if _normalize_string_series(sub[c]).nunique() > 1
            ]
            if changed:
                label_conflicts.append(
                    {
                        "rows": int(len(sub)),
                        "changed_columns": changed,
                        "sample_text": str(text)[:220],
                        "ids": _normalize_string_series(sub["id"]).tolist(),
                    }
                )

    return {
        "duplicate_id_count": int(len(set(duplicate_ids))),
        "duplicate_ids_sample": sorted(set(duplicate_ids))[:20],
        "duplicate_tweet_clean_groups": int(dup_df["tweet_clean"].nunique()) if not dup_df.empty else 0,
        "duplicate_tweet_clean_rows": int(len(dup_df)),
        "duplicate_tweet_conflict_groups": int(len(label_conflicts)),
        "duplicate_tweet_conflict_sample": label_conflicts[:10],
    }


def _canonical_merge(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    available_cols = [c for c in EXPECTED_COLUMNS if c in base_df.columns and c in ext_df.columns]
    canonical = pd.concat([base_df[available_cols], ext_df[available_cols]], ignore_index=True)
    canonical = canonical.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    return canonical[available_cols].copy()


def _merge_consistency(base_df: pd.DataFrame, ext_df: pd.DataFrame, combined_df: pd.DataFrame) -> Dict[str, object]:
    if any(c not in combined_df.columns for c in EXPECTED_COLUMNS):
        return {
            "matches_canonical_extension_precedence": False,
            "reason": "combined dataset is missing expected columns",
            "mismatched_ids_sample": [],
        }

    canonical = _canonical_merge(base_df, ext_df).sort_values("id").set_index("id")
    combined_idx = combined_df[EXPECTED_COLUMNS].sort_values("id").set_index("id")

    same_ids = canonical.index.equals(combined_idx.index)
    mismatched_ids: List[str] = []
    if same_ids:
        for oid in canonical.index:
            left = canonical.loc[oid].fillna("").astype(str)
            right = combined_idx.loc[oid].fillna("").astype(str)
            if any(left[c] != right[c] for c in EXPECTED_COLUMNS if c != "id"):
                mismatched_ids.append(str(oid))

    return {
        "matches_canonical_extension_precedence": bool(same_ids and not mismatched_ids),
        "same_id_set": bool(same_ids),
        "canonical_rows": int(len(canonical)),
        "combined_rows": int(len(combined_idx)),
        "mismatched_ids_sample": mismatched_ids[:20],
    }


def _overlap_changes(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> Dict[str, object]:
    if "id" not in base_df.columns or "id" not in ext_df.columns:
        return {
            "overlap_ids": 0,
            "changed_rows": 0,
            "changed_column_counts": {},
            "changed_rows_sample": [],
        }

    left = base_df[EXPECTED_COLUMNS].copy()
    right = ext_df[EXPECTED_COLUMNS].copy()
    left["id"] = _normalize_string_series(left["id"])
    right["id"] = _normalize_string_series(right["id"])
    left = left.set_index("id")
    right = right.set_index("id")

    overlap_ids = sorted(set(left.index) & set(right.index))
    changed_column_counts = {c: 0 for c in COMPARE_COLUMNS}
    changed_rows_sample: List[Dict[str, object]] = []
    changed_rows = 0

    for oid in overlap_ids:
        before = left.loc[oid]
        after = right.loc[oid]
        changed_cols = []
        for col in COMPARE_COLUMNS:
            left_val = "" if pd.isna(before[col]) else str(before[col]).strip()
            right_val = "" if pd.isna(after[col]) else str(after[col]).strip()
            if left_val != right_val:
                changed_cols.append(col)
                changed_column_counts[col] += 1
        if changed_cols:
            changed_rows += 1
            if len(changed_rows_sample) < 20:
                changed_rows_sample.append(
                    {
                        "id": str(oid),
                        "changed_columns": changed_cols,
                        "base_values": {c: ("" if pd.isna(before[c]) else str(before[c])) for c in changed_cols},
                        "extension_values": {c: ("" if pd.isna(after[c]) else str(after[c])) for c in changed_cols},
                        "tweet_clean_sample": ("" if pd.isna(after["tweet_clean"]) else str(after["tweet_clean"]))[:220],
                    }
                )

    return {
        "overlap_ids": int(len(overlap_ids)),
        "changed_rows": int(changed_rows),
        "changed_column_counts": {k: v for k, v in changed_column_counts.items() if v > 0},
        "changed_rows_sample": changed_rows_sample,
    }


def _risk_flags(
    *,
    label_distribution: Dict[str, object],
    blanks: Dict[str, object],
    duplicates: Dict[str, object],
    overlap: Dict[str, object],
    validation: Dict[str, object],
    merge_consistency: Dict[str, object],
    combined_df: pd.DataFrame,
) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    positives = label_distribution["positives"]

    for label, count in positives.items():
        if count == 0:
            flags.append(
                {
                    "severity": "high",
                    "title": f"{label} etiketinde pozitif örnek yok",
                    "detail": f"{label} için combined sette 0 pozitif var; bu etiket için güvenilir threshold veya model öğrenimi yapılamaz.",
                }
            )
        elif count < 10:
            flags.append(
                {
                    "severity": "high",
                    "title": f"{label} etiketi aşırı nadir",
                    "detail": f"{label} etiketi yalnızca {count} pozitif içeriyor; val/test metrikleri yüksek varyanslı olur.",
                }
            )
        elif count < 25:
            flags.append(
                {
                    "severity": "medium",
                    "title": f"{label} etiketi halen kırılgan",
                    "detail": f"{label} etiketi {count} pozitif ile düşük destekli; split ve threshold sonuçları dikkatle yorumlanmalı.",
                }
            )

    if not merge_consistency.get("matches_canonical_extension_precedence", False):
        flags.append(
            {
                "severity": "high",
                "title": "Combined dosya canonical merge ile uyuşmuyor",
                "detail": "Base + extension (extension wins) birleştirmesi ile mevcut combined CSV arasında fark var.",
            }
        )

    overlap_ids = int(overlap.get("overlap_ids", 0))
    changed_rows = int(overlap.get("changed_rows", 0))
    if overlap_ids and changed_rows:
        ratio = (changed_rows / overlap_ids) * 100.0
        flags.append(
            {
                "severity": "medium",
                "title": "Eski ve yeni anotasyonlarda çakışma mevcut",
                "detail": f"{overlap_ids} ortak id'nin {changed_rows} tanesinde (%{ratio:.1f}) anotasyon farkı var; özellikle guideline tarafında gözden geçirilmeli.",
            }
        )

    district_incomplete = int(blanks["location_incomplete_rows"]["district_blank_or_placeholder"])
    province_incomplete = int(blanks["location_incomplete_rows"]["province_blank_or_placeholder"])
    if district_incomplete or province_incomplete:
        flags.append(
            {
                "severity": "medium",
                "title": "Konum yardımcı kolonlarında eksik/placeholder değerler var",
                "detail": f"district için {district_incomplete}, province için {province_incomplete} satır blank veya placeholder içeriyor.",
            }
        )

    if int(duplicates["duplicate_tweet_conflict_groups"]) > 0:
        flags.append(
            {
                "severity": "medium",
                "title": "Aynı tweet metninde etiket tutarsızlığı var",
                "detail": f"{duplicates['duplicate_tweet_conflict_groups']} duplicate metin grubunda etiket veya yardımcı kolon farkı bulundu.",
            }
        )

    veracity_dist = validation.get("veracity_distribution", {})
    total_rows = int(len(combined_df))
    if total_rows:
        supheli = int(veracity_dist.get("supheli", 0))
        if (supheli / total_rows) >= 0.90:
            flags.append(
                {
                    "severity": "low",
                    "title": "Veracity dağılımı çok tek taraflı",
                    "detail": f"veracity_label satırlarının {supheli}/{total_rows} kadarı supheli; bu kolon karar destek için sınırlı sinyal taşıyor.",
                }
            )

    label_df = pd.DataFrame({c: _coerce_binary(combined_df, c).fillna(0).astype(int) for c in NEED_LABEL_COLS})
    aciliyet = pd.to_numeric(
        _normalize_string_series(combined_df[ACILIYET_COL]).replace({"": "0"}),
        errors="coerce",
    ).fillna(-1).astype(int)
    need_count = label_df.sum(axis=1)
    need_positive_ac0 = int(((need_count > 0) & (aciliyet == 0)).sum())
    if need_positive_ac0 > 0:
        flags.append(
            {
                "severity": "low",
                "title": "Pozitif ihtiyaca rağmen aciliyet=0 örnekleri var",
                "detail": f"{need_positive_ac0} satırda en az bir ihtiyaç etiketi pozitifken aciliyet_0_3=0. Tanım gereği mümkünse sorun yok, değilse tekrar bakılmalı.",
            }
        )

    return flags


def _render_text_report(
    *,
    base_path: Path,
    ext_path: Path,
    combined_path: Path,
    report: Dict[str, object],
) -> str:
    schema = report["schema"]
    blanks = report["blank_counts"]
    duplicates = report["duplicate_checks"]
    labels = report["label_distribution"]
    overlap = report["overlap_changes"]
    validation = report["value_validation"]
    risks = report["risk_flags"]
    merge = report["merge_consistency"]

    lines: List[str] = []
    lines.append("COMBINED GOLD DATA QUALITY AUDIT")
    lines.append("")
    lines.append(f"Base: {base_path.as_posix()}")
    lines.append(f"Extension: {ext_path.as_posix()}")
    lines.append(f"Combined: {combined_path.as_posix()}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append(
        f"- rows: base={schema['base']['rows']} extension={schema['extension']['rows']} combined={schema['combined']['rows']}"
    )
    lines.append(
        f"- schema/order match: base={schema['base']['matches_expected_order']} extension={schema['extension']['matches_expected_order']} combined={schema['combined']['matches_expected_order']}"
    )
    lines.append(f"- duplicate ids in combined: {duplicates['duplicate_id_count']}")
    lines.append(
        f"- canonical merge match (extension wins): {merge['matches_canonical_extension_precedence']}"
    )
    lines.append("")

    lines.append("CRITICAL BLANKS")
    for col, count in blanks["critical_blank_columns"].items():
        lines.append(f"- {col}: {count}")
    loc_rows = blanks["location_incomplete_rows"]
    lines.append(f"- district blank/placeholder: {loc_rows['district_blank_or_placeholder']}")
    lines.append(f"- province blank/placeholder: {loc_rows['province_blank_or_placeholder']}")
    lines.append("")

    lines.append("LABEL DISTRIBUTION")
    for label in NEED_LABEL_COLS:
        count = labels["positives"][label]
        prev = labels["prevalence_pct"][label]
        lines.append(f"- {label}: {count} ({prev:.3f}%)")
    lines.append(f"- rows with any need: {labels['rows_with_any_need']}")
    lines.append(f"- rows with no need: {labels['rows_with_no_need']}")
    lines.append(f"- label cardinality: {labels['label_cardinality_distribution']}")
    lines.append("")

    lines.append("AUXILIARY DISTRIBUTIONS")
    lines.append(f"- aciliyet_0_3: {validation['aciliyet_distribution']}")
    lines.append(f"- veracity_label: {validation['veracity_distribution']}")
    lines.append("")

    lines.append("OLD VS NEW OVERLAP")
    lines.append(f"- overlap ids: {overlap['overlap_ids']}")
    lines.append(f"- changed overlap rows: {overlap['changed_rows']}")
    lines.append(f"- changed columns: {overlap['changed_column_counts']}")
    lines.append("")

    lines.append("DUPLICATE TEXT CHECK")
    lines.append(f"- duplicate tweet_clean groups: {duplicates['duplicate_tweet_clean_groups']}")
    lines.append(f"- duplicate tweet_clean rows: {duplicates['duplicate_tweet_clean_rows']}")
    lines.append(f"- duplicate text groups with label conflict: {duplicates['duplicate_tweet_conflict_groups']}")
    lines.append("")

    lines.append("RISK FLAGS")
    for item in risks:
        lines.append(f"- [{item['severity'].upper()}] {item['title']}: {item['detail']}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit combined gold dataset quality.")
    parser.add_argument("--base", default="data/need_classification_gold.csv", help="Original gold CSV.")
    parser.add_argument(
        "--extension",
        default="data/need_classification_gold_extension_800.csv",
        help="Gold extension CSV.",
    )
    parser.add_argument(
        "--combined",
        default="data/need_classification_gold_combined.csv",
        help="Combined gold CSV to audit.",
    )
    parser.add_argument(
        "--out-json",
        default="data/need_classification_gold_combined.audit.json",
        help="Machine-readable audit report.",
    )
    parser.add_argument(
        "--out-txt",
        default="data/need_classification_gold_combined.audit.txt",
        help="Human-readable audit report.",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    ext_path = Path(args.extension)
    combined_path = Path(args.combined)
    out_json = Path(args.out_json)
    out_txt = Path(args.out_txt)

    base_df = _load_csv(base_path)
    ext_df = _load_csv(ext_path)
    combined_df = _load_csv(combined_path)

    report: Dict[str, object] = {
        "schema": {
            "base": _schema_info(base_df),
            "extension": _schema_info(ext_df),
            "combined": _schema_info(combined_df),
            "expected_columns": EXPECTED_COLUMNS,
        },
        "blank_counts": _blank_info(combined_df),
        "duplicate_checks": _duplicate_info(combined_df),
        "label_distribution": _label_distribution(combined_df),
        "value_validation": _validation_info(combined_df),
        "merge_consistency": _merge_consistency(base_df, ext_df, combined_df),
        "overlap_changes": _overlap_changes(base_df, ext_df),
    }
    report["risk_flags"] = _risk_flags(
        label_distribution=report["label_distribution"],
        blanks=report["blank_counts"],
        duplicates=report["duplicate_checks"],
        overlap=report["overlap_changes"],
        validation=report["value_validation"],
        merge_consistency=report["merge_consistency"],
        combined_df=combined_df,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_txt.write_text(
        _render_text_report(
            base_path=base_path,
            ext_path=ext_path,
            combined_path=combined_path,
            report=report,
        ),
        encoding="utf-8",
    )

    print(f"Wrote JSON report: {out_json}")
    print(f"Wrote text report: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
