#!/usr/bin/env python3
"""
Create a completed rare-label annotation pack from the candidate pool.

This fills the blank annotator pack with conservative auto-labels so the
next merge/split step can proceed without relying on an external manual pass.

Outputs:
- completed CSV in canonical gold column order
- XLSX copy for spreadsheet review
- TXT + JSON validation summaries
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ai_prefill_annotations import compile_patterns, label_text, norm


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

EXTRA_COLS: List[str] = ["aciliyet_0_3", "veracity_label", "notes"]
OUTPUT_COLS: List[str] = [*BASE_COLS, *NEED_LABEL_COLS, *EXTRA_COLS]
TARGET_LABELS: List[str] = ["guvenlik", "psikolojik", "bilgi_paylasimi"]

SECURITY_CONTEXT_RE = re.compile(
    r"\b(yagmaci\w*|yagma\w*|hirsiz\w*|asayis|kolluk|polis|asker|guvenlik|"
    r"can guvenligi|tehlike alt\w*|silah\w*)\b"
)
PET_MISSING_RE = re.compile(
    r"\b(kedi|kopek|kopegi|kopegim|kopegimiz|hayvan|yumak|pati|british shorthair|"
    r"cipli|lopez|duman)\b"
)
INFO_STRONG_RE = re.compile(
    r"\b(kayip|haber alam\w*|goren duyan|bilgisi olan|bilgi vereb\w*|duyuru|"
    r"duyurulur|bulamiyoruz|bulunamiyor|ulasam\w*|kayboldu|hastanede olabil\w*|"
    r"nerede oldugunu bulam\w*|kayip bildirimi|kayip ilani|yardim toplama alan|"
    r"yardim dagitim nokta|teslim almak|ucretsiz transfer)\b"
)
PROFESSION_NOISE_RE = re.compile(r"\bspor psikolog")
PHYSICAL_TRAUMA_RE = re.compile(r"(travmasi vardi)|((kafasinin|gozunde).{0,24}travma)")
AUTHOR_FEAR_RE = re.compile(r"dogumu baslamis.{0,120}cok korkuyorum|cok korkuyorum.{0,120}dogum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete the rare-label annotation pack.")
    parser.add_argument(
        "--annotation-pack",
        default="data/labeling/need_classification_rare_label_annotation_pack.csv",
        help="Blank rare-label annotation pack CSV.",
    )
    parser.add_argument(
        "--candidates",
        default="data/labeling/need_classification_rare_label_candidates.csv",
        help="Rare-label candidate pool CSV with rule metadata.",
    )
    parser.add_argument(
        "--output",
        default="data/labeling/need_classification_rare_label_annotation_pack.completed.csv",
        help="Completed output CSV path.",
    )
    parser.add_argument(
        "--xlsx-output",
        default="data/labeling/need_classification_rare_label_annotation_pack.completed.xlsx",
        help="Completed output XLSX path.",
    )
    parser.add_argument(
        "--report",
        default="data/analysis/need_classification_rare_label_annotation_pack.completed.report.txt",
        help="Validation report TXT path.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/analysis/need_classification_rare_label_annotation_pack.completed.summary.json",
        help="Validation summary JSON path.",
    )
    return parser.parse_args()


def parse_rule_set(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def ensure_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def classify_guvenlik(text_norm: str, rules: set[str], *, candidate_flag: str) -> Tuple[int, str]:
    if str(candidate_flag) != "1":
        return 0, ""

    if rules & {"armed_terms", "safety_risk", "security_request"}:
        return 1, ""

    text_no_hashtags = re.sub(r"#[\w_]+", " ", text_norm)
    if SECURITY_CONTEXT_RE.search(text_no_hashtags):
        return 1, ""

    return 0, "guvenlik_hashtag_noise"


def classify_psikolojik(text_norm: str, *, candidate_flag: str) -> Tuple[int, str]:
    if str(candidate_flag) != "1":
        return 0, ""

    if PROFESSION_NOISE_RE.search(text_norm):
        return 0, "psikolojik_profession_noise"
    if PHYSICAL_TRAUMA_RE.search(text_norm):
        return 0, "psikolojik_physical_trauma"
    if AUTHOR_FEAR_RE.search(text_norm):
        return 0, "psikolojik_author_fear"

    return 1, ""


def classify_bilgi_paylasimi(text_norm: str, rules: set[str], *, candidate_flag: str) -> Tuple[int, str]:
    if str(candidate_flag) != "1":
        return 0, ""

    if PET_MISSING_RE.search(text_norm):
        return 0, "bilgi_pet_missing"

    info_strong = bool(INFO_STRONG_RE.search(text_norm))

    if "missing_info" in rules:
        return 1, ""
    if "announcement" in rules:
        return 1, ""
    if "contact_request" in rules:
        return 1, ""
    if "contact_line" in rules and info_strong:
        return 1, ""
    if info_strong and rules & {"aid_ops", "share_call", "phone_like"}:
        return 1, ""

    return 0, "bilgi_weak_signal"


def rebuild_urgency(labels: Dict[str, int], prefill_urgency: int) -> int:
    urgency = max(prefill_urgency, 0)
    if labels["arama_kurtarma"] or labels["saglik"]:
        urgency = max(urgency, 3)
    if labels["guvenlik"]:
        urgency = max(urgency, 2)
    if labels["barinma"] or labels["gida_su"]:
        urgency = max(urgency, 1)
    if labels["altyapi"] or labels["lojistik"] or labels["psikolojik"]:
        urgency = max(urgency, 1)
    return min(urgency, 3)


def build_completed_pack(pack_df: pd.DataFrame, candidate_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    compiled = compile_patterns("A")
    candidate_meta = candidate_df.set_index("id").to_dict(orient="index")

    completed_rows: List[Dict[str, object]] = []
    exclusion_counter: Counter[str] = Counter()
    rare_positive_counter: Counter[str] = Counter()

    for row in pack_df.to_dict(orient="records"):
        row_id = str(row["id"]).strip()
        meta = candidate_meta[row_id]
        text = str(row.get("tweet_clean") or row.get("tweet") or "").strip()
        if not text:
            text = str(row.get("tweet") or "").strip()
        text_norm = norm(text)

        prefill = label_text(text, compiled)
        labels = {label: as_int(prefill.get(label, "0")) for label in NEED_LABEL_COLS}

        guvenlik, guvenlik_reason = classify_guvenlik(
            text_norm,
            parse_rule_set(meta.get("guvenlik_rules")),
            candidate_flag=meta.get("candidate_guvenlik", "0"),
        )
        psikolojik, psikolojik_reason = classify_psikolojik(
            text_norm,
            candidate_flag=meta.get("candidate_psikolojik", "0"),
        )
        bilgi, bilgi_reason = classify_bilgi_paylasimi(
            text_norm,
            parse_rule_set(meta.get("bilgi_paylasimi_rules")),
            candidate_flag=meta.get("candidate_bilgi_paylasimi", "0"),
        )

        labels["guvenlik"] = guvenlik
        labels["psikolojik"] = psikolojik
        labels["bilgi_paylasimi"] = bilgi

        notes: List[str] = []
        for reason in [guvenlik_reason, psikolojik_reason, bilgi_reason]:
            if reason:
                exclusion_counter[reason] += 1
                notes.append(reason)

        for label in TARGET_LABELS:
            if labels[label] == 1:
                rare_positive_counter[label] += 1

        aciliyet = rebuild_urgency(labels, as_int(prefill.get("aciliyet_0_3", "0")))
        veracity = str(prefill.get("veracity_label", "supheli") or "supheli").strip() or "supheli"

        completed = {col: str(row.get(col, "") or "") for col in BASE_COLS}
        for label in NEED_LABEL_COLS:
            completed[label] = str(labels[label])
        completed["aciliyet_0_3"] = str(aciliyet)
        completed["veracity_label"] = veracity
        completed["notes"] = "; ".join(notes)
        completed_rows.append(completed)

    completed_df = pd.DataFrame(completed_rows, columns=OUTPUT_COLS)
    summary = {
        "rows_completed": int(len(completed_df)),
        "rare_positive_counts": {label: int(rare_positive_counter[label]) for label in TARGET_LABELS},
        "rare_exclusion_counts": {key: int(value) for key, value in sorted(exclusion_counter.items())},
        "label_totals": {
            label: int(pd.to_numeric(completed_df[label], errors="coerce").fillna(0).sum())
            for label in NEED_LABEL_COLS
        },
        "urgency_distribution": {
            str(key): int(value)
            for key, value in completed_df["aciliyet_0_3"].value_counts(dropna=False).sort_index().to_dict().items()
        },
        "veracity_distribution": {
            str(key): int(value)
            for key, value in completed_df["veracity_label"].value_counts(dropna=False).to_dict().items()
        },
        "notes_non_empty": int(completed_df["notes"].astype(str).str.strip().ne("").sum()),
        "blank_label_cells": int(
            completed_df[NEED_LABEL_COLS + ["aciliyet_0_3", "veracity_label"]]
            .astype(str)
            .apply(lambda column: column.str.strip().eq(""))
            .sum()
            .sum()
        ),
        "duplicate_ids_in_output": int(completed_df["id"].duplicated().sum()),
    }
    return completed_df, summary


def write_excel_copy(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="rare_label_pack")

    workbook = load_workbook(path)
    sheet = workbook["rare_label_pack"]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    for cell in sheet["A"]:
        cell.number_format = "@"

    width_map = {
        "A": 22,
        "B": 23,
        "C": 12,
        "D": 10,
        "E": 18,
        "F": 18,
        "G": 18,
        "H": 14,
        "I": 70,
        "J": 70,
        "K": 16,
        "L": 12,
        "M": 12,
        "N": 12,
        "O": 12,
        "P": 12,
        "Q": 12,
        "R": 12,
        "S": 16,
        "T": 12,
        "U": 16,
        "V": 30,
    }
    for col_letter, width in width_map.items():
        sheet.column_dimensions[col_letter].width = width

    workbook.save(path)


def write_report(path: Path, summary: Dict[str, object], output_csv: Path, output_xlsx: Path) -> None:
    lines = [
        "COMPLETED RARE LABEL ANNOTATION PACK REPORT",
        "",
        f"- output csv: {output_csv.as_posix()}",
        f"- output xlsx: {output_xlsx.as_posix()}",
        f"- rows completed: {summary['rows_completed']}",
        f"- duplicate ids in output: {summary['duplicate_ids_in_output']}",
        f"- blank label cells: {summary['blank_label_cells']}",
        f"- non-empty notes rows: {summary['notes_non_empty']}",
        "",
        "RARE LABEL POSITIVES",
    ]
    for label, value in summary["rare_positive_counts"].items():
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "ALL LABEL TOTALS",
        ]
    )
    for label, value in summary["label_totals"].items():
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "URGENCY DISTRIBUTION",
        ]
    )
    for label, value in summary["urgency_distribution"].items():
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "VERACITY DISTRIBUTION",
        ]
    )
    for label, value in summary["veracity_distribution"].items():
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "RARE LABEL EXCLUSIONS",
        ]
    )
    exclusion_counts = summary["rare_exclusion_counts"]
    if exclusion_counts:
        for reason, value in exclusion_counts.items():
            lines.append(f"- {reason}: {value}")
    else:
        lines.append("- none")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    annotation_pack = Path(args.annotation_pack)
    candidates = Path(args.candidates)
    output_csv = Path(args.output)
    output_xlsx = Path(args.xlsx_output)
    report_path = Path(args.report)
    summary_json = Path(args.summary_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    pack_df = pd.read_csv(annotation_pack, dtype=str).fillna("")
    candidate_df = pd.read_csv(candidates, dtype=str).fillna("")

    ensure_columns(pack_df, OUTPUT_COLS, name="annotation pack")
    ensure_columns(
        candidate_df,
        ["id", "candidate_guvenlik", "candidate_psikolojik", "candidate_bilgi_paylasimi", "guvenlik_rules", "bilgi_paylasimi_rules"],
        name="candidate pool",
    )

    pack_ids = pack_df["id"].astype(str).str.strip()
    candidate_ids = set(candidate_df["id"].astype(str).str.strip())
    missing_ids = sorted(set(pack_ids) - candidate_ids)
    if missing_ids:
        raise ValueError(f"candidate metadata missing for {len(missing_ids)} ids, first few: {missing_ids[:5]}")

    completed_df, summary = build_completed_pack(pack_df, candidate_df)

    completed_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    write_excel_copy(completed_df, output_xlsx)

    summary.update(
        {
            "annotation_pack": annotation_pack.as_posix(),
            "candidate_pool": candidates.as_posix(),
            "output_csv": output_csv.as_posix(),
            "output_xlsx": output_xlsx.as_posix(),
            "report_txt": report_path.as_posix(),
            "column_order": OUTPUT_COLS,
        }
    )

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary, output_csv, output_xlsx)

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote XLSX: {output_xlsx}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
