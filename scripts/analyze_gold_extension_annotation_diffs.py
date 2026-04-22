#!/usr/bin/env python3
"""
Analyze annotation differences between the original gold set and the newer
extension for overlap ids, then export row-level diffs plus compact summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

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


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing required columns: {missing}")
    df = df[EXPECTED_COLUMNS].copy()
    df["id"] = df["id"].astype("string").fillna("").str.strip()
    if df["id"].eq("").any():
        raise SystemExit(f"{path.name} contains blank id values.")
    if df["id"].duplicated().any():
        raise SystemExit(f"{path.name} contains duplicate ids.")
    return df


def _norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_int_str(value: object) -> str:
    text = _norm(value)
    if text == "":
        return "0"
    return text


def _build_diff_rows(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> List[Dict[str, object]]:
    base_idx = base_df.set_index("id")
    ext_idx = ext_df.set_index("id")
    overlap_ids = sorted(set(base_idx.index) & set(ext_idx.index))

    out: List[Dict[str, object]] = []
    for oid in overlap_ids:
        before = base_idx.loc[oid]
        after = ext_idx.loc[oid]
        changed = [c for c in COMPARE_COLUMNS if _norm(before[c]) != _norm(after[c])]
        if not changed:
            continue

        row: Dict[str, object] = {
            "id": oid,
            "diff_signature": "|".join(changed),
            "changed_columns": ", ".join(changed),
            "changed_column_count": len(changed),
            "tweet_clean": _norm(after["tweet_clean"]),
            "base_aciliyet_0_3": _to_int_str(before[ACILIYET_COL]),
            "ext_aciliyet_0_3": _to_int_str(after[ACILIYET_COL]),
            "base_veracity_label": _norm(before[VERACITY_COL]),
            "ext_veracity_label": _norm(after[VERACITY_COL]),
            "base_notes": _norm(before[NOTES_COL]),
            "ext_notes": _norm(after[NOTES_COL]),
        }
        for col in NEED_LABEL_COLS:
            row[f"base_{col}"] = _to_int_str(before[col])
            row[f"ext_{col}"] = _to_int_str(after[col])
            row[f"{col}_changed"] = int(_norm(before[col]) != _norm(after[col]))
        out.append(row)
    return out


def _summary_payload(base_df: pd.DataFrame, ext_df: pd.DataFrame, diff_rows: List[Dict[str, object]]) -> Dict[str, object]:
    base_ids = set(base_df["id"].tolist())
    ext_ids = set(ext_df["id"].tolist())

    signature_counts = Counter(row["diff_signature"] for row in diff_rows)
    label_transition_counts: Dict[str, Dict[str, int]] = {}
    for col in NEED_LABEL_COLS:
        counter: Counter[str] = Counter()
        for row in diff_rows:
            left = row[f"base_{col}"]
            right = row[f"ext_{col}"]
            if left != right:
                counter[f"{left}->{right}"] += 1
        label_transition_counts[col] = dict(counter)

    aciliyet_counter: Counter[str] = Counter()
    veracity_counter: Counter[str] = Counter()
    for row in diff_rows:
        if row["base_aciliyet_0_3"] != row["ext_aciliyet_0_3"]:
            aciliyet_counter[f"{row['base_aciliyet_0_3']}->{row['ext_aciliyet_0_3']}"] += 1
        if row["base_veracity_label"] != row["ext_veracity_label"]:
            veracity_counter[f"{row['base_veracity_label']}->{row['ext_veracity_label']}"] += 1

    signature_examples: Dict[str, List[str]] = defaultdict(list)
    for row in diff_rows:
        key = str(row["diff_signature"])
        if len(signature_examples[key]) < 5:
            signature_examples[key].append(str(row["id"]))

    return {
        "base_rows": int(len(base_df)),
        "extension_rows": int(len(ext_df)),
        "overlap_ids": int(len(base_ids & ext_ids)),
        "changed_overlap_rows": int(len(diff_rows)),
        "signature_counts": dict(signature_counts),
        "signature_example_ids": dict(signature_examples),
        "label_transition_counts": label_transition_counts,
        "aciliyet_transition_counts": dict(aciliyet_counter),
        "veracity_transition_counts": dict(veracity_counter),
    }


def _render_summary_text(summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("GOLD VS EXTENSION ANNOTATION DIFF SUMMARY")
    lines.append("")
    lines.append(f"- base rows: {summary['base_rows']}")
    lines.append(f"- extension rows: {summary['extension_rows']}")
    lines.append(f"- overlap ids: {summary['overlap_ids']}")
    lines.append(f"- changed overlap rows: {summary['changed_overlap_rows']}")
    lines.append("")

    lines.append("DIFF SIGNATURE COUNTS")
    for key, value in sorted(summary["signature_counts"].items(), key=lambda x: (-x[1], x[0])):
        sample_ids = ", ".join(summary["signature_example_ids"].get(key, []))
        lines.append(f"- {key}: {value} (sample ids: {sample_ids})")
    lines.append("")

    lines.append("LABEL TRANSITIONS")
    for label, transitions in summary["label_transition_counts"].items():
        if transitions:
            lines.append(f"- {label}: {transitions}")
    lines.append("")

    lines.append("ACILIYET TRANSITIONS")
    lines.append(f"- {summary['aciliyet_transition_counts']}")
    lines.append("")

    lines.append("VERACITY TRANSITIONS")
    lines.append(f"- {summary['veracity_transition_counts']}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze annotation diffs between old gold and new extension.")
    parser.add_argument("--base", default="data/need_classification_gold.csv", help="Original gold CSV path.")
    parser.add_argument(
        "--extension",
        default="data/need_classification_gold_extension_800.csv",
        help="New extension CSV path.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/analysis/need_classification_gold_extension_changed_overlaps.csv",
        help="Row-level diff export CSV.",
    )
    parser.add_argument(
        "--out-json",
        default="data/analysis/need_classification_gold_extension_changed_overlaps.summary.json",
        help="Machine-readable summary JSON.",
    )
    parser.add_argument(
        "--out-txt",
        default="data/analysis/need_classification_gold_extension_changed_overlaps.summary.txt",
        help="Human-readable summary text.",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    ext_path = Path(args.extension)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_txt = Path(args.out_txt)

    base_df = _load_csv(base_path)
    ext_df = _load_csv(ext_path)
    diff_rows = _build_diff_rows(base_df, ext_df)
    summary = _summary_payload(base_df, ext_df, diff_rows)

    diff_df = pd.DataFrame(diff_rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    diff_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_txt.write_text(_render_summary_text(summary), encoding="utf-8")

    print(f"Wrote diff CSV: {out_csv}")
    print(f"Wrote summary JSON: {out_json}")
    print(f"Wrote summary TXT: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
