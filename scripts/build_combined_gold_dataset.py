#!/usr/bin/env python3
"""
Build a canonical combined gold dataset from the original gold CSV and the
newly completed gold extension CSV.

If the same tweet id appears in both files, the extension row wins by default.
This lets updated annotations replace earlier labels while keeping provenance
counts in a small JSON report.
"""

from __future__ import annotations

import argparse
import json
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
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()
        raise SystemExit(f"{path.name} contains duplicate ids: {dupes[:20]}")
    return df


def _count_changed_overlap(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> int:
    base_idx = base_df.set_index("id")
    ext_idx = ext_df.set_index("id")
    overlap_ids = sorted(set(base_idx.index) & set(ext_idx.index))
    changed = 0
    for oid in overlap_ids:
        base_row = base_idx.loc[oid]
        ext_row = ext_idx.loc[oid]
        if any(str(base_row[c]) != str(ext_row[c]) for c in COMPARE_COLUMNS):
            changed += 1
    return changed


def _build_report(
    *,
    base_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    prefer: str,
) -> Dict[str, object]:
    base_ids = set(base_df["id"].tolist())
    ext_ids = set(ext_df["id"].tolist())
    overlap = base_ids & ext_ids
    return {
        "base_rows": int(len(base_df)),
        "extension_rows": int(len(ext_df)),
        "combined_rows": int(len(combined_df)),
        "prefer": prefer,
        "overlap_ids": int(len(overlap)),
        "changed_overlap_rows": int(_count_changed_overlap(base_df, ext_df)),
        "base_only_rows": int(len(base_ids - ext_ids)),
        "extension_only_rows": int(len(ext_ids - base_ids)),
        "column_order": EXPECTED_COLUMNS,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build combined gold dataset with extension precedence.")
    p.add_argument("--base", default="data/need_classification_gold.csv", help="Original base gold CSV path.")
    p.add_argument(
        "--extension",
        default="data/need_classification_gold_extension_800.csv",
        help="Completed gold extension CSV path.",
    )
    p.add_argument(
        "--output",
        default="data/need_classification_gold_combined.csv",
        help="Output CSV path for the combined gold dataset.",
    )
    p.add_argument(
        "--report-json",
        default="data/need_classification_gold_combined.report.json",
        help="Machine-readable merge report output path.",
    )
    p.add_argument(
        "--prefer",
        choices=["base", "extension"],
        default="extension",
        help="Which row wins when the same id exists in both datasets.",
    )
    args = p.parse_args()

    base_path = Path(args.base)
    ext_path = Path(args.extension)
    output_path = Path(args.output)
    report_path = Path(args.report_json)

    base_df = _load_csv(base_path)
    ext_df = _load_csv(ext_path)

    ordered_frames = [base_df, ext_df] if args.prefer == "extension" else [ext_df, base_df]
    combined = pd.concat(ordered_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    combined = combined[EXPECTED_COLUMNS].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    report = _build_report(
        base_df=base_df,
        ext_df=ext_df,
        combined_df=combined,
        prefer=args.prefer,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote combined gold CSV: {output_path}")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
