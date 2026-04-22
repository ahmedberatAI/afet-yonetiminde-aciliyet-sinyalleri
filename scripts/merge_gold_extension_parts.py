#!/usr/bin/env python3
"""
Validate and merge gold extension part CSVs into one project-ready dataset.

The merged output uses the canonical column order expected by the project:
- base tweet metadata
- need labels
- aciliyet_0_3
- veracity_label
- notes
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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

PART_RE = re.compile(r"part_(\d+)\.csv$", flags=re.IGNORECASE)


def _normalize_binary(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip().str.lower()
    mapping = {
        "0": "0",
        "1": "1",
        "false": "0",
        "true": "1",
        "f": "0",
        "t": "1",
        "no": "0",
        "yes": "1",
        "n": "0",
        "y": "1",
        "hayir": "0",
        "hayır": "0",
        "evet": "1",
    }
    out = s.map(mapping)
    invalid = s.eq("") | out.isna()
    if invalid.any():
        bad_values = sorted(set(s[invalid].tolist()))
        raise SystemExit(f"Invalid values in {col}: {bad_values}")
    df[col] = out


def _normalize_aciliyet(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip()
    out = pd.to_numeric(s, errors="coerce")
    invalid = s.eq("") | out.isna() | ~out.isin([0, 1, 2, 3])
    if invalid.any():
        bad_values = sorted(set(s[invalid].tolist()))
        raise SystemExit(f"Invalid values in {col}: {bad_values}")
    df[col] = out.astype(int).astype(str)


def _normalize_veracity(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip().str.lower()
    mapping = {
        "dogrulanmis": "dogrulanmis",
        "doğrulanmış": "dogrulanmis",
        "dogrulanmış": "dogrulanmis",
        "doğrulanmis": "dogrulanmis",
        "doğrulandı": "dogrulanmis",
        "dogrulandi": "dogrulanmis",
        "verified": "dogrulanmis",
        "confirmed": "dogrulanmis",
        "supheli": "supheli",
        "şüpheli": "supheli",
        "doğrulanmamış": "supheli",
        "dogrulanmamis": "supheli",
        "suspicious": "supheli",
        "asilsiz": "asilsiz",
        "asılsız": "asilsiz",
        "false": "asilsiz",
        "hoax": "asilsiz",
        "spam": "asilsiz",
    }
    out = s.map(mapping)
    invalid = s.eq("") | out.isna()
    if invalid.any():
        bad_values = sorted(set(s[invalid].tolist()))
        raise SystemExit(f"Invalid values in {col}: {bad_values}")
    df[col] = out


def _normalize_base_columns(df: pd.DataFrame) -> None:
    for col in BASE_COLS:
        df[col] = df[col].astype("string").fillna("")
    df["id"] = df["id"].str.strip()
    if df["id"].eq("").any():
        raise SystemExit("Found blank id values.")


def _normalize_notes(df: pd.DataFrame) -> None:
    df[NOTES_COL] = df[NOTES_COL].astype("string").fillna("")


def _load_part(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing required columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()
    _normalize_base_columns(df)
    for col in NEED_LABEL_COLS:
        _normalize_binary(df, col)
    _normalize_aciliyet(df, ACILIYET_COL)
    _normalize_veracity(df, VERACITY_COL)
    _normalize_notes(df)

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()
        raise SystemExit(f"{path.name} has duplicate ids: {dupes[:10]}")

    return df


def _part_sort_key(path: Path) -> Tuple[int, str]:
    match = PART_RE.search(path.name)
    if not match:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def _find_part_files(input_dir: Path, pattern: str) -> List[Path]:
    paths = [p for p in input_dir.glob(pattern) if p.is_file()]
    if not paths:
        raise SystemExit(f"No files matched pattern {pattern!r} under {input_dir}")
    return sorted(paths, key=_part_sort_key)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate and merge gold extension parts.")
    p.add_argument(
        "--input-dir",
        default="gold_set",
        help="Directory containing part CSV files.",
    )
    p.add_argument(
        "--pattern",
        default="need_classification_gold_extension_800_blank_part_*.csv",
        help="Glob pattern for part files.",
    )
    p.add_argument(
        "--output",
        default="data/need_classification_gold_extension_800.csv",
        help="Merged project-ready CSV output path.",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = _find_part_files(input_dir, args.pattern)
    frames: List[pd.DataFrame] = []
    per_part_rows: Dict[str, int] = {}

    for path in parts:
        df = _load_part(path)
        frames.append(df)
        per_part_rows[path.name] = len(df)

    merged = pd.concat(frames, ignore_index=True)
    if merged["id"].duplicated().any():
        dupes = merged.loc[merged["id"].duplicated(), "id"].tolist()
        raise SystemExit(f"Duplicate ids across parts: {dupes[:20]}")

    merged = merged[EXPECTED_COLUMNS].copy()
    merged.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Merged {len(parts)} files into: {output}")
    print(f"Total rows: {len(merged)}")
    for name, row_count in per_part_rows.items():
        print(f"- {name}: {row_count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
