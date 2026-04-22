#!/usr/bin/env python3
"""
Export a clean "gold" dataset from an adjudication CSV produced by
scripts/compute_iaa_and_adjudication.py.

This expects columns like:
- arama_kurtarma_final, ..., bilgi_paylasimi_final
- aciliyet_0_3_final
- veracity_label_final
- notes_final
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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


def main() -> int:
    p = argparse.ArgumentParser(description="Export gold labels from adjudication CSV.")
    p.add_argument("--input", required=True, help="Adjudication CSV path (adjudication.csv)")
    p.add_argument(
        "--output",
        default="data/labeling/need_classification_gold.csv",
        help="Output gold CSV path.",
    )
    p.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any *_final label is missing (recommended).",
    )
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype="string")

    final_cols: List[str] = []
    for c in NEED_LABEL_COLS:
        final_cols.append(f"{c}_final")
    final_cols += [f"{ACILIYET_COL}_final", f"{VERACITY_COL}_final", f"{NOTES_COL}_final"]

    missing = [c for c in (BASE_COLS + final_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns in adjudication file: {missing}")

    out_df = df[BASE_COLS + final_cols].copy()
    # Rename *_final columns back to canonical column names.
    rename_map = {f"{c}_final": c for c in NEED_LABEL_COLS + [ACILIYET_COL, VERACITY_COL, NOTES_COL]}
    out_df = out_df.rename(columns=rename_map)

    if args.require_complete:
        # For binary labels, treat NA as missing (annotator must resolve).
        # Notes can be empty.
        required = NEED_LABEL_COLS + [ACILIYET_COL, VERACITY_COL]
        required_df = out_df[required].astype("string")
        blank = required_df.apply(lambda s: s.fillna("").str.strip().eq(""))
        missing_any = out_df[required].isna() | blank
        rows_missing = missing_any.any(axis=1)
        n_missing = int(rows_missing.sum())
        if n_missing > 0:
            raise SystemExit(
                f"Gold export blocked: {n_missing} rows still have missing final labels. "
                "Finish adjudication (fill *_final) then re-run."
            )

    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
