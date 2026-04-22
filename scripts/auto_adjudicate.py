#!/usr/bin/env python3
"""
Auto-fill *_final columns in an adjudication.csv produced by
scripts/compute_iaa_and_adjudication.py.

Intended use:
- For AI-prefilled datasets (pseudo-labeling), where manual adjudication is not
  practical.
- For providing a starting point that humans can still review.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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

ACILIYET_COL = "aciliyet_0_3"
VERACITY_COL = "veracity_label"
NOTES_COL = "notes"


def _as_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def main() -> int:
    p = argparse.ArgumentParser(description="Auto-fill *_final columns in an adjudication CSV.")
    p.add_argument("--input", required=True, help="Input adjudication.csv path.")
    p.add_argument("--output", required=True, help="Output adjudication (auto-filled) path.")
    p.add_argument(
        "--binary-rule",
        choices=["union", "prefer_a", "prefer_b"],
        default="union",
        help="How to resolve binary label disagreements.",
    )
    p.add_argument(
        "--aciliyet-rule",
        choices=["max", "prefer_a", "prefer_b"],
        default="max",
        help="How to resolve aciliyet disagreements.",
    )
    p.add_argument(
        "--veracity-rule",
        choices=["conservative", "prefer_a", "prefer_b"],
        default="conservative",
        help="How to resolve veracity disagreements (default conservative).",
    )
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype=str)

    # Fill need labels.
    for col in NEED_LABEL_COLS:
        a = _as_int(df.get(f"{col}_a", pd.NA))
        b = _as_int(df.get(f"{col}_b", pd.NA))
        final_col = f"{col}_final"
        if final_col not in df.columns:
            df[final_col] = pd.NA

        if args.binary_rule == "prefer_a":
            final = a
        elif args.binary_rule == "prefer_b":
            final = b
        else:
            # union/max with missing treated as 0
            aa = a.fillna(0)
            bb = b.fillna(0)
            final = (aa.where(aa > bb, bb)).astype("Int64")

        df[final_col] = final.astype("Int64")

    # aciliyet
    a = _as_int(df.get(f"{ACILIYET_COL}_a", pd.NA))
    b = _as_int(df.get(f"{ACILIYET_COL}_b", pd.NA))
    fcol = f"{ACILIYET_COL}_final"
    if fcol not in df.columns:
        df[fcol] = pd.NA
    if args.aciliyet_rule == "prefer_a":
        df[fcol] = a
    elif args.aciliyet_rule == "prefer_b":
        df[fcol] = b
    else:
        aa = a.fillna(0)
        bb = b.fillna(0)
        df[fcol] = aa.where(aa >= bb, bb).astype("Int64")

    # veracity
    a = df.get(f"{VERACITY_COL}_a", pd.Series(pd.NA, index=df.index, dtype="string")).astype("string").str.strip()
    b = df.get(f"{VERACITY_COL}_b", pd.Series(pd.NA, index=df.index, dtype="string")).astype("string").str.strip()
    fcol = f"{VERACITY_COL}_final"
    if fcol not in df.columns:
        df[fcol] = pd.NA

    if args.veracity_rule == "prefer_a":
        df[fcol] = a
    elif args.veracity_rule == "prefer_b":
        df[fcol] = b
    else:
        # Conservative: only mark dogrulanmis/asilsiz if both agree, else supheli.
        same = (a.fillna("") != "") & (a == b)
        outv = pd.Series("supheli", index=df.index, dtype="string")
        outv[same] = a[same]
        # If one side is empty but the other is not, keep the non-empty value.
        only_a = (a.fillna("") != "") & (b.fillna("") == "")
        only_b = (b.fillna("") != "") & (a.fillna("") == "")
        outv[only_a] = a[only_a]
        outv[only_b] = b[only_b]
        df[fcol] = outv

    # notes_final: concat A/B notes if present, but keep short.
    notes_a = df.get(f"{NOTES_COL}_a", pd.Series("", index=df.index, dtype="string")).astype("string").fillna("").str.strip()
    notes_b = df.get(f"{NOTES_COL}_b", pd.Series("", index=df.index, dtype="string")).astype("string").fillna("").str.strip()
    ncol = f"{NOTES_COL}_final"
    if ncol not in df.columns:
        df[ncol] = ""
    combined = []
    for na, nb in zip(notes_a.tolist(), notes_b.tolist()):
        if na and nb and na != nb:
            combined.append(f"A:{na} | B:{nb}")
        else:
            combined.append(na or nb or "")
    df[ncol] = combined

    # If we filled finals, this is no longer a "needs adjudication" file.
    if "needs_adjudication" in df.columns:
        df["needs_adjudication"] = False

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

