#!/usr/bin/env python3
"""
Build a pseudo-labeled ("silver") multi-label need-classification dataset from the
geolocated emergency tweets CSV.

Why:
- The 1,000-tweet sample is great for human labeling, but too small (and too
  imbalanced) for a robust baseline model.
- This script deduplicates by tweet `id` and applies the rule-based prefill
  (scripts/ai_prefill_annotations.py) to create a large pseudo-labeled dataset.

Output columns are compatible with scripts/prepare_model_splits.py and
scripts/train_need_classifier.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _len_stats(s: pd.Series) -> Dict[str, float]:
    lens = s.astype("string").fillna("").str.len()
    if lens.empty:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(lens.min()), "max": float(lens.max()), "mean": float(lens.mean())}


def main() -> int:
    p = argparse.ArgumentParser(description="Build silver pseudo-labeled need dataset from geolocated emergency CSV.")
    p.add_argument(
        "--input",
        default="data/processed/emergency_geolocated_96k.csv",
        help="Input CSV path (geolocated emergency dataset).",
    )
    p.add_argument(
        "--output",
        default="data/labeling/need_classification_silver_63k_profileA.csv",
        help="Output CSV path.",
    )
    p.add_argument("--profile", choices=["A", "B"], default="A", help="Prefill profile (A=more recall, B=more conservative).")
    p.add_argument("--dedup", action="store_true", help="Deduplicate by tweet id (recommended).")
    p.add_argument("--min-text-len", type=int, default=0, help="Optional minimum length for tweet_clean (default: 0).")
    p.add_argument(
        "--exclude-gold-csv",
        action="append",
        default=[],
        help=(
            "Path to a gold CSV whose `id` column should be excluded from the silver output. "
            "Can be passed multiple times (e.g. once for the canonical gold, or once per split)."
        ),
    )
    args = p.parse_args()

    # Local import (same folder) so this script stays copy-paste friendly in Colab.
    import ai_prefill_annotations as prefill  # type: ignore

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Keep only needed columns to reduce memory / output size.
    usecols: List[str] = [
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

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype=str, usecols=lambda c: c in usecols)
    df = df.copy()

    # Ensure all expected cols exist (some datasets may omit district/province).
    for c in usecols:
        if c not in df.columns:
            df[c] = ""

    before = len(df)
    if args.dedup:
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    # Optional min length filter
    if args.min_text_len and args.min_text_len > 0:
        s = df["tweet_clean"].astype("string").fillna("")
        df = df[s.str.len() >= int(args.min_text_len)].reset_index(drop=True)

    # Exclude gold ids from the silver output to prevent test leakage.
    excluded_count = 0
    excluded_sources: List[Dict[str, object]] = []
    if args.exclude_gold_csv:
        gold_ids_all: set = set()
        for gcsv in args.exclude_gold_csv:
            gpath = Path(gcsv)
            g = pd.read_csv(gpath, encoding="utf-8-sig", dtype=str, usecols=["id"])
            ids = set(g["id"].astype(str).tolist())
            gold_ids_all |= ids
            excluded_sources.append({
                "path": str(gpath.as_posix()),
                "rows": int(len(g)),
                "unique_ids": int(len(ids)),
            })
        if gold_ids_all:
            mask = df["id"].astype(str).isin(gold_ids_all)
            excluded_count = int(mask.sum())
            df = df.loc[~mask].reset_index(drop=True)

    after = len(df)

    compiled = prefill.compile_patterns(args.profile)
    texts = df["tweet_clean"].astype("string").fillna(df["tweet"].astype("string").fillna("")).fillna("").tolist()

    need_cols = list(prefill.NEED_COLS)
    out_cols = [
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
        *need_cols,
        "aciliyet_0_3",
        "veracity_label",
        "notes",
    ]
    out_df = df[[c for c in out_cols if c in df.columns]].copy()
    for c in out_cols:
        if c not in out_df.columns:
            out_df[c] = ""

    # Label row-by-row (63k is small enough for a single pass).
    need_lists: Dict[str, List[int]] = {c: [] for c in need_cols}
    ac_list: List[int] = []
    ver_list: List[str] = []
    notes_list: List[str] = []

    for t in texts:
        lab = prefill.label_text(t, compiled)
        for c in need_cols:
            need_lists[c].append(int(lab.get(c, "0")))
        ac_list.append(int(lab.get("aciliyet_0_3", "0")))
        ver_list.append(str(lab.get("veracity_label", "supheli")))
        notes_list.append(str(lab.get("notes", "")))

    for c in need_cols:
        out_df[c] = need_lists[c]
    out_df["aciliyet_0_3"] = ac_list
    out_df["veracity_label"] = ver_list
    out_df["notes"] = notes_list

    # Coerce numerics where possible (use 0 for missing/invalid).
    out_df["urgency_score"] = pd.to_numeric(out_df["urgency_score"], errors="coerce").fillna(0).astype(int)

    out_df.to_csv(out, index=False, encoding="utf-8-sig")

    # Write stats for quick sanity checks.
    stats_path = out.parent / f"{out.stem}_stats.txt"
    counts = {c: int(out_df[c].sum()) for c in need_cols}
    ac_dist = {str(k): int(v) for k, v in out_df["aciliyet_0_3"].value_counts().sort_index().to_dict().items()}
    ver_dist = {str(k): int(v) for k, v in out_df["veracity_label"].astype("string").fillna("supheli").value_counts().to_dict().items()}
    len_stats = _len_stats(out_df["tweet_clean"])

    lines: List[str] = []
    lines.append("SILVER DATASET BUILD REPORT\n")
    lines.append(f"Input: {inp.as_posix()}")
    lines.append(f"Output: {out.as_posix()}")
    lines.append(f"Profile: {args.profile}")
    lines.append(f"Dedup: {bool(args.dedup)} | rows before={before} after={after}")
    lines.append(f"Min tweet_clean length: {args.min_text_len}")
    if args.exclude_gold_csv:
        lines.append(
            f"Excluded gold ids: {excluded_count} rows removed (sources: {len(args.exclude_gold_csv)})"
        )
        for src in excluded_sources:
            lines.append(f"- gold_source: {src['path']} rows={src['rows']} unique_ids={src['unique_ids']}")
    else:
        lines.append("Excluded gold ids: NONE (no --exclude-gold-csv passed)")
    lines.append("")
    lines.append("LABEL POSITIVES (sum)")
    lines.append(json.dumps(counts, ensure_ascii=True, indent=2))
    lines.append("")
    lines.append("ACILIYET_0_3 DISTRIBUTION")
    lines.append(json.dumps(ac_dist, ensure_ascii=True, indent=2))
    lines.append("")
    lines.append("VERACITY DISTRIBUTION")
    lines.append(json.dumps(ver_dist, ensure_ascii=True, indent=2))
    lines.append("")
    lines.append("TWEET_CLEAN LENGTH STATS")
    lines.append(json.dumps(len_stats, ensure_ascii=True, indent=2))
    lines.append("")
    stats_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    stats_json_path = out.parent / f"{out.stem}_stats.json"
    stats = {
        "input": str(inp.as_posix()),
        "output": str(out.as_posix()),
        "profile": args.profile,
        "dedup": bool(args.dedup),
        "rows_before": int(before),
        "rows_after": int(after),
        "min_text_len": int(args.min_text_len),
        "gold_exclusion": {
            "enabled": bool(args.exclude_gold_csv),
            "sources": excluded_sources,
            "rows_removed": int(excluded_count),
        },
        "label_positives": counts,
        "aciliyet_dist": ac_dist,
        "veracity_dist": ver_dist,
        "tweet_clean_len": len_stats,
        "label_columns": need_cols,
    }
    stats_json_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {out}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {stats_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

