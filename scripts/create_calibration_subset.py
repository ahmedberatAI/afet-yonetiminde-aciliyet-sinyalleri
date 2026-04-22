#!/usr/bin/env python3
"""
Create a calibration subset (e.g., first 50-100 items) from the 1,000-tweet
labeling template, and optionally generate annotator A/B files for that subset.

Goal:
- Let 2 annotators label a smaller, representative subset first.
- Run IAA (kappa), calibrate the labeling guide, then proceed with full 1,000.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
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

LABEL_COLS: List[str] = [
    "arama_kurtarma",
    "saglik",
    "barinma",
    "gida_su",
    "altyapi",
    "guvenlik",
    "lojistik",
    "psikolojik",
    "bilgi_paylasimi",
    "aciliyet_0_3",
    "veracity_label",
    "notes",
]


def tier_name_for_score(score: int) -> str:
    if 8 <= score <= 11:
        return "8-11_ultra_critical"
    if score == 7:
        return "7_high_priority"
    if 5 <= score <= 6:
        return "5-6_medium_priority"
    if 3 <= score <= 4:
        return "3-4_standard_priority"
    return "0-2_low_priority"


def proportional_allocation(sizes: pd.Series, n: int, min_per_group: int = 0) -> Dict[str, int]:
    sizes = sizes.copy()
    sizes = sizes[sizes > 0]
    if sizes.empty:
        return {}

    raw = (sizes / int(sizes.sum())) * n
    alloc = np.floor(raw).astype(int)

    if min_per_group > 0:
        alloc = np.maximum(alloc, min_per_group)
    alloc = np.minimum(alloc, sizes).astype(int)

    diff = int(n - int(alloc.sum()))
    if diff > 0:
        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        order = list(frac.index)
        i = 0
        while diff > 0 and order and i < 1_000_000:
            g = order[i % len(order)]
            if alloc[g] < sizes[g]:
                alloc[g] += 1
                diff -= 1
            i += 1
        if diff > 0:
            for g in sizes.sort_values(ascending=False).index:
                if diff <= 0:
                    break
                cap = int(sizes[g] - alloc[g])
                if cap <= 0:
                    continue
                take = min(diff, cap)
                alloc[g] += take
                diff -= take

    if diff < 0:
        for g in alloc.sort_values(ascending=False).index:
            if diff == 0:
                break
            removable = int(alloc[g])
            if removable <= 0:
                continue
            take = min(removable, -diff)
            alloc[g] -= take
            diff += take

    if int(alloc.sum()) != n:
        raise RuntimeError("Allocation failed to hit target n.")

    return {str(k): int(v) for k, v in alloc.to_dict().items()}


def blank_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in LABEL_COLS:
        if c not in out.columns:
            out[c] = ""
        out[c] = ""
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Create calibration subset + annotator files.")
    p.add_argument(
        "--input",
        default="data/labeling/need_classification_sample_1000.csv",
        help="Master template CSV (1,000 tweets).",
    )
    p.add_argument(
        "--outdir",
        default="data/labeling/calibration",
        help="Output directory for calibration pack.",
    )
    p.add_argument("--n", type=int, default=100, help="Calibration subset size (e.g., 50 or 100).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--method",
        choices=["stratified", "random", "head"],
        default="stratified",
        help="Sampling method for calibration subset.",
    )
    p.add_argument(
        "--make-annotators",
        action="store_true",
        help="Also generate annotator A/B CSV files for this calibration subset.",
    )
    args = p.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype={"id": "string"})
    if len(df) < args.n:
        raise SystemExit(f"Input has {len(df)} rows, cannot take n={args.n}.")

    # Ensure basic columns exist.
    for c in BASE_COLS:
        if c not in df.columns:
            raise SystemExit(f"Missing required column in input: {c}")

    df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0).astype(int)
    rs = np.random.RandomState(args.seed)

    if args.method == "head":
        subset = df.head(args.n).copy()
    elif args.method == "random":
        subset = df.sample(n=args.n, random_state=int(rs.randint(0, 1_000_000))).copy()
    else:
        # Stratified by urgency tier, to mimic the full sample distribution.
        tiers = df["urgency_score"].apply(tier_name_for_score)
        sizes = tiers.value_counts()
        min_per = 1 if args.n >= sizes.size else 0
        alloc = proportional_allocation(sizes, args.n, min_per_group=min_per)

        parts: List[pd.DataFrame] = []
        for tier, k in alloc.items():
            g = df[tiers == tier]
            parts.append(g.sample(n=min(k, len(g)), random_state=int(rs.randint(0, 1_000_000))))
        subset = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rs.randint(0, 1_000_000)))

    subset = blank_labels(subset)

    out_master = outdir / f"need_classification_calibration_{args.n}.csv"
    subset.to_csv(out_master, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_master}")

    if args.make_annotators:
        ann_dir = outdir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        out_a = ann_dir / f"need_classification_calibration_{args.n}_annotator_A.csv"
        out_b = ann_dir / f"need_classification_calibration_{args.n}_annotator_B.csv"
        subset.to_csv(out_a, index=False, encoding="utf-8-sig")
        subset.to_csv(out_b, index=False, encoding="utf-8-sig")
        print(f"Wrote: {out_a}")
        print(f"Wrote: {out_b}")

        print("")
        print("Next (calibration):")
        print(f"1) Annotator A fills: {out_a}")
        print(f"2) Annotator B fills: {out_b}")
        print("3) Run IAA + adjudication:")
        print(
            "   python scripts/compute_iaa_and_adjudication.py "
            f"--a \"{out_a.as_posix()}\" --b \"{out_b.as_posix()}\" "
            "--outdir \"data/labeling/calibration/iaa\""
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

