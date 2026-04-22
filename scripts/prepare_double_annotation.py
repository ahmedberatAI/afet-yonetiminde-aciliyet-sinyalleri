#!/usr/bin/env python3
"""
Prepare 2 independent annotator files from the master labeling template.

Typical workflow:
1) Generate the base sample:
   python scripts/create_need_classification_sample.py
2) Create annotator files:
   python scripts/prepare_double_annotation.py
3) Send annotator files to 2 annotators (A/B). After labeling, compute IAA:
   python scripts/compute_iaa_and_adjudication.py --a ... --b ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


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


def main() -> int:
    p = argparse.ArgumentParser(description="Create annotator A/B CSV files from a labeling template.")
    p.add_argument(
        "--input",
        default="data/labeling/need_classification_sample_1000.csv",
        help="Master labeling CSV (template).",
    )
    p.add_argument(
        "--outdir",
        default="data/labeling/annotations",
        help="Output directory for annotator files.",
    )
    p.add_argument(
        "--prefix",
        default="need_classification_sample_1000",
        help="Output filename prefix.",
    )
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype={"id": "string"})

    # Ensure label columns exist, then blank them out for clean templates.
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = ""

    out_a = outdir / f"{args.prefix}_annotator_A.csv"
    out_b = outdir / f"{args.prefix}_annotator_B.csv"
    df.to_csv(out_a, index=False, encoding="utf-8-sig")
    df.to_csv(out_b, index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_a}")
    print(f"Wrote: {out_b}")
    print("")
    print("Next:")
    print(f"1) Annotator A fills: {out_a}")
    print(f"2) Annotator B fills: {out_b}")
    print("3) Compute IAA + adjudication file:")
    print(
        "   python scripts/compute_iaa_and_adjudication.py "
        f"--a \"{out_a.as_posix()}\" --b \"{out_b.as_posix()}\""
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

