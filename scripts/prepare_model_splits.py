#!/usr/bin/env python3
"""
Prepare train/val/test splits from a gold-labeled CSV.

This project is multi-label (needs can co-occur). "True" multi-label
stratification is non-trivial without extra dependencies, so by default we
stratify on `aciliyet_0_3` to keep overall urgency balanced across splits.

Outputs (under --outdir):
- train.csv / val.csv / test.csv
- split_report.txt (label distributions + basic stats)
- split_manifest.json (machine-readable paths + settings)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def _coerce_binary_0_1(df: pd.DataFrame, col: str) -> None:
    # Accept 0/1 as ints or strings; treat blanks/NA as 0.
    s = df[col].astype("string").fillna("").str.strip()
    s = s.replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    bad = df[~df[col].isin([0, 1])]
    if not bad.empty:
        raise SystemExit(f"Invalid values in {col}: expected 0/1 only.")


def _coerce_aciliyet(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip()
    s = s.replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    bad = df[~df[col].isin([0, 1, 2, 3])]
    if not bad.empty:
        raise SystemExit(f"Invalid values in {col}: expected 0/1/2/3 only.")


def _tweet_len_stats(df: pd.DataFrame, text_col: str) -> Dict[str, float]:
    s = df[text_col].astype("string").fillna("")
    lens = s.str.len()
    return {
        "min": float(lens.min()) if len(lens) else 0.0,
        "max": float(lens.max()) if len(lens) else 0.0,
        "mean": float(lens.mean()) if len(lens) else 0.0,
    }


def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {c: int(df[c].sum()) for c in NEED_LABEL_COLS}


def _aciliyet_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df[ACILIYET_COL].value_counts().sort_index().to_dict().items()}


def _veracity_counts(df: pd.DataFrame) -> Dict[str, int]:
    s = df[VERACITY_COL].astype("string").fillna("").str.strip()
    s = s.replace({"": "supheli"})  # default if empty
    return {str(k): int(v) for k, v in s.value_counts().to_dict().items()}


def _rare_labels(counts: Dict[str, int], n_rows: int, threshold: int = 10) -> List[str]:
    out: List[str] = []
    for k, v in counts.items():
        if v == 0:
            out.append(f"{k}:0 (NO_POSITIVES)")
        elif v < threshold:
            out.append(f"{k}:{v} (<{threshold})")
    return out


def _stratified_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    stratify_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise SystemExit("train/val/test sizes must sum to 1.0")

    strat = None
    strat_note = "none"
    if stratify_col:
        if stratify_col not in df.columns:
            raise SystemExit(f"Stratify column not found: {stratify_col}")
        strat = df[stratify_col]
        strat_note = stratify_col

    tmp_size = val_size + test_size
    try:
        train_df, tmp_df = train_test_split(
            df,
            test_size=tmp_size,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
    except ValueError as e:
        # Fall back to random split (still deterministic) if stratification fails.
        train_df, tmp_df = train_test_split(
            df,
            test_size=tmp_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        strat_note = f"{strat_note} (FAILED -> random; {e})"

    rel_test = test_size / tmp_size if tmp_size > 0 else 0.0
    strat2 = None
    if stratify_col and stratify_col in tmp_df.columns:
        strat2 = tmp_df[stratify_col]

    try:
        val_df, test_df = train_test_split(
            tmp_df,
            test_size=rel_test,
            random_state=seed + 1,
            shuffle=True,
            stratify=strat2,
        )
    except ValueError as e:
        val_df, test_df = train_test_split(
            tmp_df,
            test_size=rel_test,
            random_state=seed + 1,
            shuffle=True,
            stratify=None,
        )
        strat_note = f"{strat_note} | second split FAILED -> random; {e}"

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), strat_note


def _swap_rows(df_a: pd.DataFrame, idx_a: int, df_b: pd.DataFrame, idx_b: int) -> None:
    # Swap by position; assumes both DataFrames share the same columns.
    row_a = df_a.iloc[idx_a].copy()
    row_b = df_b.iloc[idx_b].copy()
    df_a.iloc[idx_a] = row_b
    df_b.iloc[idx_b] = row_a


def _ensure_test_label_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    stratify_col: Optional[str],
) -> List[str]:
    """
    Post-process splits to ensure the test set contains at least 1 positive
    example for each label that has >=2 positives in the full dataset.

    To keep stratification stable, we only swap within the same `stratify_col`
    value (default: aciliyet_0_3). If stratify_col is None, we swap without
    this constraint.
    """
    rng = np.random.RandomState(seed)
    notes: List[str] = []

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total_pos = {c: int(full_df[c].sum()) for c in NEED_LABEL_COLS}

    for lab in NEED_LABEL_COLS:
        if total_pos.get(lab, 0) < 2:
            continue
        if int(test_df[lab].sum()) > 0:
            continue

        # Choose donor split (prefer train but keep >=1 positive there).
        donors: List[Tuple[str, pd.DataFrame]] = []
        if int(train_df[lab].sum()) > 1:
            donors.append(("train", train_df))
        if int(val_df[lab].sum()) > 0:
            donors.append(("val", val_df))

        if not donors:
            notes.append(f"{lab}: could not ensure test coverage (no safe donor)")
            continue

        swapped = False
        for donor_name, donor_df in donors:
            cand_idx = donor_df.index[donor_df[lab] == 1].tolist()
            rng.shuffle(cand_idx)
            for di in cand_idx:
                if stratify_col:
                    donor_stratum = donor_df.loc[di, stratify_col]
                    test_candidates = test_df.index[
                        (test_df[stratify_col] == donor_stratum) & (test_df[lab] == 0)
                    ].tolist()
                else:
                    test_candidates = test_df.index[(test_df[lab] == 0)].tolist()

                if not test_candidates:
                    continue

                ti = int(rng.choice(test_candidates))
                _swap_rows(donor_df, int(di), test_df, int(ti))
                notes.append(f"{lab}: swapped 1 positive into test (donor={donor_name})")
                swapped = True
                break
            if swapped:
                break

        if not swapped:
            notes.append(f"{lab}: could not ensure test coverage (no compatible stratum swap)")

    return notes


def main() -> int:
    p = argparse.ArgumentParser(description="Create train/val/test splits from gold CSV (stratified).")
    p.add_argument("--input", default="data/labeling/need_classification_gold.csv", help="Gold CSV path.")
    p.add_argument("--outdir", default="data/modeling/need_classification", help="Output directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--train-size", type=float, default=0.80, help="Train fraction.")
    p.add_argument("--val-size", type=float, default=0.10, help="Validation fraction.")
    p.add_argument("--test-size", type=float, default=0.10, help="Test fraction.")
    p.add_argument(
        "--stratify-col",
        default=ACILIYET_COL,
        help=f"Column to stratify by (default: {ACILIYET_COL}). Use empty string to disable.",
    )
    p.add_argument(
        "--ensure-test-coverage",
        action="store_true",
        help="After splitting, swap rows (within stratify strata) to ensure each label with >=2 positives appears in test at least once.",
    )
    p.add_argument("--text-col", default="tweet_clean", help="Text column for length stats.")
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig")

    required_cols = ["id", "tweet_clean"] + NEED_LABEL_COLS + [ACILIYET_COL, VERACITY_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in gold CSV: {missing}")

    # Coerce labels to safe numeric ranges for later training.
    for c in NEED_LABEL_COLS:
        _coerce_binary_0_1(df, c)
    _coerce_aciliyet(df, ACILIYET_COL)
    df[VERACITY_COL] = df[VERACITY_COL].astype("string").fillna("").str.strip()

    stratify_col = args.stratify_col.strip() if isinstance(args.stratify_col, str) else None
    if stratify_col == "":
        stratify_col = None

    train_df, val_df, test_df, strat_note = _stratified_split(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=int(args.seed),
        stratify_col=stratify_col,
    )

    coverage_notes: List[str] = []
    if args.ensure_test_coverage:
        coverage_notes = _ensure_test_label_coverage(
            train_df,
            val_df,
            test_df,
            seed=int(args.seed),
            stratify_col=stratify_col,
        )

    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"
    report_path = outdir / "split_report.txt"
    manifest_path = outdir / "split_manifest.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Reports
    total_counts = _label_counts(df)
    train_counts = _label_counts(train_df)
    val_counts = _label_counts(val_df)
    test_counts = _label_counts(test_df)

    total_ac = _aciliyet_counts(df)
    train_ac = _aciliyet_counts(train_df)
    val_ac = _aciliyet_counts(val_df)
    test_ac = _aciliyet_counts(test_df)

    total_ver = _veracity_counts(df)
    train_ver = _veracity_counts(train_df)
    val_ver = _veracity_counts(val_df)
    test_ver = _veracity_counts(test_df)

    total_len = _tweet_len_stats(df, args.text_col)
    train_len = _tweet_len_stats(train_df, args.text_col)
    val_len = _tweet_len_stats(val_df, args.text_col)
    test_len = _tweet_len_stats(test_df, args.text_col)

    any_need_total = int((df[NEED_LABEL_COLS].sum(axis=1) > 0).sum())
    any_need_train = int((train_df[NEED_LABEL_COLS].sum(axis=1) > 0).sum())
    any_need_val = int((val_df[NEED_LABEL_COLS].sum(axis=1) > 0).sum())
    any_need_test = int((test_df[NEED_LABEL_COLS].sum(axis=1) > 0).sum())

    rare_total = _rare_labels(total_counts, len(df), threshold=10)

    lines: List[str] = []
    lines.append("MODEL SPLIT REPORT\n")
    lines.append(f"Input: {inp.as_posix()}")
    lines.append(f"Outdir: {outdir.as_posix()}")
    lines.append(f"Seed: {args.seed}")
    lines.append(f"Split sizes: train={args.train_size:.2f} val={args.val_size:.2f} test={args.test_size:.2f}")
    lines.append(f"Stratification: {strat_note}")
    if coverage_notes:
        lines.append("Test coverage adjustments:")
        for n in coverage_notes:
            lines.append(f"- {n}")
    lines.append("")

    lines.append("ROWS")
    lines.append(f"- total: {len(df)} (rows with any need=1: {any_need_total})")
    lines.append(f"- train: {len(train_df)} (any need=1: {any_need_train})")
    lines.append(f"- val:   {len(val_df)} (any need=1: {any_need_val})")
    lines.append(f"- test:  {len(test_df)} (any need=1: {any_need_test})")
    lines.append("")

    lines.append("ACILIYET_0_3 DISTRIBUTION")
    lines.append(f"- total: {total_ac}")
    lines.append(f"- train: {train_ac}")
    lines.append(f"- val:   {val_ac}")
    lines.append(f"- test:  {test_ac}")
    lines.append("")

    lines.append("VERACITY DISTRIBUTION (empty -> supheli)")
    lines.append(f"- total: {total_ver}")
    lines.append(f"- train: {train_ver}")
    lines.append(f"- val:   {val_ver}")
    lines.append(f"- test:  {test_ver}")
    lines.append("")

    lines.append("NEED LABEL POSITIVES (sum)")
    lines.append(f"- total: {total_counts}")
    lines.append(f"- train: {train_counts}")
    lines.append(f"- val:   {val_counts}")
    lines.append(f"- test:  {test_counts}")
    if rare_total:
        lines.append("")
        lines.append("RARE / MISSING LABELS (total set)")
        for item in rare_total:
            lines.append(f"- {item}")
    lines.append("")

    lines.append("TWEET_CLEAN LENGTH STATS")
    lines.append(f"- total: {total_len}")
    lines.append(f"- train: {train_len}")
    lines.append(f"- val:   {val_len}")
    lines.append(f"- test:  {test_len}")
    lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    manifest = {
        "input": str(inp.as_posix()),
        "outdir": str(outdir.as_posix()),
        "seed": int(args.seed),
        "train_size": float(args.train_size),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "stratify_col": stratify_col,
        "stratification_note": strat_note,
        "ensure_test_coverage": bool(args.ensure_test_coverage),
        "coverage_notes": coverage_notes,
        "paths": {
            "train": str(train_path.as_posix()),
            "val": str(val_path.as_posix()),
            "test": str(test_path.as_posix()),
            "report": str(report_path.as_posix()),
        },
        "label_columns": NEED_LABEL_COLS,
        "text_col": args.text_col,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
