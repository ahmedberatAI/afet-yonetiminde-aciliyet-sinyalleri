#!/usr/bin/env python3
"""
Compute inter-annotator agreement (IAA) and generate an adjudication file.

Inputs:
- Annotator A CSV
- Annotator B CSV

Outputs (under --outdir):
- iaa_report.txt
- adjudication.csv              (all rows; final columns auto-filled where A==B)
- disagreements_only.csv        (rows that still need adjudication)

Notes:
- For binary need labels, blank cells are treated as 0 by default (configurable).
- For aciliyet_0_3 and veracity_label, blanks are treated as missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


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


def _as_string(s: pd.Series) -> pd.Series:
    return s.astype("string")


def normalize_binary(s: pd.Series, *, blank_as_zero: bool) -> pd.Series:
    s = _as_string(s).str.strip()
    lower = s.str.lower()

    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    ones = {"1", "true", "t", "yes", "y", "evet"}
    zeros = {"0", "false", "f", "no", "n", "hayir", "hayır"}

    out[lower.isin(list(ones))] = 1
    out[lower.isin(list(zeros))] = 0

    if blank_as_zero:
        out[lower.isna() | (lower == "")] = 0

    return out


def normalize_aciliyet(s: pd.Series) -> pd.Series:
    s = _as_string(s).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    num = num.where(num.isin([0, 1, 2, 3]))
    return num.astype("Int64")


def normalize_veracity(s: pd.Series) -> pd.Series:
    s = _as_string(s).str.strip()
    lower = s.str.lower()

    # Map common spellings/synonyms to canonical labels (ASCII).
    mapping = {
        "dogrulanmis": "dogrulanmis",
        "doğrulanmış": "dogrulanmis",
        "dogrulanmış": "dogrulanmis",
        "doğrulanmis": "dogrulanmis",
        "verified": "dogrulanmis",
        "confirmed": "dogrulanmis",
        "supheli": "supheli",
        "şüpheli": "supheli",
        "suspicious": "supheli",
        "asilsiz": "asilsiz",
        "asılsız": "asilsiz",
        "false": "asilsiz",
        "hoax": "asilsiz",
        "spam": "asilsiz",
    }
    out = lower.map(mapping)
    out = out.where(~(lower.isna() | (lower == "")), pd.NA)
    # Keep unknown non-empty values as-is (so user sees unexpected values).
    out = out.fillna(lower.where(~(lower.isna() | (lower == "")), pd.NA))
    return out.astype("string")


def compute_kappa(a: pd.Series, b: pd.Series, *, labels=None, weights=None) -> Tuple[float, int, float]:
    mask = (~a.isna()) & (~b.isna())
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0, float("nan")
    aa = a[mask]
    bb = b[mask]
    # sklearn can emit RuntimeWarning for degenerate distributions (e.g., all zeros).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        k = float(cohen_kappa_score(aa, bb, labels=labels, weights=weights))
    agr = float((aa == bb).mean())
    return k, n, agr


def tweet_preview(s: str, limit: int = 220) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def main() -> int:
    p = argparse.ArgumentParser(description="Compute IAA (Cohen's kappa) and build adjudication CSV.")
    p.add_argument("--a", required=True, help="Annotator A CSV")
    p.add_argument("--b", required=True, help="Annotator B CSV")
    p.add_argument("--outdir", default="data/labeling/iaa", help="Output directory")
    p.add_argument(
        "--blank-as-missing",
        action="store_true",
        help="Treat blank cells in binary need labels as missing (default: treat blanks as 0).",
    )
    p.add_argument(
        "--low-kappa-threshold",
        type=float,
        default=0.60,
        help="Threshold for flagging labels for calibration review.",
    )
    args = p.parse_args()

    blank_as_zero = not args.blank_as_missing
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(Path(args.a), encoding="utf-8-sig", dtype="string")
    df_b = pd.read_csv(Path(args.b), encoding="utf-8-sig", dtype="string")

    for name, df in [("A", df_a), ("B", df_b)]:
        if "id" not in df.columns:
            raise SystemExit(f"Missing 'id' column in annotator {name} file.")
        df["id"] = df["id"].astype("string").str.strip()
        if df["id"].isna().any():
            raise SystemExit(f"Annotator {name} file has missing tweet ids.")
        if df["id"].duplicated().any():
            dup_n = int(df["id"].duplicated().sum())
            raise SystemExit(f"Annotator {name} file has duplicate ids: {dup_n}")

    merged = df_a.merge(df_b, on="id", how="outer", suffixes=("_a", "_b"), indicator=True)
    only_a = merged[merged["_merge"] == "left_only"]["id"].tolist()
    only_b = merged[merged["_merge"] == "right_only"]["id"].tolist()
    both = merged[merged["_merge"] == "both"].copy()

    # Use A's base columns as canonical for outputs.
    base_out = {}
    for col in BASE_COLS:
        if col == "id":
            base_out[col] = both["id"]
            continue
        ca = f"{col}_a"
        cb = f"{col}_b"
        if ca in both.columns:
            base_out[col] = both[ca]
        elif cb in both.columns:
            base_out[col] = both[cb]
        else:
            base_out[col] = pd.NA

    # Parse labels.
    parsed: Dict[str, Dict[str, pd.Series]] = {}
    raw_blank_counts: Dict[str, Dict[str, int]] = {}

    for col in NEED_LABEL_COLS:
        a_raw = both.get(f"{col}_a", pd.Series(pd.NA, index=both.index, dtype="string"))
        b_raw = both.get(f"{col}_b", pd.Series(pd.NA, index=both.index, dtype="string"))
        raw_blank_counts[col] = {
            "blank_a": int(_as_string(a_raw).str.strip().fillna("").eq("").sum()),
            "blank_b": int(_as_string(b_raw).str.strip().fillna("").eq("").sum()),
        }
        parsed[col] = {
            "a": normalize_binary(a_raw, blank_as_zero=blank_as_zero),
            "b": normalize_binary(b_raw, blank_as_zero=blank_as_zero),
        }

    a_ac = both.get(f"{ACILIYET_COL}_a", pd.Series(pd.NA, index=both.index, dtype="string"))
    b_ac = both.get(f"{ACILIYET_COL}_b", pd.Series(pd.NA, index=both.index, dtype="string"))
    parsed[ACILIYET_COL] = {"a": normalize_aciliyet(a_ac), "b": normalize_aciliyet(b_ac)}

    a_ver = both.get(f"{VERACITY_COL}_a", pd.Series(pd.NA, index=both.index, dtype="string"))
    b_ver = both.get(f"{VERACITY_COL}_b", pd.Series(pd.NA, index=both.index, dtype="string"))
    parsed[VERACITY_COL] = {"a": normalize_veracity(a_ver), "b": normalize_veracity(b_ver)}

    a_notes = both.get(f"{NOTES_COL}_a", pd.Series(pd.NA, index=both.index, dtype="string"))
    b_notes = both.get(f"{NOTES_COL}_b", pd.Series(pd.NA, index=both.index, dtype="string"))

    # Compute metrics.
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines: List[str] = []
    report_lines.append("IAA REPORT (Cohen's kappa)\n")
    report_lines.append(f"Generated: {now}")
    report_lines.append(f"Annotator A: {Path(args.a).as_posix()}")
    report_lines.append(f"Annotator B: {Path(args.b).as_posix()}")
    report_lines.append("")
    report_lines.append("JOIN CHECKS")
    report_lines.append(f"- ids in A: {len(df_a):,}")
    report_lines.append(f"- ids in B: {len(df_b):,}")
    report_lines.append(f"- ids in both: {len(both):,}")
    report_lines.append(f"- only in A: {len(only_a):,}")
    report_lines.append(f"- only in B: {len(only_b):,}")
    if only_a[:5]:
        report_lines.append(f"- sample only-in-A ids: {only_a[:5]}")
    if only_b[:5]:
        report_lines.append(f"- sample only-in-B ids: {only_b[:5]}")
    report_lines.append("")
    report_lines.append(f"BINARY LABEL BLANK HANDLING: blanks -> {'0' if blank_as_zero else 'missing'}")
    report_lines.append("")

    low_kappa: List[Tuple[str, float]] = []
    kappas_binary: List[float] = []

    report_lines.append("NEED LABELS (binary, per-label kappa)")
    for col in NEED_LABEL_COLS:
        a = parsed[col]["a"]
        b = parsed[col]["b"]
        k, n, agr = compute_kappa(a, b, labels=[0, 1], weights=None)
        if not np.isnan(k):
            kappas_binary.append(k)
            if k < args.low_kappa_threshold:
                low_kappa.append((col, k))

        pos_a = float(a.dropna().mean()) if a.dropna().size else float("nan")
        pos_b = float(b.dropna().mean()) if b.dropna().size else float("nan")
        miss_a = int(a.isna().sum())
        miss_b = int(b.isna().sum())

        blanks = raw_blank_counts.get(col, {"blank_a": 0, "blank_b": 0})

        report_lines.append(
            f"- {col}: kappa={k:.3f} n={n} agreement={agr:.3f} "
            f"pos_rate(A)={pos_a:.3f} pos_rate(B)={pos_b:.3f} "
            f"missing(A)={miss_a} missing(B)={miss_b} blanks_raw(A)={blanks['blank_a']} blanks_raw(B)={blanks['blank_b']}"
        )

    if kappas_binary:
        report_lines.append("")
        report_lines.append(f"Binary label macro kappa (mean): {float(np.mean(kappas_binary)):.3f}")

    # aciliyet
    report_lines.append("")
    report_lines.append("AUX LABELS")
    a = parsed[ACILIYET_COL]["a"]
    b = parsed[ACILIYET_COL]["b"]
    k_unw, n_unw, agr_unw = compute_kappa(a, b, labels=[0, 1, 2, 3], weights=None)
    k_w, n_w, agr_w = compute_kappa(a, b, labels=[0, 1, 2, 3], weights="quadratic")
    report_lines.append(
        f"- {ACILIYET_COL}: kappa(unweighted)={k_unw:.3f} kappa(weighted_quadratic)={k_w:.3f} "
        f"n={n_unw} agreement={agr_unw:.3f} missing(A)={int(a.isna().sum())} missing(B)={int(b.isna().sum())}"
    )

    a = parsed[VERACITY_COL]["a"]
    b = parsed[VERACITY_COL]["b"]
    # cohen_kappa_score works on strings as well.
    k, n, agr = compute_kappa(a, b, labels=None, weights=None)
    report_lines.append(
        f"- {VERACITY_COL}: kappa={k:.3f} n={n} agreement={agr:.3f} missing(A)={int(a.isna().sum())} missing(B)={int(b.isna().sum())}"
    )

    report_lines.append("")
    if low_kappa:
        low_kappa_sorted = sorted(low_kappa, key=lambda x: x[1])
        report_lines.append(f"LABELS BELOW THRESHOLD (< {args.low_kappa_threshold:.2f})")
        for col, k in low_kappa_sorted:
            report_lines.append(f"- {col}: {k:.3f}")
    else:
        report_lines.append(f"No binary labels below threshold (< {args.low_kappa_threshold:.2f}).")

    # Build adjudication dataframe.
    adjudication = pd.DataFrame(base_out)

    for col in NEED_LABEL_COLS:
        aa = parsed[col]["a"]
        bb = parsed[col]["b"]
        final = aa.where(aa == bb, pd.NA)
        adjudication[f"{col}_a"] = aa.astype("Int64")
        adjudication[f"{col}_b"] = bb.astype("Int64")
        adjudication[f"{col}_final"] = final.astype("Int64")

    # aciliyet/veracity
    aa = parsed[ACILIYET_COL]["a"]
    bb = parsed[ACILIYET_COL]["b"]
    adjudication[f"{ACILIYET_COL}_a"] = aa
    adjudication[f"{ACILIYET_COL}_b"] = bb
    adjudication[f"{ACILIYET_COL}_final"] = aa.where(aa == bb, pd.NA)

    aa = parsed[VERACITY_COL]["a"]
    bb = parsed[VERACITY_COL]["b"]
    adjudication[f"{VERACITY_COL}_a"] = aa
    adjudication[f"{VERACITY_COL}_b"] = bb
    adjudication[f"{VERACITY_COL}_final"] = aa.where(aa == bb, pd.NA)

    adjudication[f"{NOTES_COL}_a"] = _as_string(a_notes)
    adjudication[f"{NOTES_COL}_b"] = _as_string(b_notes)
    adjudication[f"{NOTES_COL}_final"] = ""

    # Rows needing adjudication (any label final missing where at least one annotator provided a value).
    final_cols = [f"{c}_final" for c in NEED_LABEL_COLS] + [f"{ACILIYET_COL}_final", f"{VERACITY_COL}_final"]
    needs_adjudication = pd.Series(False, index=adjudication.index)

    for col in NEED_LABEL_COLS:
        aa = adjudication[f"{col}_a"]
        bb = adjudication[f"{col}_b"]
        ff = adjudication[f"{col}_final"]
        # Disagreement if both present and different.
        needs_adjudication |= (~aa.isna()) & (~bb.isna()) & (ff.isna()) & (aa != bb)

    aa = adjudication[f"{ACILIYET_COL}_a"]
    bb = adjudication[f"{ACILIYET_COL}_b"]
    ff = adjudication[f"{ACILIYET_COL}_final"]
    needs_adjudication |= (~aa.isna()) & (~bb.isna()) & (ff.isna()) & (aa != bb)

    aa = adjudication[f"{VERACITY_COL}_a"]
    bb = adjudication[f"{VERACITY_COL}_b"]
    ff = adjudication[f"{VERACITY_COL}_final"]
    needs_adjudication |= (~aa.isna()) & (~bb.isna()) & (ff.isna()) & (aa != bb)

    # Add a helper flag for easy filtering.
    adjudication["needs_adjudication"] = needs_adjudication.astype(bool)

    # Add some calibration helpers: show a few example disagreements for low-kappa labels.
    if low_kappa:
        report_lines.append("")
        report_lines.append("DISAGREEMENT EXAMPLES (for calibration)")
        # Attach tweet text for easy reading.
        tweets = _as_string(adjudication["tweet"]).fillna("")
        for col, k in sorted(low_kappa, key=lambda x: x[1])[:5]:
            report_lines.append(f"- {col} (kappa={k:.3f}):")
            a_col = adjudication[f"{col}_a"]
            b_col = adjudication[f"{col}_b"]
            mask = (~a_col.isna()) & (~b_col.isna()) & (a_col != b_col)
            ex = adjudication[mask].head(10)
            if ex.empty:
                report_lines.append("  (no disagreements found after normalization)")
                continue
            for _, row in ex.iterrows():
                report_lines.append(
                    f"  id={row['id']} A={row[f'{col}_a']} B={row[f'{col}_b']} | {tweet_preview(str(row['tweet']))}"
                )

    report_path = outdir / "iaa_report.txt"
    adjudication_path = outdir / "adjudication.csv"
    disagreements_path = outdir / "disagreements_only.csv"

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")
    adjudication.to_csv(adjudication_path, index=False, encoding="utf-8-sig")
    adjudication[adjudication["needs_adjudication"]].to_csv(disagreements_path, index=False, encoding="utf-8-sig")

    print(f"Wrote: {report_path}")
    print(f"Wrote: {adjudication_path}")
    print(f"Wrote: {disagreements_path}")
    print("")
    print("Adjudication workflow:")
    print(f"1) Open {adjudication_path} and fill *_final columns where needs_adjudication=True")
    print("2) After adjudication, export gold CSV:")
    print(
        "   python scripts/export_gold_from_adjudication.py "
        f"--input \"{adjudication_path.as_posix()}\" "
        "--output \"data/labeling/need_classification_gold.csv\" "
        "--require-complete"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
