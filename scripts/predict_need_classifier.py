#!/usr/bin/env python3
"""
Run inference with a trained multi-label need classifier and write per-tweet
predictions for dashboarding.

Typical use (Colab / Drive):
  python scripts/predict_need_classifier.py \
    --model-dir models/exp3_silver_then_gold_v3_exgold/final \
    --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
    --input data/processed/emergency_geolocated_96k.csv \
    --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
    --output data/predictions/need_predictions_geolocated_v2_final.csv \
    --dedup-by-id

Notes:
- Input is expected to have at least `id` and `tweet_clean` (or `tweet`).
- This writes BOTH probabilities and 0/1 predictions by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_KEEP_COLS: List[str] = [
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _load_label_cols(model_dir: Path, labels_json: Optional[str]) -> List[str]:
    labels_path = Path(labels_json) if labels_json else (model_dir.parent / "label_columns.json")
    if not labels_path.exists():
        raise SystemExit(f"Could not find label list JSON: {labels_path}")
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("label_columns.json must be a non-empty JSON list.")
    return [str(x) for x in data]


def _load_thresholds(label_cols: List[str], threshold: float, thresholds_json: Optional[str]) -> Dict[str, float]:
    thresholds_used = {lab: float(threshold) for lab in label_cols}
    if thresholds_json:
        tpath = Path(thresholds_json)
        tdata = json.loads(tpath.read_text(encoding="utf-8"))
        if not isinstance(tdata, dict):
            raise SystemExit("--thresholds-json must be a JSON dict: label -> float")
        for lab in label_cols:
            if lab in tdata and tdata[lab] is not None:
                thresholds_used[lab] = float(tdata[lab])
    return thresholds_used


def _read_input_csv(path: Path, *, id_col: str, text_col: str, keep_cols: List[str]) -> pd.DataFrame:
    # Read header to avoid failing usecols on missing columns.
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    cols = list(header.columns)

    required = {id_col, text_col}
    # allow fallback to `tweet` if `tweet_clean` is missing
    if text_col not in cols and "tweet" in cols:
        required.discard(text_col)
        required.add("tweet")

    missing = [c for c in required if c not in cols]
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {missing}")

    want = set(keep_cols)
    want.add(id_col)
    want.add(text_col)
    want.add("tweet")  # useful fallback / dashboard
    usecols = [c for c in cols if c in want]

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=usecols)
    for c in want:
        if c not in df.columns:
            df[c] = ""
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Predict need labels with a trained HF model and export per-row predictions.")
    p.add_argument("--model-dir", required=True, help="Model directory (e.g., models/.../final).")
    p.add_argument("--input", required=True, help="Input CSV (e.g., data/processed/emergency_geolocated_96k.csv).")
    p.add_argument("--output", required=True, help="Output CSV path (predictions).")
    p.add_argument("--id-col", default="id", help="Tweet id column name.")
    p.add_argument("--text-col", default="tweet_clean", help="Text column name.")
    p.add_argument("--labels-json", default=None, help="Optional label_columns.json path.")
    p.add_argument("--threshold", type=float, default=0.5, help="Global threshold (used if --thresholds-json is not provided).")
    p.add_argument("--thresholds-json", default=None, help="Optional per-label thresholds JSON (label -> float).")
    p.add_argument("--max-length", type=int, default=192, help="Tokenizer max_length.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    p.add_argument("--prob-digits", type=int, default=4, help="Round probabilities to N decimals (default: 4).")
    p.add_argument(
        "--dedup-by-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate rows by id before inference (recommended).",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional max rows (debug).")
    p.add_argument("--prep-only", action="store_true", help="Only validate inputs and write a small metadata JSON (no inference).")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"--input not found: {inp}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    label_cols = _load_label_cols(model_dir=model_dir, labels_json=args.labels_json)
    thresholds_used = _load_thresholds(
        label_cols=label_cols,
        threshold=float(args.threshold),
        thresholds_json=args.thresholds_json,
    )

    df = _read_input_csv(inp, id_col=str(args.id_col), text_col=str(args.text_col), keep_cols=DEFAULT_KEEP_COLS)

    # Apply optional max-rows early (faster debug).
    if args.max_rows is not None and int(args.max_rows) > 0:
        df = df.head(int(args.max_rows)).copy()

    before = len(df)
    if args.dedup_by_id and args.id_col in df.columns:
        df = df.drop_duplicates(subset=[args.id_col], keep="first").reset_index(drop=True)
    after = len(df)

    # Normalize numeric columns used in UI.
    if "urgency_score" in df.columns:
        df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0).astype(int)

    # Resolve actual text source.
    text_col = args.text_col if args.text_col in df.columns else "tweet"
    texts = df[text_col].astype("string").fillna("").tolist()

    meta_path = out.with_suffix(".meta.json")
    meta = {
        "model_dir": str(model_dir.as_posix()),
        "input": str(inp.as_posix()),
        "output": str(out.as_posix()),
        "dedup_by_id": bool(args.dedup_by_id),
        "rows_before": int(before),
        "rows_after": int(after),
        "text_col_used": str(text_col),
        "labels": label_cols,
        "thresholds_per_label": thresholds_used,
        "threshold_global": float(args.threshold),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
    }

    if args.prep_only:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Prep-only: wrote {meta_path}")
        return 0

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as e:
        raise SystemExit(f"Missing dependency for prediction: {e}. Install torch + transformers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)

    bs = max(1, int(args.batch_size))
    all_logits: List[np.ndarray] = []
    for i in range(0, len(texts), bs):
        batch_texts = texts[i : i + bs]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=int(args.max_length),
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outp = model(**enc)
        all_logits.append(outp.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = _sigmoid(logits)

    thr_vec = np.array([thresholds_used[lab] for lab in label_cols], dtype=np.float32).reshape(1, -1)
    y_pred = (probs >= thr_vec).astype(np.int64)

    # Output: keep base cols + prob_* + pred_*
    out_df = df.copy()
    for j, lab in enumerate(label_cols):
        pcol = f"prob_{lab}"
        ycol = f"pred_{lab}"
        out_df[pcol] = probs[:, j].astype(np.float32)
        out_df[ycol] = y_pred[:, j].astype(int)
        if args.prob_digits is not None and int(args.prob_digits) >= 0:
            out_df[pcol] = out_df[pcol].round(int(args.prob_digits))

    pred_cols = [f"pred_{lab}" for lab in label_cols]
    out_df["pred_label_count"] = out_df[pred_cols].sum(axis=1).astype(int)
    out_df["pred_any_need"] = (out_df["pred_label_count"] > 0).astype(int)

    out_df.to_csv(out, index=False, encoding="utf-8-sig")

    # Write metadata + basic prediction counts.
    pred_counts = {lab: int(out_df[f"pred_{lab}"].sum()) for lab in label_cols}
    meta["pred_positives"] = pred_counts
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Wrote: {out}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
