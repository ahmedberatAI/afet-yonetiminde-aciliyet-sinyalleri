#!/usr/bin/env python3
"""
Evaluate a trained multi-label need classifier on a CSV split.

This script expects:
- A model directory saved by HuggingFace (`trainer.save_model()`), and
- A CSV file with the same label columns used in training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _coerce_binary_0_1(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip().replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    if not df[col].isin([0, 1]).all():
        raise SystemExit(f"Invalid values in {col}: expected 0/1 only.")


def _load_xy(csv_path: Path, text_col: str, label_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in ([text_col] + label_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")
    for c in label_cols:
        _coerce_binary_0_1(df, c)
    x = df[text_col].astype("string").fillna("").tolist()
    y = df[label_cols].astype(int).to_numpy(dtype=np.int64)
    return x, y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_cols: List[str]) -> Dict[str, Any]:
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: scikit-learn. Install it before running evaluation.")

    out: Dict[str, Any] = {}
    out["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    out["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    out["f1_per_label"] = {label_cols[i]: float(per_label_f1[i]) for i in range(len(label_cols))}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained need classifier on a CSV split.")
    p.add_argument("--model-dir", required=True, help="Path to model directory (e.g., models/need_classification/final).")
    p.add_argument("--csv", required=True, help="CSV split path (e.g., data/modeling/need_classification/test.csv).")
    p.add_argument("--text-col", default="tweet_clean", help="Text column name.")
    p.add_argument(
        "--labels-json",
        default=None,
        help="Optional label_columns.json. If omitted, tries <model-dir>/../label_columns.json",
    )
    p.add_argument("--max-length", type=int, default=192, help="Tokenizer max_length (should match training).")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Global prediction threshold (used if --thresholds-json is not provided).",
    )
    p.add_argument(
        "--thresholds-json",
        default=None,
        help="Optional per-label threshold JSON (dict: label -> float). Overrides --threshold.",
    )
    p.add_argument("--out", default=None, help="Write metrics JSON to this path.")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    csv_path = Path(args.csv)

    labels_path = Path(args.labels_json) if args.labels_json else (model_dir.parent / "label_columns.json")
    if not labels_path.exists():
        raise SystemExit(f"Could not find label list JSON: {labels_path}")
    label_cols = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(label_cols, list) or not label_cols:
        raise SystemExit("label_columns.json must be a non-empty JSON list.")
    label_cols = [str(x) for x in label_cols]

    x, y_true = _load_xy(csv_path, text_col=args.text_col, label_cols=label_cols)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as e:
        raise SystemExit(f"Missing dependency for evaluation: {e}. Install torch + transformers.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 32
    all_logits: List[np.ndarray] = []
    for i in range(0, len(x), batch_size):
        batch_texts = x[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=int(args.max_length),
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = _sigmoid(logits)

    thresholds_used = {lab: float(args.threshold) for lab in label_cols}
    if args.thresholds_json:
        tpath = Path(args.thresholds_json)
        tdata = json.loads(tpath.read_text(encoding="utf-8"))
        if not isinstance(tdata, dict):
            raise SystemExit("--thresholds-json must be a JSON dict: label -> float")
        for lab in label_cols:
            if lab in tdata and tdata[lab] is not None:
                thresholds_used[lab] = float(tdata[lab])

    thr_vec = np.array([thresholds_used[lab] for lab in label_cols], dtype=np.float32).reshape(1, -1)
    y_pred = (probs >= thr_vec).astype(int)

    metrics = _compute_metrics(y_true=y_true, y_pred=y_pred, label_cols=label_cols)
    metrics["threshold"] = float(args.threshold)
    metrics["thresholds_per_label"] = thresholds_used
    metrics["rows"] = int(len(x))
    metrics["labels"] = label_cols
    metrics["csv"] = str(csv_path.as_posix())
    metrics["model_dir"] = str(model_dir.as_posix())

    out_path = Path(args.out) if args.out else (csv_path.parent / "eval_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
