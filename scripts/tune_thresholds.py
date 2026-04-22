#!/usr/bin/env python3
"""
Tune per-label thresholds for a multi-label classifier using a validation set.

This script:
- loads a trained HF model,
- runs inference on a CSV split,
- searches a threshold grid to maximize F1 per label,
- writes thresholds JSON that can be passed to scripts/evaluate_need_classifier.py

Why:
- A single global threshold often fails under severe label imbalance.
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


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_true/y_pred are 0/1 arrays (1D)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    p = argparse.ArgumentParser(description="Tune per-label thresholds on a validation CSV split.")
    p.add_argument("--model-dir", required=True, help="Path to model directory (e.g., models/need_classification/final).")
    p.add_argument("--csv", required=True, help="CSV split path (e.g., data/modeling/need_classification/val.csv).")
    p.add_argument("--text-col", default="tweet_clean", help="Text column name.")
    p.add_argument("--labels-json", default=None, help="Optional label_columns.json. If omitted, tries <model-dir>/../label_columns.json")
    p.add_argument("--max-length", type=int, default=192, help="Tokenizer max_length.")
    p.add_argument("--min-thr", type=float, default=0.01, help="Min threshold (inclusive).")
    p.add_argument("--max-thr", type=float, default=0.99, help="Max threshold (inclusive).")
    p.add_argument("--step", type=float, default=0.01, help="Threshold grid step.")
    p.add_argument("--out", default=None, help="Write thresholds JSON to this path (default: <model-dir>/../thresholds.json).")
    p.add_argument("--report", default=None, help="Optional report text path.")
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
        raise SystemExit(f"Missing dependency for tuning: {e}. Install torch + transformers.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference
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

    # Threshold grid
    if args.step <= 0:
        raise SystemExit("--step must be > 0")
    thr_grid = np.arange(args.min_thr, args.max_thr + 1e-9, args.step, dtype=np.float32)
    thr_grid = np.clip(thr_grid, 0.0, 1.0)

    thresholds: Dict[str, float] = {}
    per_label_best: Dict[str, Dict[str, Any]] = {}

    for j, lab in enumerate(label_cols):
        yt = y_true[:, j]
        pos = int(yt.sum())
        if pos == 0:
            thresholds[lab] = 0.5
            per_label_best[lab] = {"pos": pos, "best_thr": 0.5, "best_f1": 0.0, "note": "no positives in split"}
            continue

        best_thr = 0.5
        best_f1 = -1.0
        for thr in thr_grid:
            yp = (probs[:, j] >= float(thr)).astype(np.int64)
            f1 = _f1_binary(yt, yp)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        thresholds[lab] = float(best_thr)
        per_label_best[lab] = {"pos": pos, "best_thr": float(best_thr), "best_f1": float(best_f1)}

    out_path = Path(args.out) if args.out else (model_dir.parent / "thresholds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thresholds, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")

    # Optional report
    if args.report:
        rpath = Path(args.report)
        lines: List[str] = []
        lines.append("THRESHOLD TUNING REPORT\n")
        lines.append(f"Model: {model_dir.as_posix()}")
        lines.append(f"CSV: {csv_path.as_posix()}")
        lines.append(f"Grid: min={args.min_thr} max={args.max_thr} step={args.step} (n={len(thr_grid)})")
        lines.append("")
        for lab in label_cols:
            info = per_label_best[lab]
            note = info.get("note", "")
            if note:
                lines.append(f"- {lab}: pos={info['pos']} best_thr={info['best_thr']:.3f} best_f1={info['best_f1']:.3f} ({note})")
            else:
                lines.append(f"- {lab}: pos={info['pos']} best_thr={info['best_thr']:.3f} best_f1={info['best_f1']:.3f}")
        rpath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"Wrote: {rpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
