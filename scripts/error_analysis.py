#!/usr/bin/env python3
"""
Error analysis for a trained multi-label need classifier.

Given a model directory, a test CSV, and optionally a per-label threshold
JSON, this script produces:

1. Per-label confusion counts (TP/FP/FN/TN) and F1.
2. Top-N false-positive and false-negative example rows per label.
3. A co-occurrence / "confusion pair" matrix: for each FN on label A, which
   OTHER labels fired (predicted 1) on that row? This surfaces systematic
   over-firing of one label at the expense of another.
4. Error-rate slices by `aciliyet_0_3` (urgency bucket) and by text length
   bucket (short / medium / long), using character counts on tweet_clean.
5. A plain-text "first-pass" markdown report plus a machine-readable JSON.

Usage:
    python scripts/error_analysis.py \
        --model-dir models/exp3_silver_then_gold_v3_exgold/final \
        --test-csv data/modeling/need_classification_gold_combined/test.csv \
        --labels-json models/exp3_silver_then_gold_v3_exgold/label_columns.json \
        --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json \
        --out-md data/analysis/error_analysis_v2_leakfree.md \
        --out-json data/analysis/error_analysis_v2_leakfree.json \
        --top-n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _predict(
    model_dir: Path,
    texts: List[str],
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_logits: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        all_logits.append(out.logits.detach().cpu().numpy())
    return _sigmoid(np.concatenate(all_logits, axis=0))


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1}


def _length_bucket(n: int) -> str:
    if n < 60:
        return "short(<60)"
    if n < 140:
        return "medium(60-139)"
    return "long(>=140)"


def _slice_error_rate(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    any_err = (y_true != y_pred).any(axis=1).astype(int)
    grp = df.assign(_err=any_err).groupby(group_col, dropna=False)
    for g, sub in grp:
        n = len(sub)
        err = int(sub["_err"].sum())
        rows.append(
            {
                "group": str(g),
                "rows": n,
                "rows_with_any_error": err,
                "error_rate": err / n if n else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["error_rate"], -r["rows"]))
    return rows


def _confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """For each label A, how often does label B fire while A is a FN on the same row?"""
    out: Dict[str, Dict[str, int]] = {}
    for i, la in enumerate(labels):
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        if not fn_mask.any():
            out[la] = {}
            continue
        other_fire = y_pred[fn_mask].sum(axis=0)
        pair_counts: Dict[str, int] = {}
        for j, lb in enumerate(labels):
            if j == i:
                continue
            c = int(other_fire[j])
            if c > 0:
                pair_counts[lb] = c
        out[la] = dict(sorted(pair_counts.items(), key=lambda kv: -kv[1]))
    return out


def _pick_examples(
    df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    kind: str,
    top_n: int,
    text_col: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """kind ∈ {"fp", "fn"}. Ranks by probability (FP desc, FN asc)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, lab in enumerate(labels):
        if kind == "fp":
            mask = (y_true[:, i] == 0) & (y_pred[:, i] == 1)
            # highest prob = most confident FP
            order = np.argsort(-probs[:, i])
        else:
            mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
            # lowest prob = most missed FN
            order = np.argsort(probs[:, i])
        idxs = [int(k) for k in order if bool(mask[int(k)])][:top_n]
        rows = []
        for k in idxs:
            row = {
                "prob": float(probs[k, i]),
                "gold_labels": [labels[j] for j in range(len(labels)) if y_true[k, j] == 1],
                "pred_labels": [labels[j] for j in range(len(labels)) if y_pred[k, j] == 1],
                "text": str(df.iloc[k].get(text_col, ""))[:300],
            }
            if "id" in df.columns:
                row["id"] = str(df.iloc[k]["id"])
            if "aciliyet_0_3" in df.columns:
                try:
                    row["aciliyet_0_3"] = int(df.iloc[k]["aciliyet_0_3"])
                except Exception:
                    pass
            rows.append(row)
        out[lab] = rows
    return out


def _render_markdown(
    model_dir: str,
    test_csv: str,
    labels: List[str],
    thresholds_used: Dict[str, float],
    per_label_conf: Dict[str, Dict[str, Any]],
    confusion_pairs: Dict[str, Dict[str, int]],
    slice_aciliyet: List[Dict[str, Any]],
    slice_length: List[Dict[str, Any]],
    fp_examples: Dict[str, List[Dict[str, Any]]],
    fn_examples: Dict[str, List[Dict[str, Any]]],
    top_n: int,
) -> str:
    lines: List[str] = []
    lines.append("# Error Analysis\n")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Test CSV: `{test_csv}`")
    lines.append(f"- Top-N per label: {top_n}")
    lines.append("")

    lines.append("## Per-label confusion\n")
    lines.append("| label | thr | TP | FP | FN | TN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        c = per_label_conf[lab]
        lines.append(
            f"| {lab} | {thresholds_used.get(lab, 0.5):.3f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | "
            f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |"
        )
    lines.append("")

    lines.append("## Confusion pairs (for each label's FNs, which OTHER labels fired)\n")
    for lab in labels:
        pairs = confusion_pairs.get(lab, {})
        if not pairs:
            lines.append(f"- **{lab}**: (no false negatives, or no co-fires)")
            continue
        top = list(pairs.items())[:5]
        lines.append(f"- **{lab}**: " + ", ".join(f"{k}×{v}" for k, v in top))
    lines.append("")

    if slice_aciliyet:
        lines.append("## Error rate by aciliyet_0_3\n")
        lines.append("| bucket | rows | with_any_error | rate |")
        lines.append("|---|---|---|---|")
        for r in slice_aciliyet:
            lines.append(f"| {r['group']} | {r['rows']} | {r['rows_with_any_error']} | {r['error_rate']:.3f} |")
        lines.append("")

    if slice_length:
        lines.append("## Error rate by text length\n")
        lines.append("| bucket | rows | with_any_error | rate |")
        lines.append("|---|---|---|---|")
        for r in slice_length:
            lines.append(f"| {r['group']} | {r['rows']} | {r['rows_with_any_error']} | {r['error_rate']:.3f} |")
        lines.append("")

    lines.append("## False negatives (missed positives) — most-confident misses\n")
    for lab in labels:
        rows = fn_examples.get(lab, [])
        if not rows:
            lines.append(f"### {lab}\n\n(none)\n")
            continue
        lines.append(f"### {lab}\n")
        for r in rows:
            rid = r.get("id", "-")
            ac = r.get("aciliyet_0_3", "-")
            gold = ",".join(r["gold_labels"]) or "-"
            pred = ",".join(r["pred_labels"]) or "-"
            lines.append(f"- `id={rid}` p={r['prob']:.3f} aciliyet={ac}")
            lines.append(f"  - gold: `{gold}` | pred: `{pred}`")
            lines.append(f"  - text: {r['text']}")
        lines.append("")

    lines.append("## False positives (spurious fires) — most-confident FPs\n")
    for lab in labels:
        rows = fp_examples.get(lab, [])
        if not rows:
            lines.append(f"### {lab}\n\n(none)\n")
            continue
        lines.append(f"### {lab}\n")
        for r in rows:
            rid = r.get("id", "-")
            ac = r.get("aciliyet_0_3", "-")
            gold = ",".join(r["gold_labels"]) or "-"
            pred = ",".join(r["pred_labels"]) or "-"
            lines.append(f"- `id={rid}` p={r['prob']:.3f} aciliyet={ac}")
            lines.append(f"  - gold: `{gold}` | pred: `{pred}`")
            lines.append(f"  - text: {r['text']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Per-label error analysis for a multi-label need classifier.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--labels-json", default=None)
    p.add_argument("--thresholds-json", default=None)
    p.add_argument("--text-col", default="tweet_clean")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--threshold", type=float, default=0.5, help="Fallback if --thresholds-json is missing.")
    p.add_argument("--top-n", type=int, default=10, help="Number of FP/FN examples per label.")
    p.add_argument("--out-md", default=None)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    test_csv = Path(args.test_csv)

    labels_path: Optional[Path]
    if args.labels_json:
        labels_path = Path(args.labels_json)
    else:
        # Try common layouts: <model_dir>/label_columns.json or <model_dir>/../label_columns.json
        if (model_dir / "label_columns.json").exists():
            labels_path = model_dir / "label_columns.json"
        else:
            labels_path = model_dir.parent / "label_columns.json"
    if not labels_path or not labels_path.exists():
        raise SystemExit(f"Could not locate label_columns.json (tried {labels_path}).")
    labels: List[str] = [str(x) for x in json.loads(labels_path.read_text(encoding="utf-8"))]

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    for c in labels:
        if c not in df.columns:
            raise SystemExit(f"Missing label column in test csv: {c}")
        _coerce_binary_0_1(df, c)
    texts = df[args.text_col].astype("string").fillna("").tolist()
    y_true = df[labels].astype(int).to_numpy()

    probs = _predict(model_dir, texts, max_length=int(args.max_length))

    thresholds_used: Dict[str, float] = {lab: float(args.threshold) for lab in labels}
    if args.thresholds_json:
        tdata = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
        if not isinstance(tdata, dict):
            raise SystemExit("--thresholds-json must be a dict: label -> float.")
        for lab in labels:
            if lab in tdata and tdata[lab] is not None:
                thresholds_used[lab] = float(tdata[lab])
    thr_vec = np.array([thresholds_used[lab] for lab in labels], dtype=np.float32).reshape(1, -1)
    y_pred = (probs >= thr_vec).astype(int)

    per_label_conf: Dict[str, Dict[str, Any]] = {
        lab: _confusion_counts(y_true[:, i], y_pred[:, i]) for i, lab in enumerate(labels)
    }
    pairs = _confusion_pairs(y_true, y_pred, labels)

    slice_aciliyet: List[Dict[str, Any]] = []
    if "aciliyet_0_3" in df.columns:
        slice_aciliyet = _slice_error_rate(df, y_true, y_pred, "aciliyet_0_3")

    df_len = df.assign(_len_bucket=df[args.text_col].astype("string").fillna("").str.len().map(_length_bucket))
    slice_length = _slice_error_rate(df_len, y_true, y_pred, "_len_bucket")

    fp_ex = _pick_examples(df, probs, y_true, y_pred, labels, "fp", int(args.top_n), args.text_col)
    fn_ex = _pick_examples(df, probs, y_true, y_pred, labels, "fn", int(args.top_n), args.text_col)

    if args.out_md:
        md = _render_markdown(
            model_dir=str(model_dir.as_posix()),
            test_csv=str(test_csv.as_posix()),
            labels=labels,
            thresholds_used=thresholds_used,
            per_label_conf=per_label_conf,
            confusion_pairs=pairs,
            slice_aciliyet=slice_aciliyet,
            slice_length=slice_length,
            fp_examples=fp_ex,
            fn_examples=fn_ex,
            top_n=int(args.top_n),
        )
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"Wrote: {args.out_md}")

    if args.out_json:
        payload = {
            "model_dir": str(model_dir.as_posix()),
            "test_csv": str(test_csv.as_posix()),
            "labels": labels,
            "thresholds_used": thresholds_used,
            "per_label_confusion": per_label_conf,
            "confusion_pairs": pairs,
            "slice_by_aciliyet_0_3": slice_aciliyet,
            "slice_by_text_length": slice_length,
            "fp_examples": fp_ex,
            "fn_examples": fn_ex,
            "rows": int(len(df)),
            "top_n": int(args.top_n),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_json}")

    # Short console summary
    print("\n=== Per-label F1 ===")
    for lab in labels:
        c = per_label_conf[lab]
        print(f"  {lab:18s}  F1={c['f1']:.3f}  (TP={c['tp']} FP={c['fp']} FN={c['fn']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
