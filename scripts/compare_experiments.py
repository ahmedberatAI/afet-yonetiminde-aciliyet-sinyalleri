#!/usr/bin/env python3
"""
Build a side-by-side comparison table of multi-label need-classification
experiments.

Reads `eval_test_tuned.json` (produced by scripts/evaluate_need_classifier.py)
from one or more model directories, prints a markdown table, and optionally
writes CSV / markdown / JSON outputs.

Usage:
    python scripts/compare_experiments.py \
        --exp models/exp1_gold_v2_bce \
        --exp models/exp2_gold_v2_posw \
        --exp models/exp3_silver_then_gold_v2 \
        --out-md data/analysis/experiment_comparison.md \
        --out-csv data/analysis/experiment_comparison.csv

You can also include historical v1 baselines (e.g.
  --exp models/need_classification_gold_combined_weighted --tag "v1_combined")
to sanity-check direction of change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "-"


def _load_experiment(path: Path, tag: Optional[str] = None) -> Dict[str, Any]:
    eval_json = path / "eval_test_tuned.json"
    thresholds_json = path / "thresholds.json"
    cv_thresholds = path / "thresholds_cv.json"

    if not eval_json.exists():
        # Fall back to evaluator at another conventional name.
        alt = list(path.glob("eval_test*.json"))
        if not alt:
            raise SystemExit(f"No eval_test_tuned.json in {path}")
        eval_json = sorted(alt)[0]

    with eval_json.open("r", encoding="utf-8") as f:
        ev = json.load(f)

    out: Dict[str, Any] = {
        "tag": tag or path.name,
        "model_dir": str(path.as_posix()),
        "eval_path": str(eval_json.as_posix()),
        "rows": int(ev.get("rows", 0)),
        "f1_micro": ev.get("f1_micro"),
        "f1_macro": ev.get("f1_macro"),
        "precision_micro": ev.get("precision_micro"),
        "recall_micro": ev.get("recall_micro"),
        "threshold_mode": "per_label" if "thresholds_per_label" in ev else "global",
        "thresholds_per_label": ev.get("thresholds_per_label", {}),
        "per_label_f1": ev.get("f1_per_label", {}),
        "has_val_thresholds": thresholds_json.exists(),
        "has_cv_thresholds": cv_thresholds.exists(),
    }
    return out


def _render_markdown(results: List[Dict[str, Any]], labels: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Experiment Comparison\n")
    lines.append(
        "Thresholds reflect whatever was written to `eval_test_tuned.json`. "
        "If the experiment was tuned with `tune_thresholds.py` on val, expect "
        "rare-label instability; CV-tuned versions (step 6) are more reliable.\n"
    )

    # Summary metrics table
    lines.append("## Summary metrics (test)\n")
    header = "| experiment | rows | F1-micro | F1-macro | P-micro | R-micro | thr-mode | cv-thr? |"
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            "| {tag} | {rows} | {f1m} | {f1M} | {pm} | {rm} | {thrmode} | {cv} |".format(
                tag=r["tag"],
                rows=r.get("rows", "-"),
                f1m=_fmt(r.get("f1_micro")),
                f1M=_fmt(r.get("f1_macro")),
                pm=_fmt(r.get("precision_micro")),
                rm=_fmt(r.get("recall_micro")),
                thrmode=r.get("threshold_mode", "-"),
                cv="yes" if r.get("has_cv_thresholds") else "no",
            )
        )
    lines.append("")

    # Per-label F1 matrix
    lines.append("## Per-label F1 (test)\n")
    head_cells = ["label"] + [r["tag"] for r in results]
    lines.append("| " + " | ".join(head_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(head_cells)) + "|")
    for lab in labels:
        row_cells = [lab]
        for r in results:
            v = r.get("per_label_f1", {}).get(lab)
            row_cells.append(_fmt(v))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Per-label thresholds used
    any_thr = any(r.get("thresholds_per_label") for r in results)
    if any_thr:
        lines.append("## Per-label thresholds used (test eval)\n")
        lines.append("| label | " + " | ".join(r["tag"] for r in results) + " |")
        lines.append("|" + "|".join(["---"] * (len(results) + 1)) + "|")
        for lab in labels:
            row = [lab]
            for r in results:
                v = r.get("thresholds_per_label", {}).get(lab)
                row.append(_fmt(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Winners per label
    lines.append("## Winner per label (highest F1)\n")
    for lab in labels:
        best_tag = None
        best_val = -1.0
        for r in results:
            v = r.get("per_label_f1", {}).get(lab)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f > best_val:
                best_val = f
                best_tag = r["tag"]
        if best_tag is None:
            lines.append(f"- **{lab}**: (no valid results)")
        else:
            lines.append(f"- **{lab}**: {best_tag} (F1={best_val:.3f})")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_csv(results: List[Dict[str, Any]], labels: List[str]) -> str:
    rows = []
    header = [
        "tag",
        "model_dir",
        "rows",
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "threshold_mode",
        "has_cv_thresholds",
    ]
    for lab in labels:
        header.append(f"f1_{lab}")
    for lab in labels:
        header.append(f"thr_{lab}")
    rows.append(",".join(header))
    for r in results:
        row: List[str] = [
            r["tag"],
            r["model_dir"],
            str(r.get("rows", "")),
            _fmt(r.get("f1_micro")),
            _fmt(r.get("f1_macro")),
            _fmt(r.get("precision_micro")),
            _fmt(r.get("recall_micro")),
            str(r.get("threshold_mode", "")),
            "1" if r.get("has_cv_thresholds") else "0",
        ]
        for lab in labels:
            row.append(_fmt(r.get("per_label_f1", {}).get(lab)))
        for lab in labels:
            row.append(_fmt(r.get("thresholds_per_label", {}).get(lab)))
        rows.append(",".join(row))
    return "\n".join(rows) + "\n"


def _parse_exp_spec(spec: str) -> Tuple[Path, Optional[str]]:
    if "::" in spec:
        path_str, tag = spec.split("::", 1)
        return Path(path_str), tag
    return Path(spec), None


def main() -> int:
    p = argparse.ArgumentParser(description="Compare need-classifier experiments side-by-side.")
    p.add_argument(
        "--exp",
        action="append",
        default=[],
        help="Path to a model directory (format: path or path::tag). Repeatable.",
    )
    p.add_argument("--labels-json", default=None, help="Override label list JSON.")
    p.add_argument("--out-md", default=None, help="Markdown output path.")
    p.add_argument("--out-csv", default=None, help="CSV output path.")
    p.add_argument("--out-json", default=None, help="JSON output path (raw aggregated results).")
    args = p.parse_args()

    if not args.exp:
        raise SystemExit("Pass at least one --exp <model_dir>")

    labels = NEED_LABEL_COLS
    if args.labels_json:
        labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []
    for spec in args.exp:
        path, tag = _parse_exp_spec(spec)
        if not path.exists():
            print(f"WARN: {path} does not exist, skipping.")
            continue
        try:
            results.append(_load_experiment(path, tag=tag))
        except SystemExit as e:
            print(f"WARN: {e}, skipping {path}.")

    if not results:
        raise SystemExit("No valid experiments found.")

    md = _render_markdown(results, labels)
    print(md)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"Wrote: {args.out_md}")

    if args.out_csv:
        csv = _render_csv(results, labels)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_csv).write_text(csv, encoding="utf-8")
        print(f"Wrote: {args.out_csv}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
