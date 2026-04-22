#!/usr/bin/env python3
"""
Select the "final" need classifier from a set of trained experiments.

Selection rule (documented in data/modeling/experiments/README.md):

    1. Highest F1-macro on the test set (CV-tuned thresholds).
    2. Tie-breaker: highest MIN per-label F1 across the three rare labels
       (altyapi, guvenlik, psikolojik). This biases the choice toward
       experiments that actually fix the rare-label blind spots, rather
       than ones that only inflate the micro average.

Inputs are the model directories (with eval_test_tuned.json, thresholds,
label_columns.json). The script prints the ranking, writes a pointer file
`models/final/selection.json` (symlinks are unreliable on Windows), and a
human-readable rationale at `docs/final_model_selection.md`.

Usage:
    python scripts/select_final_model.py \
        --exp models/exp1_gold_v2_bce \
        --exp models/exp2_gold_v2_posw \
        --exp models/exp3_silver_then_gold_v2 \
        --out-pointer models/final/selection.json \
        --out-md docs/final_model_selection.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RARE_LABELS: Tuple[str, ...] = ("altyapi", "guvenlik", "psikolojik")


def _parse_spec(spec: str) -> Tuple[Path, Optional[str]]:
    if "::" in spec:
        pstr, tag = spec.split("::", 1)
        return Path(pstr), tag
    return Path(spec), None


def _load_eval(path: Path) -> Dict[str, Any]:
    cands = [path / "eval_test_tuned.json"]
    cands += sorted(path.glob("eval_test*.json"))
    for c in cands:
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    raise SystemExit(f"No eval_test*.json in {path}")


def _summarize(path: Path, tag: Optional[str]) -> Dict[str, Any]:
    ev = _load_eval(path)
    per_label = ev.get("f1_per_label", {}) or {}
    rare_vals = [float(per_label[l]) for l in RARE_LABELS if l in per_label and per_label[l] is not None]
    # Prefer CV-tuned thresholds if present; otherwise fall back to val-tuned.
    if (path / "thresholds_cv.json").exists():
        thr_path = path / "thresholds_cv.json"
    elif (path / "thresholds.json").exists():
        thr_path = path / "thresholds.json"
    else:
        thr_path = path / "thresholds.json"  # will be flagged missing downstream
    return {
        "tag": tag or path.name,
        "model_dir": str(path.as_posix()),
        "final_dir": str((path / "final").as_posix()),
        "thresholds_json": str(thr_path.as_posix()),
        "thresholds_source": "cv" if thr_path.name == "thresholds_cv.json" else "val",
        "f1_macro": float(ev.get("f1_macro") or 0.0),
        "f1_micro": float(ev.get("f1_micro") or 0.0),
        "precision_micro": float(ev.get("precision_micro") or 0.0),
        "recall_micro": float(ev.get("recall_micro") or 0.0),
        "per_label_f1": {k: float(v) if v is not None else None for k, v in per_label.items()},
        "thresholds_per_label": ev.get("thresholds_per_label", {}),
        "rare_label_min_f1": (min(rare_vals) if rare_vals else 0.0),
        "rare_label_vals": {l: per_label.get(l) for l in RARE_LABELS},
        "rows": int(ev.get("rows") or 0),
    }


def _rank(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort: macro-F1 DESC, then rare_label_min_f1 DESC, then micro-F1 DESC.
    return sorted(
        results,
        key=lambda r: (
            -r["f1_macro"],
            -r["rare_label_min_f1"],
            -r["f1_micro"],
        ),
    )


def _render_md(ranked: List[Dict[str, Any]], winner: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Final Model Selection\n")
    lines.append("## Rule\n")
    lines.append("1. Maximize **F1-macro** on test (CV-tuned thresholds).")
    lines.append(f"2. Tie-breaker: maximize **min F1** across rare labels ({', '.join(RARE_LABELS)}).")
    lines.append("3. Last tie-breaker: F1-micro.\n")
    lines.append("## Ranking\n")
    lines.append("| rank | experiment | F1-macro | F1-micro | rare-min-F1 | "
                 "altyapi | guvenlik | psikolojik |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, start=1):
        rvals = r["rare_label_vals"]
        lines.append(
            f"| {i} | {r['tag']} | {r['f1_macro']:.3f} | {r['f1_micro']:.3f} | "
            f"{r['rare_label_min_f1']:.3f} | "
            f"{(rvals.get('altyapi') or 0.0):.3f} | "
            f"{(rvals.get('guvenlik') or 0.0):.3f} | "
            f"{(rvals.get('psikolojik') or 0.0):.3f} |"
        )
    lines.append("")
    lines.append("## Winner\n")
    lines.append(f"- **{winner['tag']}**")
    lines.append(f"- Model weights: `{winner['final_dir']}`")
    lines.append(f"- F1-macro={winner['f1_macro']:.3f}, F1-micro={winner['f1_micro']:.3f}, "
                 f"rare-min-F1={winner['rare_label_min_f1']:.3f}")
    lines.append("")
    lines.append("## Next steps\n")
    lines.append("1. Use the pointer at `models/final/selection.json` for predictions.")
    lines.append("2. Regenerate tweet predictions (step 9b):")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/predict_need_classifier.py \\")
    lines.append(f"  --model-dir {winner['final_dir']} \\")
    lines.append(f"  --labels-json {Path(winner['model_dir']).as_posix()}/label_columns.json \\")
    lines.append(f"  --thresholds-json {winner['thresholds_json']} \\")
    lines.append("  --input data/processed/emergency_geolocated_96k.csv \\")
    lines.append(f"  --output data/predictions/need_predictions_geolocated_{winner['tag']}.csv \\")
    lines.append("  --dedup-by-id")
    lines.append("```")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Pick the final multi-label need classifier.")
    p.add_argument("--exp", action="append", default=[], help="Model dir (optionally path::tag). Repeatable.")
    p.add_argument("--out-pointer", default="models/final/selection.json")
    p.add_argument("--out-md", default="docs/final_model_selection.md")
    args = p.parse_args()

    if not args.exp:
        raise SystemExit("Pass at least one --exp <model_dir>")

    results: List[Dict[str, Any]] = []
    for spec in args.exp:
        path, tag = _parse_spec(spec)
        if not path.exists():
            print(f"WARN: {path} missing, skipping.")
            continue
        try:
            results.append(_summarize(path, tag))
        except SystemExit as e:
            print(f"WARN: {e}")

    if not results:
        raise SystemExit("No valid experiments found.")

    ranked = _rank(results)
    winner = ranked[0]

    # Console recap
    print("\n=== Ranked experiments ===")
    for i, r in enumerate(ranked, start=1):
        print(
            f"  #{i} {r['tag']:40s}  "
            f"macro={r['f1_macro']:.3f}  micro={r['f1_micro']:.3f}  "
            f"rare-min={r['rare_label_min_f1']:.3f}"
        )
    print(f"\nWinner: {winner['tag']}  -> {winner['final_dir']}")

    pointer = Path(args.out_pointer)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "selected_tag": winner["tag"],
                "model_dir": winner["model_dir"],
                "final_dir": winner["final_dir"],
                "label_columns_json": str((Path(winner["model_dir"]) / "label_columns.json").as_posix()),
                "thresholds_json": winner["thresholds_json"],
                "thresholds_source": winner["thresholds_source"],
                "metrics": {
                    "f1_macro": winner["f1_macro"],
                    "f1_micro": winner["f1_micro"],
                    "rare_label_min_f1": winner["rare_label_min_f1"],
                    "rare_label_f1": winner["rare_label_vals"],
                },
                "ranking": [
                    {
                        "rank": i + 1,
                        "tag": r["tag"],
                        "f1_macro": r["f1_macro"],
                        "f1_micro": r["f1_micro"],
                        "rare_label_min_f1": r["rare_label_min_f1"],
                    }
                    for i, r in enumerate(ranked)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {pointer}")

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_md(ranked, winner), encoding="utf-8")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
