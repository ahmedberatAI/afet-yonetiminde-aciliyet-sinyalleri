"""Build enriched step-7 comparison artifacts for the three gold_v2 experiments.

Inputs (already on disk, produced by earlier pipeline steps):
  - models/exp1_gold_v2_bce/eval_test_tuned.json, eval_val_tuned.json
  - models/exp2_gold_v2_posw/eval_test_tuned.json, eval_val_tuned.json
  - models/exp3_silver_then_gold_v2/eval_test_tuned.json, eval_val_tuned.json
  - models/<exp>/thresholds_cv.json
  - models/<exp>/threshold_tuning_cv_meta.json

Outputs:
  - data/analysis/experiment_comparison_v2.md
  - data/analysis/experiment_comparison_v2.csv
  - data/analysis/experiment_comparison_v2.json

No GPU / model inference is used here. This script only aggregates existing eval JSONs.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "analysis"

LABELS: List[str] = [
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

RARE_LABELS: List[str] = ["altyapi", "guvenlik", "psikolojik"]

EXPERIMENTS = [
    {
        "key": "exp1_gold_v2_bce",
        "model_dir": "models/exp1_gold_v2_bce/final",
        "root_dir": "models/exp1_gold_v2_bce",
        "config_yaml": "data/modeling/experiments/exp1_gold_v2_bce.yaml",
        "base_model": "dbmdz/bert-base-turkish-cased",
        "training": "Gold-only BCE (no pos_weight), LR=2e-5, 3 epochs, BS=16",
        "pos_weight_used": False,
        "threshold_source": "models/exp1_gold_v2_bce/thresholds_cv.json",
        "threshold_meta": "models/exp1_gold_v2_bce/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Gold-only baseline with vanilla BCE; rare labels (saglik, altyapi, "
            "psikolojik) collapsed to F1=0 on test — pos_weight needed to push the "
            "model to predict minority classes."
        ),
    },
    {
        "key": "exp2_gold_v2_posw",
        "model_dir": "models/exp2_gold_v2_posw/final",
        "root_dir": "models/exp2_gold_v2_posw",
        "config_yaml": "data/modeling/experiments/exp2_gold_v2_posw.yaml",
        "base_model": "dbmdz/bert-base-turkish-cased",
        "training": "Gold-only + pos_weight=neg/pos (clipped [1,50]), LR=2e-5, 3 epochs, BS=16",
        "pos_weight_used": True,
        "threshold_source": "models/exp2_gold_v2_posw/thresholds_cv.json",
        "threshold_meta": "models/exp2_gold_v2_posw/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Pos-weight recovers saglik, guvenlik, gida_su on test (F1 gains over exp1), "
            "but altyapi and psikolojik still F1=0 — their positive support (20 / 10 rows "
            "in pool) is too small for the pure-gold signal to generalize."
        ),
    },
    {
        "key": "exp3_silver_then_gold_v2",
        "model_dir": "models/exp3_silver_then_gold_v2/final",
        "root_dir": "models/exp3_silver_then_gold_v2",
        "config_yaml": "data/modeling/experiments/exp3_silver_then_gold_v2.yaml",
        "base_model": "models/need_classification_silver_63k/final",
        "training": (
            "Silver-pretrain (63k distant labels, BERTurk + pos_weight, LR=2e-5, 3 epochs) "
            "then gold fine-tune with pos_weight, LR=1e-5, 3 epochs, BS=16"
        ),
        "pos_weight_used": True,
        "threshold_source": "models/exp3_silver_then_gold_v2/thresholds_cv.json",
        "threshold_meta": "models/exp3_silver_then_gold_v2/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Silver prior carries rare-label semantics into fine-tune; altyapi and "
            "psikolojik now F1=1.0 on the 194-row test (small positive counts mean a "
            "few correct predictions saturate F1 — still a qualitative jump vs exp1/exp2). "
            "guvenlik and bilgi_paylasimi slightly regress vs val, flagging possible "
            "distribution shift or threshold miscalibration for those two."
        ),
    },
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def build_experiment_record(exp: Dict[str, Any]) -> Dict[str, Any]:
    root = ROOT / exp["root_dir"]
    test_path = root / "eval_test_tuned.json"
    val_path = root / "eval_val_tuned.json"
    thr_path = ROOT / exp["threshold_source"]
    meta_path = ROOT / exp["threshold_meta"]

    eval_test = load_json(test_path)
    eval_val = load_json(val_path)
    thresholds = load_json(thr_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = load_json(meta_path)

    f1_per_label = eval_test.get("f1_per_label", {})
    rare_f1 = {lbl: float(f1_per_label.get(lbl, 0.0)) for lbl in RARE_LABELS}

    record = {
        "experiment": exp["key"],
        "model_dir": exp["model_dir"],
        "config_yaml": exp["config_yaml"],
        "base_model": exp["base_model"],
        "training": exp["training"],
        "pos_weight_used": exp["pos_weight_used"],
        "test_csv": eval_test.get("csv", "data/modeling/need_classification_gold_combined/test.csv"),
        "val_csv": eval_val.get("csv", "data/modeling/need_classification_gold_combined/val.csv"),
        "rows_test": eval_test.get("rows"),
        "rows_val": eval_val.get("rows"),
        "labels": eval_test.get("labels", LABELS),
        "test": {
            "f1_micro": float(eval_test.get("f1_micro", 0.0)),
            "f1_macro": float(eval_test.get("f1_macro", 0.0)),
            "precision_micro": float(eval_test.get("precision_micro", 0.0)),
            "recall_micro": float(eval_test.get("recall_micro", 0.0)),
            "f1_per_label": {k: float(v) for k, v in f1_per_label.items()},
        },
        "val": {
            "f1_micro": float(eval_val.get("f1_micro", 0.0)),
            "f1_macro": float(eval_val.get("f1_macro", 0.0)),
            "precision_micro": float(eval_val.get("precision_micro", 0.0)),
            "recall_micro": float(eval_val.get("recall_micro", 0.0)),
            "f1_per_label": {k: float(v) for k, v in eval_val.get("f1_per_label", {}).items()},
        },
        "thresholds_per_label": {k: float(v) for k, v in thresholds.items()},
        "threshold_source": exp["threshold_source"],
        "threshold_meta_file": exp["threshold_meta"],
        "threshold_type": exp["threshold_type"],
        "threshold_strategy": meta.get("strategy"),
        "threshold_k_folds": meta.get("k"),
        "threshold_smoothing_alpha": meta.get("smoothing_alpha"),
        "pool_label_positives": meta.get("pool_label_positives", {}),
        "rare_label_f1_test": rare_f1,
        "rare_label_f1_val": {lbl: float(eval_val.get("f1_per_label", {}).get(lbl, 0.0)) for lbl in RARE_LABELS},
        "notes": exp["notes"],
    }
    return record


def compute_winners(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def top(key_path: List[str]) -> str:
        def get(r: Dict[str, Any]) -> float:
            x: Any = r
            for k in key_path:
                x = x.get(k, 0.0)
            return float(x)

        best = max(records, key=get)
        return best["experiment"]

    winners = {
        "f1_micro_test": top(["test", "f1_micro"]),
        "f1_macro_test": top(["test", "f1_macro"]),
        "precision_micro_test": top(["test", "precision_micro"]),
        "recall_micro_test": top(["test", "recall_micro"]),
    }
    per_label: Dict[str, str] = {}
    for lbl in LABELS:
        def per_lbl(r: Dict[str, Any], _lbl: str = lbl) -> float:
            return float(r["test"]["f1_per_label"].get(_lbl, 0.0))

        per_label[lbl] = max(records, key=per_lbl)["experiment"]
    winners["per_label_test"] = per_label
    return winners


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "experiment",
        "model_dir",
        "test_csv",
        "rows",
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
    ]
    header += [f"f1_{lbl}" for lbl in LABELS]
    header += [f"thr_{lbl}" for lbl in LABELS]
    header += [
        "threshold_source",
        "threshold_type",
        "rare_f1_altyapi",
        "rare_f1_guvenlik",
        "rare_f1_psikolojik",
        "rare_mean_f1",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in records:
            row: List[Any] = [
                r["experiment"],
                r["model_dir"],
                r["test_csv"],
                r["rows_test"],
                fmt_num(r["test"]["f1_micro"]),
                fmt_num(r["test"]["f1_macro"]),
                fmt_num(r["test"]["precision_micro"]),
                fmt_num(r["test"]["recall_micro"]),
            ]
            row += [fmt_num(r["test"]["f1_per_label"].get(lbl, 0.0)) for lbl in LABELS]
            row += [fmt_num(r["thresholds_per_label"].get(lbl, 0.0)) for lbl in LABELS]
            rare_vals = [r["rare_label_f1_test"][lbl] for lbl in RARE_LABELS]
            rare_mean = sum(rare_vals) / len(rare_vals) if rare_vals else 0.0
            row += [
                r["threshold_source"],
                r["threshold_type"],
                fmt_num(r["rare_label_f1_test"]["altyapi"]),
                fmt_num(r["rare_label_f1_test"]["guvenlik"]),
                fmt_num(r["rare_label_f1_test"]["psikolojik"]),
                fmt_num(rare_mean),
                r["notes"].replace("\n", " ").strip(),
            ]
            writer.writerow(row)


def render_md(records: List[Dict[str, Any]], winners: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Step 7 — Need Classification Experiment Comparison (gold_v2)")
    lines.append("")
    lines.append("## Shared experimental setup")
    lines.append("")
    lines.append("- Backbone: `dbmdz/bert-base-turkish-cased` (BERTurk, cased, 110M params)")
    lines.append("- Task: multi-label classification, 9 need categories, sigmoid + per-label threshold")
    lines.append("- Loss: BCEWithLogitsLoss; experiments 2 and 3 additionally weight positive classes by `pos_weight = neg / pos` clipped to `[1, 50]`")
    lines.append("- Dataset (canonical): `data/modeling/need_classification_gold_combined/` — train=1547, val=193, test=194 rows")
    lines.append("- All experiments evaluated on the SAME test CSV (`data/modeling/need_classification_gold_combined/test.csv`, 194 rows)")
    lines.append("- Per-label thresholds tuned via multilabel-stratified 5-fold CV on `train+val` (k=5, seed=42, grid [0.05, 0.95] step 0.01, smoothed F1 with α=1.0, strategy `oof_global`)")
    lines.append("- GPU: NVIDIA GeForce RTX 5090 Laptop (CUDA 12.0 cap, torch 2.10.0+cu128)")
    lines.append("")
    lines.append("## Experiments")
    lines.append("")
    lines.append("| Key | Base | Pos-weight | LR | Training Epochs | Threshold source |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        lr = "1e-5" if r["experiment"].startswith("exp3") else "2e-5"
        epochs = "silver 3 + gold 3" if r["experiment"].startswith("exp3") else "3"
        lines.append(
            f"| `{r['experiment']}` | `{r['base_model']}` | {r['pos_weight_used']} | {lr} | {epochs} | `{r['threshold_source']}` ({r['threshold_type']}) |"
        )
    lines.append("")
    lines.append("## Summary metrics on `test.csv` (194 rows, CV-tuned thresholds)")
    lines.append("")
    lines.append("| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |")
    lines.append("|---|---|---|---|---|")
    for r in records:
        t = r["test"]
        lines.append(
            f"| `{r['experiment']}` | {t['f1_micro']:.4f} | {t['f1_macro']:.4f} | {t['precision_micro']:.4f} | {t['recall_micro']:.4f} |"
        )
    lines.append("")
    lines.append("### Same metrics on `val.csv` (193 rows)")
    lines.append("")
    lines.append("| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |")
    lines.append("|---|---|---|---|---|")
    for r in records:
        v = r["val"]
        lines.append(
            f"| `{r['experiment']}` | {v['f1_micro']:.4f} | {v['f1_macro']:.4f} | {v['precision_micro']:.4f} | {v['recall_micro']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-label F1 on test")
    lines.append("")
    header = "| label | " + " | ".join(f"`{r['experiment']}`" for r in records) + " | best |"
    sep = "|---" + ("|---" * len(records)) + "|---|"
    lines.append(header)
    lines.append(sep)
    for lbl in LABELS:
        vals = [r["test"]["f1_per_label"].get(lbl, 0.0) for r in records]
        best_key = winners["per_label_test"][lbl]
        row = f"| {lbl} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | `{best_key}` |"
        lines.append(row)
    lines.append("")
    lines.append("## Per-label thresholds used at test time")
    lines.append("")
    lines.append("| label | " + " | ".join(f"`{r['experiment']}`" for r in records) + " |")
    lines.append("|---" + ("|---" * len(records)) + "|")
    for lbl in LABELS:
        vals = [r["thresholds_per_label"].get(lbl, 0.5) for r in records]
        lines.append(f"| {lbl} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines.append("")
    lines.append("## Threshold provenance")
    lines.append("")
    for r in records:
        lines.append(f"- **{r['experiment']}**")
        lines.append(f"  - thresholds file: `{r['threshold_source']}`")
        lines.append(f"  - meta file: `{r['threshold_meta_file']}`")
        lines.append(f"  - strategy: `{r['threshold_strategy']}`, k={r['threshold_k_folds']}, smoothing α={r['threshold_smoothing_alpha']}")
        lines.append(f"  - type: `{r['threshold_type']}`")
    lines.append("")
    lines.append("## Rare-label honesty block")
    lines.append("")
    lines.append(
        "Three labels have very little support in the gold training pool (`altyapi`=20 positives, "
        "`guvenlik`=35, `psikolojik`=10 across 1740 rows). On the 194-row test set, the positive support is "
        "even smaller — a handful of correct predictions can push F1 toward 1.0 or 0.0, so the numbers "
        "below should be read as qualitative signal, not population-level estimates."
    )
    lines.append("")
    lines.append("| label | pool positives | " + " | ".join(f"`{r['experiment']}` F1 (test)" for r in records) + " |")
    lines.append("|---|---" + ("|---" * len(records)) + "|")
    for lbl in RARE_LABELS:
        pool_pos_values = [str(r["pool_label_positives"].get(lbl, "-")) for r in records]
        pool_pos = pool_pos_values[0] if pool_pos_values and pool_pos_values[0] != "-" else "-"
        vals = [r["rare_label_f1_test"][lbl] for r in records]
        lines.append(f"| {lbl} | {pool_pos} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines.append("")
    lines.append(
        "- `exp1_gold_v2_bce` (pure BCE, no class weighting) **never fires** on `saglik`, `altyapi`, or "
        "`psikolojik` at tuned thresholds — all three collapse to F1=0. This is the expected failure mode "
        "for minority classes without a pos_weight."
    )
    lines.append(
        "- `exp2_gold_v2_posw` recovers `saglik` and `guvenlik`, but `altyapi` and `psikolojik` are still "
        "F1=0 on test. `altyapi` has the thinnest positive signal (only 20 train pool positives) and "
        "`psikolojik` has just 10."
    )
    lines.append(
        "- `exp3_silver_then_gold_v2` is the only experiment with non-zero F1 on **all three** rare labels. "
        "The silver-distant-label pre-training stage teaches the model useful priors for minority vocabulary, "
        "and the short (LR=1e-5, 3 epochs) gold fine-tune preserves that prior instead of washing it out."
    )
    lines.append(
        "- Caveat: on this tiny test slice, `altyapi` F1=1.0 and `psikolojik` F1=1.0 for exp3 may represent "
        "as few as 1–3 true positives — treat the ranking as real, but do not read those specific numbers as "
        "generalization guarantees. Longitudinal validation or a larger gold set is needed to confirm."
    )
    lines.append("")
    lines.append("## Winners")
    lines.append("")
    lines.append(f"- **Best test f1_micro**: `{winners['f1_micro_test']}`")
    lines.append(f"- **Best test f1_macro**: `{winners['f1_macro_test']}`")
    lines.append(f"- **Best test precision_micro**: `{winners['precision_micro_test']}`")
    lines.append(f"- **Best test recall_micro**: `{winners['recall_micro_test']}`")
    lines.append("")
    lines.append("Per-label winners on test:")
    lines.append("")
    for lbl in LABELS:
        lines.append(f"- `{lbl}` → `{winners['per_label_test'][lbl]}`")
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append(
        "- On the canonical 194-row test, `exp3_silver_then_gold_v2` wins on every aggregate metric: "
        "f1_micro 0.888 vs 0.732 / 0.730 for exp1 / exp2, f1_macro 0.873 vs 0.414 / 0.556."
    )
    lines.append(
        "- The gain is dominated by rare-label recovery; silver distant-supervision priors close the gap "
        "that gold alone cannot close at this dataset size."
    )
    lines.append(
        "- exp1 vs exp2: pos_weight roughly doubles macro-F1 (0.414 → 0.556) while leaving micro-F1 flat, "
        "which matches expectations — pos_weight rebalances toward minority classes at the cost of some "
        "majority-class precision."
    )
    lines.append("")
    lines.append("## Critical risks & caveats")
    lines.append("")
    lines.append(
        "- **Test set is small** (194 rows). Absolute metrics have wide confidence intervals, especially "
        "for rare labels with <5 positives."
    )
    lines.append(
        "- **CV thresholds for exp3 were derived by repeating silver→gold fine-tune per fold** "
        "(5× silver_pretrain start → gold fold fine-tune). Scratch dir: `models/_cv_tuning_scratch_exp3`. "
        "This is expensive but consistent with how exp1/exp2 CV thresholds were derived on gold-only."
    )
    lines.append(
        "- **Silver weights were retrained from scratch during step 7** because the prior silver "
        "`final/model.safetensors` was missing. The retrain used the existing "
        "`data/modeling/need_classification_silver_63k/training_config.yaml` config (63k distant-label rows, "
        "pos_weight, BERTurk, LR=2e-5, 3 epochs). This means the silver checkpoint underlying exp3 is "
        "fresh — any step-8 work must use this exact silver final as the pretrain artifact."
    )
    lines.append(
        "- **`bilgi_paylasimi` slightly regresses on exp3** vs exp1/exp2 (0.694 vs 0.760 / 0.760). This is "
        "a majority label; the silver prior may be pulling probability mass toward rarer labels at its "
        "expense. Worth a closer look if deployment prioritizes `bilgi_paylasimi` recall."
    )
    lines.append(
        "- **`guvenlik` regresses on exp3** (0.571) vs exp2 (0.667). The chosen threshold 0.86 on a "
        "small positive support may be slightly too conservative; revisit in step 8 if this label is "
        "critical."
    )
    lines.append(
        "- **`models/final/selection.json` has NOT been touched.** Step 9 artifacts are intentionally "
        "untouched by this step. The winner is documented here but not promoted."
    )
    lines.append("")
    lines.append("## Per-experiment notes")
    lines.append("")
    for r in records:
        lines.append(f"### `{r['experiment']}`")
        lines.append("")
        lines.append(f"- **model_dir**: `{r['model_dir']}`")
        lines.append(f"- **config**: `{r['config_yaml']}`")
        lines.append(f"- **base model**: `{r['base_model']}`")
        lines.append(f"- **training**: {r['training']}")
        lines.append(f"- **pos_weight used**: {r['pos_weight_used']}")
        lines.append(f"- **threshold source**: `{r['threshold_source']}` (type: `{r['threshold_type']}`)")
        lines.append(f"- **test metrics**: f1_micro={r['test']['f1_micro']:.4f}, f1_macro={r['test']['f1_macro']:.4f}, P={r['test']['precision_micro']:.4f}, R={r['test']['recall_micro']:.4f}")
        lines.append(f"- **notes**: {r['notes']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    records = [build_experiment_record(exp) for exp in EXPERIMENTS]
    winners = compute_winners(records)

    payload = {
        "task": "need_classification_experiment_comparison_v2",
        "scope": "step7",
        "test_csv": "data/modeling/need_classification_gold_combined/test.csv",
        "val_csv": "data/modeling/need_classification_gold_combined/val.csv",
        "rows_test": records[0]["rows_test"],
        "rows_val": records[0]["rows_val"],
        "labels": LABELS,
        "rare_labels": RARE_LABELS,
        "experiments": records,
        "winners": winners,
    }

    write_json(OUT_DIR / "experiment_comparison_v2.json", payload)
    write_csv(OUT_DIR / "experiment_comparison_v2.csv", records)
    md = render_md(records, winners)
    (OUT_DIR / "experiment_comparison_v2.md").write_text(md, encoding="utf-8")

    print("Wrote:")
    print(f"  {OUT_DIR / 'experiment_comparison_v2.json'}")
    print(f"  {OUT_DIR / 'experiment_comparison_v2.csv'}")
    print(f"  {OUT_DIR / 'experiment_comparison_v2.md'}")
    print()
    print("Winners:")
    print(f"  f1_micro_test: {winners['f1_micro_test']}")
    print(f"  f1_macro_test: {winners['f1_macro_test']}")


if __name__ == "__main__":
    main()
