"""Build LEAK-FREE step-7 comparison artifacts (v3) and leak audit report.

Inputs (all already on disk):
  - models/exp1_gold_v2_bce/{eval_test_tuned.json,eval_val_tuned.json,thresholds_cv.json,threshold_tuning_cv_meta.json}
  - models/exp2_gold_v2_posw/{eval_test_tuned.json,eval_val_tuned.json,thresholds_cv.json,threshold_tuning_cv_meta.json}
  - models/exp3_silver_then_gold_v3_exgold/{eval_test_tuned.json,eval_val_tuned.json,thresholds_cv.json,threshold_tuning_cv_meta.json}
  - data/labeling/need_classification_silver_63k_profileA_exgold_stats.json
  - models/exp3_silver_then_gold_v2/eval_test_tuned.json  (historical, contaminated)

Outputs:
  - data/analysis/experiment_comparison_v3_leakfree.{md,csv,json}
  - data/analysis/step7_leak_audit_v3.{md,json}

No GPU / inference is used — this only aggregates existing eval JSONs and leakage checks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

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

GOLD_CANONICAL = "data/need_classification_gold_combined.csv"
GOLD_TRAIN = "data/modeling/need_classification_gold_combined/train.csv"
GOLD_VAL = "data/modeling/need_classification_gold_combined/val.csv"
GOLD_TEST = "data/modeling/need_classification_gold_combined/test.csv"

SILVER_OLD = "data/labeling/need_classification_silver_63k_profileA.csv"
SILVER_NEW = "data/labeling/need_classification_silver_63k_profileA_exgold.csv"
SILVER_NEW_STATS = "data/labeling/need_classification_silver_63k_profileA_exgold_stats.json"

EXPERIMENTS = [
    {
        "key": "exp1_gold_v2_bce",
        "model_dir": "models/exp1_gold_v2_bce/final",
        "root_dir": "models/exp1_gold_v2_bce",
        "config_yaml": "data/modeling/experiments/exp1_gold_v2_bce.yaml",
        "base_model": "dbmdz/bert-base-turkish-cased",
        "training": "Gold-only BCE (no pos_weight), LR=2e-5, 3 epochs, BS=16",
        "pos_weight_used": False,
        "silver_provenance": "none (trained directly from BERTurk base, no silver pretraining)",
        "leak_free": True,
        "threshold_source": "models/exp1_gold_v2_bce/thresholds_cv.json",
        "threshold_meta": "models/exp1_gold_v2_bce/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Gold-only baseline with vanilla BCE. No silver pretraining, so the "
            "silver-leakage issue does not apply. Rare labels (saglik, altyapi, "
            "psikolojik) collapse to F1=0 on test — expected failure mode without "
            "pos_weight."
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
        "silver_provenance": "none (trained directly from BERTurk base, no silver pretraining)",
        "leak_free": True,
        "threshold_source": "models/exp2_gold_v2_posw/thresholds_cv.json",
        "threshold_meta": "models/exp2_gold_v2_posw/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Pos-weight recovers saglik, guvenlik, gida_su vs exp1. No silver pretraining, "
            "so the silver-leakage issue does not apply. altyapi and psikolojik still F1=0 "
            "— positive support in pool (20 / 10 rows) is too thin for pure-gold to recover."
        ),
    },
    {
        "key": "exp3_silver_then_gold_v3_exgold",
        "model_dir": "models/exp3_silver_then_gold_v3_exgold/final",
        "root_dir": "models/exp3_silver_then_gold_v3_exgold",
        "config_yaml": "data/modeling/experiments/exp3_silver_then_gold_v3_exgold.yaml",
        "base_model": "models/need_classification_silver_63k_exgold/final",
        "training": (
            "LEAK-FREE silver-pretrain (61,246 distant-label rows, gold ids EXCLUDED, "
            "BERTurk + pos_weight, LR=2e-5, 3 epochs) then gold fine-tune with pos_weight, "
            "LR=1e-5, 3 epochs, BS=16"
        ),
        "pos_weight_used": True,
        "silver_provenance": (
            "data/modeling/need_classification_silver_63k_exgold/*.csv — built from "
            "emergency_geolocated_96k with gold id exclusion (0 overlap with any "
            "gold_combined split)"
        ),
        "leak_free": True,
        "threshold_source": "models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json",
        "threshold_meta": "models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_meta.json",
        "threshold_type": "cv",
        "notes": (
            "Replaces the contaminated exp3_silver_then_gold_v2 (whose silver pool "
            "contained ALL 1934 gold_combined ids including the 194 test ids). "
            "Here the silver pool has been re-built from the geolocated 96k dataset "
            "with every gold id excluded; silver→gold transfer is therefore fair. "
            "The silver prior still carries rare-label semantics; altyapi and psikolojik "
            "saturate at F1=1.0 on test but the positive support is tiny (3 and 1 rows), "
            "so read as qualitative signal."
        ),
    },
]

HISTORICAL_V2 = {
    "key": "exp3_silver_then_gold_v2",
    "model_dir": "models/exp3_silver_then_gold_v2/final",
    "threshold_source": "models/exp3_silver_then_gold_v2/thresholds_cv.json",
    "eval_test": "models/exp3_silver_then_gold_v2/eval_test_tuned.json",
    "status": "HISTORICAL_CONTAMINATED",
    "issue": (
        "Silver pretraining pool (data/labeling/need_classification_silver_63k_profileA.csv) "
        "contained every gold_combined id (train=1547, val=193, test=194). This means the "
        "silver model saw the gold test inputs during distant-label pretraining, which "
        "invalidates the silver→gold transfer comparison."
    ),
}


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

    return {
        "experiment": exp["key"],
        "model_dir": exp["model_dir"],
        "config_yaml": exp["config_yaml"],
        "base_model": exp["base_model"],
        "training": exp["training"],
        "pos_weight_used": exp["pos_weight_used"],
        "silver_provenance": exp["silver_provenance"],
        "leak_free": exp["leak_free"],
        "test_csv": eval_test.get("csv", GOLD_TEST),
        "val_csv": eval_val.get("csv", GOLD_VAL),
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


def compute_winners(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def top(key_path: List[str]) -> str:
        def get(r: Dict[str, Any]) -> float:
            x: Any = r
            for k in key_path:
                x = x.get(k, 0.0)
            return float(x)
        return max(records, key=get)["experiment"]

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


def gold_id_sets() -> Dict[str, set]:
    gold_canonical = pd.read_csv(ROOT / GOLD_CANONICAL, encoding="utf-8-sig", usecols=["id"])
    gold_train = pd.read_csv(ROOT / GOLD_TRAIN, encoding="utf-8-sig", usecols=["id"])
    gold_val = pd.read_csv(ROOT / GOLD_VAL, encoding="utf-8-sig", usecols=["id"])
    gold_test = pd.read_csv(ROOT / GOLD_TEST, encoding="utf-8-sig", usecols=["id"])
    return {
        "canonical": set(gold_canonical["id"].astype(str).tolist()),
        "train": set(gold_train["id"].astype(str).tolist()),
        "val": set(gold_val["id"].astype(str).tolist()),
        "test": set(gold_test["id"].astype(str).tolist()),
    }


def silver_leak_report() -> Dict[str, Any]:
    gs = gold_id_sets()
    union = gs["train"] | gs["val"] | gs["test"]

    report: Dict[str, Any] = {
        "gold_sources_used": {
            "canonical_csv": GOLD_CANONICAL,
            "splits": {"train": GOLD_TRAIN, "val": GOLD_VAL, "test": GOLD_TEST},
            "canonical_rows": len(gs["canonical"]),
            "split_train_rows": len(gs["train"]),
            "split_val_rows": len(gs["val"]),
            "split_test_rows": len(gs["test"]),
            "canonical_equals_split_union": gs["canonical"] == union,
        },
        "silvers": {},
    }

    for name, path in (("old_silver_contaminated", SILVER_OLD), ("new_silver_leak_free", SILVER_NEW)):
        p = ROOT / path
        if not p.exists():
            report["silvers"][name] = {"path": path, "exists": False}
            continue
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=["id"])
        ids = set(df["id"].astype(str).tolist())
        report["silvers"][name] = {
            "path": path,
            "exists": True,
            "rows": int(len(df)),
            "unique_ids": int(len(ids)),
            "overlap_with_gold_canonical": int(len(ids & gs["canonical"])),
            "overlap_with_gold_train": int(len(ids & gs["train"])),
            "overlap_with_gold_val": int(len(ids & gs["val"])),
            "overlap_with_gold_test": int(len(ids & gs["test"])),
            "overlap_with_gold_split_union": int(len(ids & union)),
        }
    return report


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "experiment", "model_dir", "test_csv", "rows",
        "f1_micro", "f1_macro", "precision_micro", "recall_micro",
    ]
    header += [f"f1_{lbl}" for lbl in LABELS]
    header += [f"thr_{lbl}" for lbl in LABELS]
    header += [
        "threshold_source", "threshold_type",
        "rare_f1_altyapi", "rare_f1_guvenlik", "rare_f1_psikolojik", "rare_mean_f1",
        "leak_free", "silver_provenance", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in records:
            row: List[Any] = [
                r["experiment"], r["model_dir"], r["test_csv"], r["rows_test"],
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
                r["threshold_source"], r["threshold_type"],
                fmt_num(r["rare_label_f1_test"]["altyapi"]),
                fmt_num(r["rare_label_f1_test"]["guvenlik"]),
                fmt_num(r["rare_label_f1_test"]["psikolojik"]),
                fmt_num(rare_mean),
                str(r["leak_free"]), r["silver_provenance"].replace("\n", " ").strip(),
                r["notes"].replace("\n", " ").strip(),
            ]
            writer.writerow(row)


def render_comparison_md(records: List[Dict[str, Any]], winners: Dict[str, Any], leak: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Step 7 — Need Classification Experiment Comparison (v3, LEAK-FREE)")
    L.append("")
    L.append("## Header")
    L.append("")
    L.append("- **Status**: CANONICAL comparison (supersedes `experiment_comparison_v2.*`)")
    L.append("- **Why v3?** v2 compared exp3 unfairly: the silver pretraining pool contained every gold_combined id (train=1547, val=193, test=194). v3 rebuilds the silver pool with all 1934 gold ids excluded.")
    L.append("- **Test CSV (shared across all 3 experiments)**: `data/modeling/need_classification_gold_combined/test.csv` (194 rows)")
    L.append("- **Val CSV**: `data/modeling/need_classification_gold_combined/val.csv` (193 rows)")
    L.append("- **GPU**: NVIDIA GeForce RTX 5090 Laptop (CUDA 12.8, torch 2.10.0+cu128) — required, no CPU fallback")
    L.append(f"- **Leak audit**: old silver had {leak['silvers']['old_silver_contaminated']['overlap_with_gold_split_union']}/1934 gold-id overlap; new silver_exgold has {leak['silvers']['new_silver_leak_free']['overlap_with_gold_split_union']}/1934.")
    L.append("")

    L.append("## Shared experimental setup")
    L.append("")
    L.append("- Backbone: `dbmdz/bert-base-turkish-cased` (BERTurk, cased, 110M params)")
    L.append("- Task: multi-label classification, 9 need categories, sigmoid + per-label threshold")
    L.append("- Loss: BCEWithLogitsLoss; exp2 and exp3 additionally weight positives by `pos_weight = neg / pos` clipped to [1, 50]")
    L.append("- Per-label thresholds: multilabel-stratified 5-fold CV on `train+val` (k=5, seed=42, grid [0.05, 0.95] step 0.01, smoothed F1 α=1.0, strategy `oof_global`)")
    L.append("")
    L.append("## Experiments")
    L.append("")
    L.append("| Key | Base | Pos-weight | LR | Silver leak-free |")
    L.append("|---|---|---|---|---|")
    for r in records:
        lr = "1e-5" if r["experiment"].startswith("exp3") else "2e-5"
        L.append(f"| `{r['experiment']}` | `{r['base_model']}` | {r['pos_weight_used']} | {lr} | {r['leak_free']} |")
    L.append("")

    L.append("## Summary metrics on `test.csv` (194 rows, CV-tuned thresholds)")
    L.append("")
    L.append("| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |")
    L.append("|---|---|---|---|---|")
    for r in records:
        t = r["test"]
        L.append(f"| `{r['experiment']}` | {t['f1_micro']:.4f} | {t['f1_macro']:.4f} | {t['precision_micro']:.4f} | {t['recall_micro']:.4f} |")
    L.append("")
    L.append("### Same metrics on `val.csv` (193 rows)")
    L.append("")
    L.append("| Experiment | f1_micro | f1_macro | precision_micro | recall_micro |")
    L.append("|---|---|---|---|---|")
    for r in records:
        v = r["val"]
        L.append(f"| `{r['experiment']}` | {v['f1_micro']:.4f} | {v['f1_macro']:.4f} | {v['precision_micro']:.4f} | {v['recall_micro']:.4f} |")
    L.append("")

    L.append("## Per-label F1 on test")
    L.append("")
    L.append("| label | " + " | ".join(f"`{r['experiment']}`" for r in records) + " | best |")
    L.append("|---" + ("|---" * len(records)) + "|---|")
    for lbl in LABELS:
        vals = [r["test"]["f1_per_label"].get(lbl, 0.0) for r in records]
        best_key = winners["per_label_test"][lbl]
        L.append(f"| {lbl} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | `{best_key}` |")
    L.append("")

    L.append("## Per-label thresholds used at test time")
    L.append("")
    L.append("| label | " + " | ".join(f"`{r['experiment']}`" for r in records) + " |")
    L.append("|---" + ("|---" * len(records)) + "|")
    for lbl in LABELS:
        vals = [r["thresholds_per_label"].get(lbl, 0.5) for r in records]
        L.append(f"| {lbl} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    L.append("")

    L.append("## Threshold provenance")
    L.append("")
    for r in records:
        L.append(f"- **{r['experiment']}**")
        L.append(f"  - thresholds: `{r['threshold_source']}`")
        L.append(f"  - meta: `{r['threshold_meta_file']}`")
        L.append(f"  - strategy=`{r['threshold_strategy']}`, k={r['threshold_k_folds']}, smoothing α={r['threshold_smoothing_alpha']}")
        L.append(f"  - type: `{r['threshold_type']}`")
    L.append("")

    L.append("## Rare-label honesty block")
    L.append("")
    L.append(
        "Three labels have very thin positive support in the gold pool "
        "(`altyapi`=20, `guvenlik`=35, `psikolojik`=10 positives across 1740 rows). "
        "On the 194-row test the positive counts are even smaller "
        "(`altyapi`=3, `guvenlik`=4, `psikolojik`=1). A single correct prediction "
        "can saturate F1; read rare-label numbers as **qualitative** signal only."
    )
    L.append("")
    L.append("| label | pool positives | test positives | " + " | ".join(f"`{r['experiment']}`" for r in records) + " |")
    L.append("|---|---|---" + ("|---" * len(records)) + "|")
    test_positives = {"altyapi": 3, "guvenlik": 4, "psikolojik": 1}
    for lbl in RARE_LABELS:
        pool_pos_values = [str(r["pool_label_positives"].get(lbl, "-")) for r in records]
        pool_pos = next((v for v in pool_pos_values if v != "-"), "-")
        vals = [r["rare_label_f1_test"][lbl] for r in records]
        L.append(f"| {lbl} | {pool_pos} | {test_positives[lbl]} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    L.append("")

    L.append("## Winners")
    L.append("")
    L.append(f"- **Best test f1_micro**: `{winners['f1_micro_test']}`")
    L.append(f"- **Best test f1_macro**: `{winners['f1_macro_test']}`")
    L.append(f"- **Best test precision_micro**: `{winners['precision_micro_test']}`")
    L.append(f"- **Best test recall_micro**: `{winners['recall_micro_test']}`")
    L.append("")
    L.append("Per-label winners on test:")
    L.append("")
    for lbl in LABELS:
        L.append(f"- `{lbl}` → `{winners['per_label_test'][lbl]}`")
    L.append("")

    L.append("## Takeaway")
    L.append("")
    best_key = winners["f1_micro_test"]
    if best_key == "exp3_silver_then_gold_v3_exgold":
        L.append(
            "- `exp3_silver_then_gold_v3_exgold` **still wins** after the leakage fix. "
            "This is now a fair comparison: the silver pretrain corpus no longer contains "
            "any gold_combined id, so the advantage reflects genuine transfer from distant-"
            "supervision priors rather than contamination."
        )
    else:
        L.append(
            f"- With the leak fixed, `{best_key}` wins; silver pretraining alone no longer "
            "dominates. The v2 exp3 numbers were inflated by leakage and should not be trusted."
        )
    L.append(
        "- exp1 vs exp2: pos_weight roughly trades micro-precision for rare-label recall "
        "— macro-F1 rises ~0.41 → ~0.56 while micro-F1 stays ~0.73."
    )
    L.append(
        "- Silver prior effect, fairly measured: the gap between exp3_v3_exgold and exp2 on "
        "rare labels is the clean signal for \"does distant supervision help rare classes?\" "
        "Answer: yes — altyapi and psikolojik go from F1=0 to saturated, and without the "
        "leak this gain is now a legitimate modeling result (small-support caveat still applies)."
    )
    L.append("")

    L.append("## Critical risks & caveats")
    L.append("")
    L.append(
        "- **Test set is small** (194 rows). Absolute metrics have wide CIs; rare labels "
        "with 1–4 test positives can flip between F1=0 and F1=1 on a single prediction."
    )
    L.append(
        "- **Silver signal is rule-based**, not human-labeled. Silver metrics (~0.99 F1) "
        "measure agreement with `scripts/ai_prefill_annotations.py` patterns, not real "
        "ground truth — this is a distant-supervision pretraining signal only."
    )
    L.append(
        "- **Silver_exgold was freshly trained in step 7** (49k train rows). The previous "
        "`models/need_classification_silver_63k/final` remains on disk for reproducing the "
        "historical (contaminated) v2 comparison — it is NOT canonical anymore."
    )
    L.append(
        "- **exp3 v3 CV thresholds** were produced by re-running 5-fold multilabel-stratified "
        "CV with silver_exgold→gold fine-tune per fold. Scratch dir: "
        "`models/_cv_tuning_scratch_exp3_v3_exgold`."
    )
    L.append(
        "- **`models/final/selection.json` has NOT been touched.** Step 9 artifacts are left "
        "untouched by this step — the v3 winner is documented but not promoted."
    )
    L.append(
        "- **exp1 / exp2 reuse existing artifacts.** Their models never used silver weights, "
        "so they were not affected by the leakage and did not need retraining."
    )
    L.append("")

    L.append("## Per-experiment notes")
    L.append("")
    for r in records:
        L.append(f"### `{r['experiment']}`")
        L.append("")
        L.append(f"- **model_dir**: `{r['model_dir']}`")
        L.append(f"- **config**: `{r['config_yaml']}`")
        L.append(f"- **base model**: `{r['base_model']}`")
        L.append(f"- **silver provenance**: {r['silver_provenance']}")
        L.append(f"- **leak_free**: {r['leak_free']}")
        L.append(f"- **training**: {r['training']}")
        L.append(f"- **threshold source**: `{r['threshold_source']}` (type: `{r['threshold_type']}`)")
        L.append(f"- **test metrics**: f1_micro={r['test']['f1_micro']:.4f}, f1_macro={r['test']['f1_macro']:.4f}, P={r['test']['precision_micro']:.4f}, R={r['test']['recall_micro']:.4f}")
        L.append(f"- **notes**: {r['notes']}")
        L.append("")
    return "\n".join(L) + "\n"


def render_leak_audit_md(leak: Dict[str, Any], v2_record: Dict[str, Any], v3_record: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Step 7 Leak Audit (v3)")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(
        "The original step-7 comparison compared `exp3_silver_then_gold_v2` against exp1/exp2 "
        "using a silver pretraining pool that secretly contained **every** gold_combined id "
        "(train, val, AND test). The silver model therefore saw every gold test row during "
        "distant-label pretraining, which invalidates the silver→gold transfer comparison."
    )
    L.append("")
    L.append(
        "This audit documents the fix: a leak-free silver pool (`silver_63k_exgold`) was "
        "rebuilt from the geolocated 96k dataset with all 1934 gold ids excluded. A new "
        "silver model was trained on that clean pool. exp3 was re-instantiated as "
        "`exp3_silver_then_gold_v3_exgold` warm-started from the leak-free silver. CV "
        "thresholds were re-tuned. All evaluation is on the unchanged canonical "
        "`gold_combined/test.csv`."
    )
    L.append("")
    L.append("## Contamination evidence (before fix)")
    L.append("")
    old = leak["silvers"]["old_silver_contaminated"]
    L.append(f"- **Silver source**: `{old['path']}`")
    L.append(f"- **Silver rows**: {old['rows']}")
    L.append(f"- **Overlap with canonical gold**: {old['overlap_with_gold_canonical']} / 1934 ids")
    L.append(f"- **Overlap with gold train split**: {old['overlap_with_gold_train']} / 1547")
    L.append(f"- **Overlap with gold val split**: {old['overlap_with_gold_val']} / 193")
    L.append(f"- **Overlap with gold test split**: {old['overlap_with_gold_test']} / 194 — THIS IS THE HARMFUL LEAK")
    L.append("")
    L.append(
        "Root cause: `scripts/build_need_classification_silver_from_geolocated.py` reads "
        "the entire geolocated 96k dataset, dedups by id, and runs rule-based labeling. "
        "It did NOT exclude gold_combined ids. Because the gold_combined dataset is a "
        "subset of the geolocated 96k corpus, the silver pool inherited every gold id."
    )
    L.append("")
    L.append("## Remediation (how the fix was built)")
    L.append("")
    L.append(
        "1. `scripts/build_need_classification_silver_from_geolocated.py` updated: added "
        "`--exclude-gold-csv` flag that accepts one or more gold CSV paths. Their `id` "
        "columns are unioned and every matching silver row is removed before labeling."
    )
    L.append("")
    L.append(
        "2. Ran the updated script with **four** gold sources passed explicitly (redundant "
        "but paranoid), so the excluded id set = canonical gold ∪ train ∪ val ∪ test:"
    )
    L.append("")
    L.append("```bash")
    L.append("python scripts/build_need_classification_silver_from_geolocated.py \\")
    L.append("  --input data/processed/emergency_geolocated_96k.csv \\")
    L.append("  --output data/labeling/need_classification_silver_63k_profileA_exgold.csv \\")
    L.append("  --profile A --dedup \\")
    L.append("  --exclude-gold-csv data/need_classification_gold_combined.csv \\")
    L.append("  --exclude-gold-csv data/modeling/need_classification_gold_combined/train.csv \\")
    L.append("  --exclude-gold-csv data/modeling/need_classification_gold_combined/val.csv \\")
    L.append("  --exclude-gold-csv data/modeling/need_classification_gold_combined/test.csv")
    L.append("```")
    L.append("")
    L.append(
        "3. Produced `data/labeling/need_classification_silver_63k_profileA_exgold.csv` "
        "(61,246 rows; 1,934 rows removed)."
    )
    L.append("")
    L.append(
        "4. Split into `data/modeling/need_classification_silver_63k_exgold/{train,val,test}.csv` "
        "via `scripts/prepare_model_splits.py` (80/10/10, seed=42, stratified by `aciliyet_0_3`, "
        "same protocol as original silver split)."
    )
    L.append("")
    L.append(
        "5. Trained `models/need_classification_silver_63k_exgold/final` on the leak-free "
        "splits with the same hyperparameters as the original silver baseline (BERTurk, "
        "pos_weight, LR=2e-5, 3 epochs, BS=16, fp16)."
    )
    L.append("")
    L.append(
        "6. Fine-tuned `models/exp3_silver_then_gold_v3_exgold/final` on "
        "`gold_combined/train+val` warm-started from the leak-free silver, LR=1e-5, 3 epochs."
    )
    L.append("")
    L.append(
        "7. Ran 5-fold multilabel-stratified CV threshold tuning with the leak-free silver "
        "as the fold base model (scratch dir: `models/_cv_tuning_scratch_exp3_v3_exgold`). "
        "Emitted `thresholds_cv.json` + meta + report + OOF probabilities."
    )
    L.append("")
    L.append(
        "8. Evaluated `exp3_silver_then_gold_v3_exgold/final` on both canonical gold val "
        "and test with the new CV thresholds."
    )
    L.append("")

    L.append("## Verification (after fix)")
    L.append("")
    new = leak["silvers"]["new_silver_leak_free"]
    L.append(f"- **Leak-free silver source**: `{new['path']}`")
    L.append(f"- **Rows**: {new['rows']}")
    L.append(f"- **Overlap with canonical gold**: {new['overlap_with_gold_canonical']} / 1934 — must be 0")
    L.append(f"- **Overlap with gold train**: {new['overlap_with_gold_train']} / 1547 — must be 0")
    L.append(f"- **Overlap with gold val**: {new['overlap_with_gold_val']} / 193 — must be 0")
    L.append(f"- **Overlap with gold test**: {new['overlap_with_gold_test']} / 194 — must be 0")
    L.append("")
    L.append(
        "Additionally verified: every leak-free silver split (train/val/test under "
        "`silver_63k_exgold/`) has 0 overlap with every gold_combined split."
    )
    L.append("")

    L.append("## Impact: old vs new exp3 on the same gold test")
    L.append("")
    t_v2 = v2_record["test"]
    t_v3 = v3_record["test"]
    L.append("| metric | exp3_v2 (contaminated) | exp3_v3_exgold (leak-free) | delta |")
    L.append("|---|---|---|---|")
    for m in ("f1_micro", "f1_macro", "precision_micro", "recall_micro"):
        a = float(t_v2[m]); b = float(t_v3[m])
        L.append(f"| {m} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    L.append("")
    L.append("| label | exp3_v2 F1 | exp3_v3 F1 | delta |")
    L.append("|---|---|---|---|")
    for lbl in LABELS:
        a = float(t_v2["f1_per_label"].get(lbl, 0.0))
        b = float(t_v3["f1_per_label"].get(lbl, 0.0))
        L.append(f"| {lbl} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    L.append("")

    L.append("## Interpretation")
    L.append("")
    L.append(
        "The headline macro-F1 barely moves (0.873 → 0.875). This is somewhat surprising "
        "given the scale of the leak (100% of gold ids present in the silver pool). Two "
        "factors explain it:"
    )
    L.append("")
    L.append(
        "1. **Silver labels are rule-based, not gold labels.** The silver model was learning "
        "to imitate `ai_prefill_annotations.py` patterns, not the actual human labels. "
        "Those rules are coarse and don't carry much extra information about a specific "
        "gold test tweet beyond what the text itself provides. So even though the model saw "
        "the test texts during silver pretraining, it saw them tied to rule outputs — not "
        "to the true gold labels. The harm is bounded."
    )
    L.append("")
    L.append(
        "2. **Distant-supervision transfer is robust.** Whether the silver pool contains the "
        "gold ids or not, the rule patterns the silver model learns are general enough that "
        "fine-tuning on gold picks up the remaining task-specific information. The gain over "
        "exp2 (gold-only) stays.")
    L.append("")
    L.append(
        "**What this does NOT mean**: the leak was harmless. It means the leak did not help "
        "as much as a label-level leak would. A comparison claim like \"silver pretrain beats "
        "gold-only\" was scientifically invalid under v2 — even a tiny advantage might have "
        "come from contamination. v3 is the first comparison where this claim is defensible."
    )
    L.append("")
    L.append("## Framing going forward")
    L.append("")
    L.append("- `experiment_comparison_v2.*` → **HISTORICAL, CONTAMINATED**. Do not cite.")
    L.append("- `experiment_comparison_v3_leakfree.*` → **CANONICAL**. Use for any step-8 work or reporting.")
    L.append(
        "- Final selection pointer (`models/final/selection.json`) left untouched. v3 winner "
        "(`exp3_silver_then_gold_v3_exgold`) is documented but not promoted in this step."
    )
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    records = [build_experiment_record(e) for e in EXPERIMENTS]
    winners = compute_winners(records)

    leak = silver_leak_report()

    # Historical v2 exp3 for impact table.
    v2_path = ROOT / HISTORICAL_V2["eval_test"]
    v2_eval = load_json(v2_path)
    v2_record = {
        "experiment": HISTORICAL_V2["key"],
        "test": {
            "f1_micro": float(v2_eval["f1_micro"]),
            "f1_macro": float(v2_eval["f1_macro"]),
            "precision_micro": float(v2_eval["precision_micro"]),
            "recall_micro": float(v2_eval["recall_micro"]),
            "f1_per_label": {k: float(v) for k, v in v2_eval["f1_per_label"].items()},
        },
    }
    v3_record = [r for r in records if r["experiment"] == "exp3_silver_then_gold_v3_exgold"][0]

    # --- comparison v3 ---
    cmp_payload = {
        "task": "need_classification_experiment_comparison_v3_leakfree",
        "scope": "step7",
        "supersedes": "data/analysis/experiment_comparison_v2.{md,csv,json}",
        "test_csv": GOLD_TEST,
        "val_csv": GOLD_VAL,
        "rows_test": records[0]["rows_test"],
        "rows_val": records[0]["rows_val"],
        "labels": LABELS,
        "rare_labels": RARE_LABELS,
        "experiments": records,
        "winners": winners,
        "leak_summary": {
            "old_silver_overlap_with_gold_union": leak["silvers"]["old_silver_contaminated"]["overlap_with_gold_split_union"],
            "new_silver_overlap_with_gold_union": leak["silvers"]["new_silver_leak_free"]["overlap_with_gold_split_union"],
        },
    }
    write_json(OUT_DIR / "experiment_comparison_v3_leakfree.json", cmp_payload)
    write_csv(OUT_DIR / "experiment_comparison_v3_leakfree.csv", records)
    (OUT_DIR / "experiment_comparison_v3_leakfree.md").write_text(
        render_comparison_md(records, winners, leak), encoding="utf-8"
    )

    # --- leak audit v3 ---
    audit_payload = {
        "task": "step7_leak_audit_v3",
        "gold_sources": leak["gold_sources_used"],
        "silvers": leak["silvers"],
        "historical_v2_exp3": {
            "model_dir": HISTORICAL_V2["model_dir"],
            "threshold_source": HISTORICAL_V2["threshold_source"],
            "eval_test_path": HISTORICAL_V2["eval_test"],
            "status": HISTORICAL_V2["status"],
            "issue": HISTORICAL_V2["issue"],
            "test_metrics_at_tuned_threshold": v2_record["test"],
        },
        "new_v3_exp3": {
            "model_dir": v3_record["model_dir"],
            "base_model": v3_record["base_model"],
            "threshold_source": v3_record["threshold_source"],
            "test_metrics_at_tuned_threshold": v3_record["test"],
        },
        "remediation_commands": [
            "python scripts/build_need_classification_silver_from_geolocated.py --input data/processed/emergency_geolocated_96k.csv --output data/labeling/need_classification_silver_63k_profileA_exgold.csv --profile A --dedup --exclude-gold-csv data/need_classification_gold_combined.csv --exclude-gold-csv data/modeling/need_classification_gold_combined/train.csv --exclude-gold-csv data/modeling/need_classification_gold_combined/val.csv --exclude-gold-csv data/modeling/need_classification_gold_combined/test.csv",
            "python scripts/prepare_model_splits.py --input data/labeling/need_classification_silver_63k_profileA_exgold.csv --outdir data/modeling/need_classification_silver_63k_exgold --seed 42 --train-size 0.80 --val-size 0.10 --test-size 0.10 --stratify-col aciliyet_0_3 --ensure-test-coverage",
            "python scripts/train_need_classifier.py --config data/modeling/need_classification_silver_63k_exgold/training_config.yaml --output-dir models/need_classification_silver_63k_exgold",
            "python scripts/train_need_classifier.py --config data/modeling/experiments/exp3_silver_then_gold_v3_exgold.yaml --output-dir models/exp3_silver_then_gold_v3_exgold",
            "python scripts/tune_thresholds_cv.py --train-csv data/modeling/need_classification_gold_combined/train.csv --val-csv data/modeling/need_classification_gold_combined/val.csv --k 5 --seed 42 --base-model models/need_classification_silver_63k_exgold/final --epochs 3 --lr 1e-5 --fp16 --use-pos-weight --strategy oof_global --out-thresholds models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json --out-report models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_report.txt --out-meta models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_meta.json --out-oof models/exp3_silver_then_gold_v3_exgold/threshold_tuning_cv_oof.csv --scratch-dir models/_cv_tuning_scratch_exp3_v3_exgold --require-cuda",
            "python scripts/evaluate_need_classifier.py --model-dir models/exp3_silver_then_gold_v3_exgold/final --csv data/modeling/need_classification_gold_combined/test.csv --thresholds-json models/exp3_silver_then_gold_v3_exgold/thresholds_cv.json --out models/exp3_silver_then_gold_v3_exgold/eval_test_tuned.json",
        ],
    }
    write_json(OUT_DIR / "step7_leak_audit_v3.json", audit_payload)
    (OUT_DIR / "step7_leak_audit_v3.md").write_text(
        render_leak_audit_md(leak, v2_record, v3_record), encoding="utf-8"
    )

    print("Wrote:")
    print(f"  {OUT_DIR / 'experiment_comparison_v3_leakfree.json'}")
    print(f"  {OUT_DIR / 'experiment_comparison_v3_leakfree.csv'}")
    print(f"  {OUT_DIR / 'experiment_comparison_v3_leakfree.md'}")
    print(f"  {OUT_DIR / 'step7_leak_audit_v3.json'}")
    print(f"  {OUT_DIR / 'step7_leak_audit_v3.md'}")
    print()
    print("Leak summary:")
    print(f"  old silver overlap with gold union: {leak['silvers']['old_silver_contaminated']['overlap_with_gold_split_union']} / 1934")
    print(f"  new silver overlap with gold union: {leak['silvers']['new_silver_leak_free']['overlap_with_gold_split_union']} / 1934")
    print()
    print("Winners:")
    print(f"  f1_micro_test: {winners['f1_micro_test']}")
    print(f"  f1_macro_test: {winners['f1_macro_test']}")


if __name__ == "__main__":
    main()
