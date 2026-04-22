#!/usr/bin/env python3
"""
Step-8 error analysis for the LEAK-FREE canonical winner.

Canonical winner (from step 7 v3 leak-free comparison):
    models/exp3_silver_then_gold_v3_exgold/final

This script produces:
  - data/analysis/error_analysis_v2_leakfree.md
  - data/analysis/error_analysis_v2_leakfree.json
  - data/analysis/error_analysis_v2_leakfree.fp.csv
  - data/analysis/error_analysis_v2_leakfree.fn.csv
  - data/analysis/error_analysis_v2_leakfree.slices.csv

Design notes:
  - Deterministic: fixed seeds, torch.use_deterministic_algorithms(True) where safe,
    stable sort orders for example ranking.
  - GPU required: aborts if CUDA is not available (no CPU fallback).
  - UTF-8 output with explicit encoding to preserve Turkish diacritics.
  - Reads thresholds from the canonical CV artifact (thresholds_cv.json).
  - Analyses: per-label confusion, FP/FN examples, confusion pairs (what else fires
    when a label is FN'd?), co-occurrence (gold-level pair statistics), slices by
    aciliyet_0_3 / text length / multi-label count / heuristic tweet type, pattern
    observations, and an honest leak-free caveat block.

Usage:
    python scripts/error_analysis_v2_leakfree.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "final"
DEFAULT_TEST_CSV = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "test.csv"
DEFAULT_THRESHOLDS = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "thresholds_cv.json"
DEFAULT_LABELS = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "label_columns.json"
DEFAULT_COMPARISON_SRC = REPO_ROOT / "data" / "analysis" / "experiment_comparison_v3_leakfree.json"

OUT_DIR = REPO_ROOT / "data" / "analysis"
OUT_MD = OUT_DIR / "error_analysis_v2_leakfree.md"
OUT_JSON = OUT_DIR / "error_analysis_v2_leakfree.json"
OUT_FP_CSV = OUT_DIR / "error_analysis_v2_leakfree.fp.csv"
OUT_FN_CSV = OUT_DIR / "error_analysis_v2_leakfree.fn.csv"
OUT_SLICE_CSV = OUT_DIR / "error_analysis_v2_leakfree.slices.csv"

SEED = 42
MAX_LENGTH = 192
BATCH_SIZE = 32
TOP_N = 15


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _set_deterministic() -> None:
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this step. Aborting (no CPU fallback).")
    name = torch.cuda.get_device_name(0)
    return name


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


def _predict_probs(
    model_dir: Path,
    texts: List[str],
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda")
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1}


def _micro_macro(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1_micro = (2 * p * r / (p + r)) if (p + r) else 0.0

    per_f1 = []
    for j in range(y_true.shape[1]):
        c = _confusion(y_true[:, j], y_pred[:, j])
        per_f1.append(c["f1"])
    f1_macro = float(np.mean(per_f1)) if per_f1 else 0.0
    return {"precision_micro": p, "recall_micro": r, "f1_micro": f1_micro, "f1_macro": f1_macro}


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------


def _length_bucket(n: int) -> str:
    if n < 60:
        return "short(<60)"
    if n < 140:
        return "medium(60-139)"
    return "long(>=140)"


CALL_PAT = re.compile(
    r"(yard[ıi]m\s*ed[iı]n|kurtar[ıi]n|ac[ıi]l|imdat|hala\s+ulas\w+|enkaz|l[uü]tfen|y[aâ]lvar|duy\w*\s+var\s+m[iı])",
    flags=re.IGNORECASE,
)
ANNOUNCE_PAT = re.compile(
    r"(da[gğ][ıi]t[ıi]ld|topl[aı]n\w+|kuruld|a[çc][ıi]ld\w+|duyur\w+|ula[şs]t[ıi]r[ıi]ld|temin\s+edildi)",
    flags=re.IGNORECASE,
)
INFO_Q_PAT = re.compile(
    r"(nerede|nas[ıi]l|kim\s|niye|neden|\?|durumu\s+ne|hangi\s+yerde)",
    flags=re.IGNORECASE,
)
STATUS_PAT = re.compile(
    r"(y[ıi]k[ıi]ld|[çc][oö]kt[uü]|hasar|kes[iı]ldi|yok\s+|olmad[ıi]|y[oa]k\s*[,\.]|kapand[ıi])",
    flags=re.IGNORECASE,
)


def _tweet_type(text: str) -> str:
    """Heuristic buckets for qualitative slice analysis only."""
    t = text or ""
    if CALL_PAT.search(t):
        return "call_for_help"
    if ANNOUNCE_PAT.search(t):
        return "announcement"
    if INFO_Q_PAT.search(t):
        return "info_request"
    if STATUS_PAT.search(t):
        return "status_report"
    return "other"


def _multilabel_count_bucket(k: int) -> str:
    if k == 0:
        return "0"
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def _slice_any_error(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    extra: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    any_err = (y_true != y_pred).any(axis=1).astype(int)
    working = df.copy()
    if extra is not None:
        working[group_col] = extra.values
    working["_err"] = any_err
    rows: List[Dict[str, Any]] = []
    for g, sub in working.groupby(group_col, dropna=False):
        n = len(sub)
        e = int(sub["_err"].sum())
        rows.append(
            {
                "slice_col": group_col,
                "group": str(g),
                "rows": n,
                "rows_with_any_error": e,
                "error_rate": e / n if n else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["error_rate"], -r["rows"]))
    return rows


# ---------------------------------------------------------------------------
# Pair analyses
# ---------------------------------------------------------------------------


def _fn_cofire(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Dict[str, int]]:
    """For each label A, count how many of A's FN rows had label B predicted=1."""
    out: Dict[str, Dict[str, int]] = {}
    for i, la in enumerate(labels):
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        pair_counts: Dict[str, int] = {}
        if fn_mask.any():
            other_fire = y_pred[fn_mask].sum(axis=0)
            for j, lb in enumerate(labels):
                if j == i:
                    continue
                c = int(other_fire[j])
                if c > 0:
                    pair_counts[lb] = c
        out[la] = dict(sorted(pair_counts.items(), key=lambda kv: -kv[1]))
    return out


def _fp_cofire(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Dict[str, int]]:
    """For each label A, count how many of A's FP rows had label B truly=1."""
    out: Dict[str, Dict[str, int]] = {}
    for i, la in enumerate(labels):
        fp_mask = (y_true[:, i] == 0) & (y_pred[:, i] == 1)
        pair_counts: Dict[str, int] = {}
        if fp_mask.any():
            other_true = y_true[fp_mask].sum(axis=0)
            for j, lb in enumerate(labels):
                if j == i:
                    continue
                c = int(other_true[j])
                if c > 0:
                    pair_counts[lb] = c
        out[la] = dict(sorted(pair_counts.items(), key=lambda kv: -kv[1]))
    return out


def _gold_cooccurrence(y_true: np.ndarray, labels: List[str]) -> List[Dict[str, Any]]:
    """Gold-level pair frequencies: how often A and B both =1."""
    pairs: List[Dict[str, Any]] = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j <= i:
                continue
            c = int(((y_true[:, i] == 1) & (y_true[:, j] == 1)).sum())
            if c > 0:
                pairs.append({"a": la, "b": lb, "count": c})
    pairs.sort(key=lambda r: -r["count"])
    return pairs


def _both_true_one_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> List[Dict[str, Any]]:
    """For each gold pair (A,B), count how often model got only ONE of the two."""
    out: List[Dict[str, Any]] = []
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j <= i:
                continue
            both_true = (y_true[:, i] == 1) & (y_true[:, j] == 1)
            if not both_true.any():
                continue
            only_a = int((both_true & (y_pred[:, i] == 1) & (y_pred[:, j] == 0)).sum())
            only_b = int((both_true & (y_pred[:, i] == 0) & (y_pred[:, j] == 1)).sum())
            both = int((both_true & (y_pred[:, i] == 1) & (y_pred[:, j] == 1)).sum())
            neither = int((both_true & (y_pred[:, i] == 0) & (y_pred[:, j] == 0)).sum())
            out.append(
                {
                    "a": la,
                    "b": lb,
                    "gold_both": int(both_true.sum()),
                    "pred_both": both,
                    "pred_only_a": only_a,
                    "pred_only_b": only_b,
                    "pred_neither": neither,
                }
            )
    out.sort(key=lambda r: -(r["pred_only_a"] + r["pred_only_b"] + r["pred_neither"]))
    return out


# ---------------------------------------------------------------------------
# Example pickers + CSV export
# ---------------------------------------------------------------------------


def _pick_examples(
    df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    kind: str,
    top_n: int,
    text_col: str,
    thresholds: Dict[str, float],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, lab in enumerate(labels):
        if kind == "fp":
            mask = (y_true[:, i] == 0) & (y_pred[:, i] == 1)
            order = np.argsort(-probs[:, i], kind="stable")
        else:
            mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
            order = np.argsort(probs[:, i], kind="stable")
        idxs = [int(k) for k in order if bool(mask[int(k)])][:top_n]
        rows: List[Dict[str, Any]] = []
        for k in idxs:
            row = {
                "kind": kind,
                "label": lab,
                "threshold": float(thresholds.get(lab, 0.5)),
                "prob": float(probs[k, i]),
                "gold_labels": [labels[j] for j in range(len(labels)) if y_true[k, j] == 1],
                "pred_labels": [labels[j] for j in range(len(labels)) if y_pred[k, j] == 1],
                "text": str(df.iloc[k].get(text_col, "")),
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


def _examples_to_rows(
    examples: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lab, xs in examples.items():
        for r in xs:
            rows.append(
                {
                    "kind": r["kind"],
                    "label": lab,
                    "threshold": r["threshold"],
                    "prob": r["prob"],
                    "id": r.get("id", ""),
                    "aciliyet_0_3": r.get("aciliyet_0_3", ""),
                    "gold_labels": ",".join(r["gold_labels"]),
                    "pred_labels": ",".join(r["pred_labels"]),
                    "text": (r["text"] or "").replace("\r", " ").replace("\n", " "),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pattern observations (deterministic, signal-driven)
# ---------------------------------------------------------------------------


def _pattern_observations(
    per_label_conf: Dict[str, Dict[str, Any]],
    fp_cofire: Dict[str, Dict[str, int]],
    fn_cofire: Dict[str, Dict[str, int]],
    thresholds: Dict[str, float],
    gold_totals: Dict[str, int],
    test_positives: Dict[str, int],
) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []

    # 1) Rare-label saturation warning.
    for lab in ["altyapi", "psikolojik", "guvenlik"]:
        c = per_label_conf.get(lab, {})
        pos = test_positives.get(lab, 0)
        if c and pos <= 5 and c["f1"] >= 0.9:
            obs.append(
                {
                    "topic": f"`{lab}`: ince destekte F1 ~1.0",
                    "detail": (
                        f"`{lab}` testte yalnızca {pos} pozitifle F1={c['f1']:.3f} veriyor "
                        f"(TP={c['tp']}, FP={c['fp']}, FN={c['fn']}). Tek tahmin bu sayıyı oynatabilir; "
                        "kalibre bir skor olarak değil, nitel sinyal olarak oku."
                    ),
                }
            )

    # 2) Threshold asymmetry.
    hi = {k: v for k, v in thresholds.items() if v >= 0.85}
    lo = {k: v for k, v in thresholds.items() if v <= 0.25}
    if hi:
        obs.append(
            {
                "topic": "Yüksek eşikler (≥0.85) — tutucu etiketler",
                "detail": (
                    "CV şu etiketlere çok yüksek eşik seçti: "
                    + ", ".join(f"`{k}`={v:.2f}" for k, v in sorted(hi.items(), key=lambda kv: -kv[1]))
                    + ". Bu etiketler yalnızca model çok eminken ateşliyor; bu precision'ı korur ama belirsiz "
                    "pozitifleri kaçırır (FN riski)."
                ),
            }
        )
    if lo:
        obs.append(
            {
                "topic": "Düşük eşikler (≤0.25) — agresif etiketler",
                "detail": (
                    "CV şu etiketlere çok düşük eşik seçti: "
                    + ", ".join(f"`{k}`={v:.2f}" for k, v in sorted(lo.items(), key=lambda kv: kv[1]))
                    + ". Bu etiketler zayıf sinyalde bile ateşliyor; recall yüksek, precision düşük (FP riski)."
                ),
            }
        )

    # 3) Dominant FN cofire (what fires while the gold label is missed).
    for lab, cof in fn_cofire.items():
        if not cof:
            continue
        top_lab, top_c = next(iter(cof.items()))
        fn_total = per_label_conf.get(lab, {}).get("fn", 0)
        if fn_total >= 3 and top_c / max(fn_total, 1) >= 0.4:
            obs.append(
                {
                    "topic": f"`{lab}` kaçırıldığında model sıkça `{top_lab}` diyor",
                    "detail": (
                        f"`{lab}` için {top_c}/{fn_total} FN satırında `{top_lab}` predicted=1 — "
                        "iki etiket arasında sınır bulanık."
                    ),
                }
            )

    # 4) Dominant FP cofire (spurious fires on rows that truly carry another need).
    for lab, cof in fp_cofire.items():
        if not cof:
            continue
        top_lab, top_c = next(iter(cof.items()))
        fp_total = per_label_conf.get(lab, {}).get("fp", 0)
        if fp_total >= 3 and top_c / max(fp_total, 1) >= 0.4:
            obs.append(
                {
                    "topic": f"`{lab}` yanlış ateşlediğinde gold'da sıkça `{top_lab}` var",
                    "detail": (
                        f"`{lab}` için {top_c}/{fp_total} FP satırında gerçek etiket `{top_lab}` idi — "
                        f"model `{top_lab}` benzeri dili `{lab}` olarak etiketliyor."
                    ),
                }
            )

    return obs


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _render_markdown(
    model_dir: str,
    test_csv: str,
    thresholds_src: str,
    comparison_src: str,
    gpu_name: str,
    rows: int,
    labels: List[str],
    thresholds: Dict[str, float],
    overall: Dict[str, float],
    per_label_conf: Dict[str, Dict[str, Any]],
    test_positives: Dict[str, int],
    gold_pool_positives: Optional[Dict[str, int]],
    fn_cofire: Dict[str, Dict[str, int]],
    fp_cofire: Dict[str, Dict[str, int]],
    gold_cooc: List[Dict[str, Any]],
    both_true_one_pred: List[Dict[str, Any]],
    slices: Dict[str, List[Dict[str, Any]]],
    fp_examples: Dict[str, List[Dict[str, Any]]],
    fn_examples: Dict[str, List[Dict[str, Any]]],
    observations: List[Dict[str, Any]],
    emphasis_labels: List[str],
    top_patterns: List[str],
    step9_suggestions: List[str],
) -> str:
    L: List[str] = []
    L.append("# Step 8 — Error Analysis (v2, leak-free)\n")
    L.append(
        "Canonical winner'ın ([step 7 v3](experiment_comparison_v3_leakfree.md)) "
        "leak-free test setinde hata örüntüleri. Model seçimi, tahmin üretimi ve "
        "final pointer güncelleme bu adımda yapılmaz.\n"
    )

    # 1. Kaynaklar
    L.append("## 1. Kullanılan kaynaklar\n")
    L.append(f"- **Model**: `{model_dir}`")
    L.append(f"- **Test CSV**: `{test_csv}` ({rows} satır)")
    L.append(f"- **Thresholds**: `{thresholds_src}` (type=`cv`, strategy=`oof_global`, k=5)")
    L.append(f"- **Comparison source**: `{comparison_src}`")
    L.append(f"- **GPU**: {gpu_name} (CUDA required; CPU fallback disabled)")
    L.append(f"- **Seed**: {SEED} — deterministic inference")
    L.append("")

    # 2. Genel metrik bağlamı
    L.append("## 2. Genel metrik bağlamı\n")
    L.append(
        f"- **f1_micro** = {overall['f1_micro']:.4f}, **f1_macro** = {overall['f1_macro']:.4f}, "
        f"P_micro = {overall['precision_micro']:.4f}, R_micro = {overall['recall_micro']:.4f}."
    )
    L.append("")
    L.append("### Per-label F1 özet\n")
    L.append("| label | thr | TP | FP | FN | TN | P | R | F1 | test_pozitif | pool_pozitif |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        c = per_label_conf[lab]
        pool = gold_pool_positives.get(lab, "-") if gold_pool_positives else "-"
        L.append(
            f"| `{lab}` | {thresholds.get(lab, 0.5):.2f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | "
            f"{c['precision']:.2f} | {c['recall']:.2f} | **{c['f1']:.3f}** | {test_positives.get(lab, 0)} | {pool} |"
        )
    L.append("")

    if emphasis_labels:
        L.append("### Vurgulanan etiketler\n")
        for lab in emphasis_labels:
            c = per_label_conf.get(lab, {})
            if not c:
                continue
            L.append(
                f"- **`{lab}`** → F1={c['f1']:.3f} (thr={thresholds.get(lab, 0.5):.2f}, TP={c['tp']}, "
                f"FP={c['fp']}, FN={c['fn']}, test pozitif={test_positives.get(lab, 0)})"
            )
        L.append("")

    # 3. Per-label error breakdown
    L.append("## 3. Per-label error breakdown\n")
    for lab in labels:
        c = per_label_conf[lab]
        L.append(f"### `{lab}` (thr={thresholds.get(lab, 0.5):.2f})\n")
        L.append(
            f"- TP={c['tp']}, FP={c['fp']}, FN={c['fn']}, TN={c['tn']} — "
            f"P={c['precision']:.3f}, R={c['recall']:.3f}, F1={c['f1']:.3f}"
        )
        fn_rows = fn_examples.get(lab, [])
        fp_rows = fp_examples.get(lab, [])
        if fn_rows:
            L.append("- **En kritik FN örnekleri** (en düşük prob):")
            for r in fn_rows[:5]:
                rid = r.get("id", "-")
                ac = r.get("aciliyet_0_3", "-")
                gold = ",".join(r["gold_labels"]) or "-"
                pred = ",".join(r["pred_labels"]) or "-"
                L.append(
                    f"  - `id={rid}` p={r['prob']:.3f} aciliyet={ac} — gold=`{gold}` | pred=`{pred}`"
                )
                L.append(f"    - {_md_escape(r['text'])[:280]}")
        else:
            L.append("- FN yok.")
        if fp_rows:
            L.append("- **En kritik FP örnekleri** (en yüksek prob):")
            for r in fp_rows[:5]:
                rid = r.get("id", "-")
                ac = r.get("aciliyet_0_3", "-")
                gold = ",".join(r["gold_labels"]) or "-"
                pred = ",".join(r["pred_labels"]) or "-"
                L.append(
                    f"  - `id={rid}` p={r['prob']:.3f} aciliyet={ac} — gold=`{gold}` | pred=`{pred}`"
                )
                L.append(f"    - {_md_escape(r['text'])[:280]}")
        else:
            L.append("- FP yok.")
        L.append("")

    # 4. Pattern analysis
    L.append("## 4. Pattern analysis (örüntü yorumu)\n")
    if observations:
        for o in observations:
            L.append(f"- **{o['topic']}** — {o['detail']}")
    else:
        L.append("- (deterministic pattern extractor bir şey yakalamadı)")
    L.append("")

    # 5. Confusion / co-occurrence
    L.append("## 5. Confusion & co-occurrence\n")
    L.append("### FN cofire (etiket kaçırılırken hangi başka etiket ateşledi?)\n")
    for lab in labels:
        cof = fn_cofire.get(lab, {})
        if not cof:
            L.append(f"- `{lab}`: (FN yok veya cofire yok)")
        else:
            top = list(cof.items())[:5]
            L.append(f"- `{lab}`: " + ", ".join(f"`{k}`×{v}" for k, v in top))
    L.append("")
    L.append("### FP cofire (yanlış ateş eden etiketin satırında gerçekten hangi etiketler vardı?)\n")
    for lab in labels:
        cof = fp_cofire.get(lab, {})
        if not cof:
            L.append(f"- `{lab}`: (FP yok veya cofire yok)")
        else:
            top = list(cof.items())[:5]
            L.append(f"- `{lab}`: " + ", ".join(f"`{k}`×{v}" for k, v in top))
    L.append("")
    if gold_cooc:
        L.append("### Gold-level co-occurrence (en sık birlikte gelen etiket çiftleri)\n")
        L.append("| a | b | count |")
        L.append("|---|---|---|")
        for p in gold_cooc[:10]:
            L.append(f"| `{p['a']}` | `{p['b']}` | {p['count']} |")
        L.append("")
    if both_true_one_pred:
        L.append("### Multi-label: her iki etiket gold'da var iken modelin tahmini\n")
        L.append("| a | b | gold_ikisi | pred_ikisi | sadece_a | sadece_b | hiçbiri |")
        L.append("|---|---|---|---|---|---|---|")
        for p in both_true_one_pred[:10]:
            L.append(
                f"| `{p['a']}` | `{p['b']}` | {p['gold_both']} | {p['pred_both']} | "
                f"{p['pred_only_a']} | {p['pred_only_b']} | {p['pred_neither']} |"
            )
        L.append("")

    # 6. Slice analysis
    L.append("## 6. Slice analysis\n")
    for title, slice_rows in slices.items():
        if not slice_rows:
            continue
        L.append(f"### {title}\n")
        L.append("| group | rows | with_any_error | error_rate |")
        L.append("|---|---|---|---|")
        for r in slice_rows:
            L.append(f"| {r['group']} | {r['rows']} | {r['rows_with_any_error']} | {r['error_rate']:.3f} |")
        L.append("")

    # 7. Leak-free caveat
    L.append("## 7. Leak-free caveat\n")
    L.append(
        "- Step 7 v3'te id-level silver→gold leakage kapatıldı (canonical gold'un 1934 id'sinin tamamı silver havuzundan çıkarıldı).\n"
        "- **Ancak content-level risk hâlâ tamamen elenmiş değildir.** Silver kaynağı "
        "(`data/processed/emergency_geolocated_96k.csv`), gold ile aynı tweet havuzundan türedi; "
        "gold id'leri çıkarılsa bile, retweet / alıntı / near-duplicate metinler silver'da kalmış olabilir.\n"
        "- Bu nedenle özellikle F1=1.0 saturasyonları (ör. `altyapi`, `psikolojik`) gerçek genelleme değil, "
        "**dar desen ezberi** sonucu da olabilir. Başarı örneklerini yorumlarken bu sınırı unutmayın.\n"
    )

    # 8. Son bölüm
    L.append("## 8. Son bölüm\n")
    L.append("### Modelin en çok zorlandığı 5 örüntü\n")
    for i, p in enumerate(top_patterns[:5], start=1):
        L.append(f"{i}. {p}")
    L.append("")
    L.append("### Step 9 için öneriler (bu adımda uygulanmaz)\n")
    for s in step9_suggestions:
        L.append(f"- {s}")
    L.append("")
    L.append(
        "> Bu adımda model seçimi / tahmin üretimi / `models/final/selection.json` güncellemesi yapılmadı. "
        "Step 9 kararları için bu rapor girdi; aksiyonu onay sonrası.\n"
    )
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Top-patterns & step-9 suggestions (data-driven but deterministic text)
# ---------------------------------------------------------------------------


def _top5_patterns(
    per_label_conf: Dict[str, Dict[str, Any]],
    fn_cofire: Dict[str, Dict[str, int]],
    fp_cofire: Dict[str, Dict[str, int]],
    thresholds: Dict[str, float],
    test_positives: Dict[str, int],
) -> List[str]:
    patterns: List[Tuple[float, str]] = []

    for lab, c in per_label_conf.items():
        pool_pos = test_positives.get(lab, 0)
        cof = fn_cofire.get(lab, {})
        if c["fn"] >= 3 and cof:
            top_lab, top_c = next(iter(cof.items()))
            if top_c / c["fn"] >= 0.4:
                patterns.append(
                    (
                        c["fn"],
                        f"`{lab}` kaçırıldığında modelin en sık önerdiği etiket `{top_lab}` — "
                        f"{top_c}/{c['fn']} FN satırında `{top_lab}` ateşledi; bu iki etiket arasında sınır bulanık.",
                    )
                )

    for lab, c in per_label_conf.items():
        cof = fp_cofire.get(lab, {})
        if c["fp"] >= 3 and cof:
            top_lab, top_c = next(iter(cof.items()))
            if top_c / c["fp"] >= 0.4:
                patterns.append(
                    (
                        c["fp"],
                        f"`{lab}` yanlış ateşlediğinde gold'da en sık `{top_lab}` bulunuyor — "
                        f"{top_c}/{c['fp']} FP satırında `{top_lab}` gerçek pozitifti; model `{top_lab}`-benzeri dili `{lab}` olarak etiketliyor.",
                    )
                )

    for lab, thr in thresholds.items():
        c = per_label_conf.get(lab, {})
        if thr >= 0.85 and c.get("fn", 0) >= 3 and c.get("recall", 0.0) < 0.8:
            patterns.append(
                (
                    c["fn"],
                    f"`{lab}` için CV eşiği çok yüksek (thr={thr:.2f}); recall={c['recall']:.2f} ile FN={c['fn']} — "
                    "threshold kalibrasyonu tutucu, eşik düşürmek FN'i azaltabilir ama FP riski var.",
                )
            )

    for lab in ["altyapi", "psikolojik"]:
        c = per_label_conf.get(lab, {})
        pos = test_positives.get(lab, 0)
        if c and pos <= 5 and c["f1"] >= 0.9:
            patterns.append(
                (
                    0.5,
                    f"`{lab}` F1={c['f1']:.3f} ama test pozitif sayısı yalnızca {pos}; "
                    "tek tahmin F1'i dramatik oynatır, bu yüksek skoru kalibre bir başarı olarak okuma.",
                )
            )

    patterns.sort(key=lambda t: -t[0])
    seen = set()
    out: List[str] = []
    for _, txt in patterns:
        if txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out[:8]


def _step9_suggestions(
    per_label_conf: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, float],
    test_positives: Dict[str, int],
) -> List[str]:
    tips: List[str] = []

    tips.append(
        "Step 9 (seçim/çıkarım) öncesinde, `bilgi_paylasimi` ve `guvenlik` için eşik duyarlılığını sensitivity-plot ile dökümante et "
        "(CV thresholds sabit kalsa da production'da neyi feda ettiğimizi bilelim)."
    )
    for lab, c in per_label_conf.items():
        thr = thresholds.get(lab, 0.5)
        if c["fn"] >= 3 and c["recall"] < 0.8 and thr >= 0.85:
            tips.append(
                f"`{lab}`: eşik {thr:.2f} ile recall={c['recall']:.2f} — step 9'da ayrı bir `threshold_production.json` "
                "üretip recall-öncelikli senaryoda daha düşük eşik dene (ama bu scoreboard'u değiştirmez)."
            )
    rare_hit = [l for l in ("altyapi", "psikolojik") if test_positives.get(l, 0) <= 3 and per_label_conf.get(l, {}).get("f1", 0) >= 0.9]
    if rare_hit:
        tips.append(
            "Rare etiketler ("
            + ", ".join(f"`{l}`" for l in rare_hit)
            + ") için step 9 raporuna confidence interval ekleyerek tek-tahmin saturasyonunu açıkça belirt."
        )

    tips.append(
        "Step 7 leak-free kapsamı id-seviyesiydi; step 9'dan önce silver ↔ gold **içerik** örtüşmesini "
        "(char n-gram veya normalize metin hash) ölç ve raporla; near-duplicate kalıntı varsa dokümante et."
    )
    tips.append(
        "`bilgi_paylasimi` F1 exp3'te exp1'in altında kaldı — step 9'da etiket tanımı review et "
        "(arama çağrısı vs bilgi çağrısı sınırı) ve gerekirse annotation guideline'a küçük bir not düş."
    )
    tips.append(
        "FN örneklerini gözle inceleyip potansiyel annotation hatalarını işaretle; tespit edilenleri bir `gold_review_candidates.csv` olarak saklayıp step 10/11 için bekleme listesine koy."
    )
    return tips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 8 error analysis for the leak-free canonical winner.")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.as_posix()))
    ap.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV.as_posix()))
    ap.add_argument("--labels-json", default=str(DEFAULT_LABELS.as_posix()))
    ap.add_argument("--thresholds-json", default=str(DEFAULT_THRESHOLDS.as_posix()))
    ap.add_argument("--comparison-src", default=str(DEFAULT_COMPARISON_SRC.as_posix()))
    ap.add_argument("--text-col", default="tweet_clean")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    _set_deterministic()
    gpu_name = _require_cuda()

    model_dir = Path(args.model_dir)
    test_csv = Path(args.test_csv)
    labels_path = Path(args.labels_json)
    thr_path = Path(args.thresholds_json)

    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")
    if not labels_path.exists():
        raise SystemExit(f"label_columns.json not found: {labels_path}")
    if not thr_path.exists():
        raise SystemExit(f"thresholds_cv.json not found: {thr_path}")

    labels: List[str] = [str(x) for x in json.loads(labels_path.read_text(encoding="utf-8"))]
    thresholds: Dict[str, float] = {
        k: float(v) for k, v in json.loads(thr_path.read_text(encoding="utf-8")).items()
    }
    for lab in labels:
        if lab not in thresholds:
            raise SystemExit(f"Missing threshold for label: {lab}")

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    for c in labels:
        if c not in df.columns:
            raise SystemExit(f"Missing label column in test csv: {c}")
        _coerce_binary_0_1(df, c)

    texts = df[args.text_col].astype("string").fillna("").tolist()
    y_true = df[labels].astype(int).to_numpy()

    probs = _predict_probs(model_dir, texts)
    thr_vec = np.array([thresholds[l] for l in labels], dtype=np.float32).reshape(1, -1)
    y_pred = (probs >= thr_vec).astype(int)

    overall = _micro_macro(y_true, y_pred)
    per_label_conf = {lab: _confusion(y_true[:, i], y_pred[:, i]) for i, lab in enumerate(labels)}
    test_positives = {lab: int(y_true[:, i].sum()) for i, lab in enumerate(labels)}

    # Pool positives across train+val+test if available (for context only).
    gold_pool_positives: Optional[Dict[str, int]] = None
    try:
        pool_frames = []
        for split in ("train", "val", "test"):
            p = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / f"{split}.csv"
            if p.exists():
                pool_frames.append(pd.read_csv(p, encoding="utf-8-sig", usecols=["id"] + labels))
        if pool_frames:
            pool = pd.concat(pool_frames, ignore_index=True)
            for c in labels:
                _coerce_binary_0_1(pool, c)
            gold_pool_positives = {lab: int(pool[lab].sum()) for lab in labels}
    except Exception:
        gold_pool_positives = None

    fn_cof = _fn_cofire(y_true, y_pred, labels)
    fp_cof = _fp_cofire(y_true, y_pred, labels)
    gold_cooc = _gold_cooccurrence(y_true, labels)
    both_one = _both_true_one_pred(y_true, y_pred, labels)

    # Slices
    slices: Dict[str, List[Dict[str, Any]]] = {}
    if "aciliyet_0_3" in df.columns:
        slices["Aciliyet (0-3)"] = _slice_any_error(df, y_true, y_pred, "aciliyet_0_3")
    lens = df[args.text_col].astype("string").fillna("").str.len().map(_length_bucket)
    slices["Metin uzunluğu"] = _slice_any_error(df, y_true, y_pred, "_len_bucket", extra=lens)
    k_gold = pd.Series(y_true.sum(axis=1)).map(_multilabel_count_bucket)
    slices["Gold etiket sayısı"] = _slice_any_error(df, y_true, y_pred, "_k_gold", extra=k_gold)
    tw_type = df[args.text_col].astype("string").fillna("").map(_tweet_type)
    slices["Metin türü (heuristic)"] = _slice_any_error(df, y_true, y_pred, "_tw_type", extra=tw_type)

    fp_ex = _pick_examples(df, probs, y_true, y_pred, labels, "fp", args.top_n, args.text_col, thresholds)
    fn_ex = _pick_examples(df, probs, y_true, y_pred, labels, "fn", args.top_n, args.text_col, thresholds)

    observations = _pattern_observations(
        per_label_conf=per_label_conf,
        fp_cofire=fp_cof,
        fn_cofire=fn_cof,
        thresholds=thresholds,
        gold_totals=gold_pool_positives or {},
        test_positives=test_positives,
    )
    top_patterns = _top5_patterns(per_label_conf, fn_cof, fp_cof, thresholds, test_positives)
    step9_tips = _step9_suggestions(per_label_conf, thresholds, test_positives)

    emphasis_labels = ["altyapi", "guvenlik", "psikolojik", "bilgi_paylasimi"]

    # ----- Write CSV exports -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fp_rows = _examples_to_rows(fp_ex)
    fn_rows = _examples_to_rows(fn_ex)
    pd.DataFrame(fp_rows).to_csv(OUT_FP_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(fn_rows).to_csv(OUT_FN_CSV, index=False, encoding="utf-8-sig")

    slice_flat: List[Dict[str, Any]] = []
    for section, rows in slices.items():
        for r in rows:
            slice_flat.append({"section": section, **r})
    pd.DataFrame(slice_flat).to_csv(OUT_SLICE_CSV, index=False, encoding="utf-8-sig")

    # ----- Write JSON -----
    payload: Dict[str, Any] = {
        "model_dir": str(model_dir.as_posix()),
        "test_csv": str(test_csv.as_posix()),
        "thresholds_json": str(thr_path.as_posix()),
        "comparison_src": str(Path(args.comparison_src).as_posix()),
        "gpu": gpu_name,
        "seed": SEED,
        "rows": int(len(df)),
        "labels": labels,
        "thresholds": thresholds,
        "overall_metrics": overall,
        "per_label_confusion": per_label_conf,
        "test_positives": test_positives,
        "gold_pool_positives": gold_pool_positives,
        "fn_cofire": fn_cof,
        "fp_cofire": fp_cof,
        "gold_cooccurrence_top": gold_cooc[:30],
        "both_true_one_pred_top": both_one[:30],
        "slices": slices,
        "fp_examples": fp_ex,
        "fn_examples": fn_ex,
        "observations": observations,
        "top_patterns": top_patterns,
        "step9_suggestions": step9_tips,
        "emphasis_labels": emphasis_labels,
        "leak_free_caveat": (
            "Step 7 v3 id-level silver→gold leakage kapatıldı; ancak content-level "
            "(exact-text / near-duplicate) örtüşme riski tamamen ölçülmedi. Bu rapordaki "
            "yüksek per-label F1 değerleri (özellikle rare etiketler) bu sınır hatırlanarak okunmalı."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ----- Write Markdown -----
    md = _render_markdown(
        model_dir=str(model_dir.as_posix()),
        test_csv=str(test_csv.as_posix()),
        thresholds_src=str(thr_path.as_posix()),
        comparison_src=str(Path(args.comparison_src).as_posix()),
        gpu_name=gpu_name,
        rows=int(len(df)),
        labels=labels,
        thresholds=thresholds,
        overall=overall,
        per_label_conf=per_label_conf,
        test_positives=test_positives,
        gold_pool_positives=gold_pool_positives,
        fn_cofire=fn_cof,
        fp_cofire=fp_cof,
        gold_cooc=gold_cooc,
        both_true_one_pred=both_one,
        slices=slices,
        fp_examples=fp_ex,
        fn_examples=fn_ex,
        observations=observations,
        emphasis_labels=emphasis_labels,
        top_patterns=top_patterns,
        step9_suggestions=step9_tips,
    )
    OUT_MD.write_text(md, encoding="utf-8")

    print("Wrote:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_FP_CSV}")
    print(f"  {OUT_FN_CSV}")
    print(f"  {OUT_SLICE_CSV}")

    print("\n=== Overall (CV thresholds) ===")
    print(
        f"  f1_micro={overall['f1_micro']:.4f}  f1_macro={overall['f1_macro']:.4f}  "
        f"P={overall['precision_micro']:.4f}  R={overall['recall_micro']:.4f}"
    )
    print("\n=== Per-label F1 ===")
    for lab in labels:
        c = per_label_conf[lab]
        print(
            f"  {lab:18s} F1={c['f1']:.3f}  TP={c['tp']:3d} FP={c['fp']:3d} FN={c['fn']:3d}  "
            f"(test_pos={test_positives[lab]}, thr={thresholds[lab]:.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
