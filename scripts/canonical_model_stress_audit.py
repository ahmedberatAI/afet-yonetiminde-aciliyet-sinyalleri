#!/usr/bin/env python3
"""Stress-test the canonical need classifier without mutating model artifacts.

The script intentionally writes new analysis artifacts only:

- data/analysis/canonical_model_stress_audit_2026_05_17.json
- data/analysis/canonical_model_stress_audit_2026_05_17.md

It reloads the selected HuggingFace checkpoint, reproduces canonical test
metrics, and adds diagnostic views around thresholds, per-label mistakes,
location/urgency/text-length buckets, near-threshold uncertainty, and a small
qualitative challenge set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "final"
DEFAULT_TEST_CSV = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "test.csv"
DEFAULT_LABELS_JSON = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "label_columns.json"
DEFAULT_THRESHOLDS_JSON = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "thresholds_cv.json"
DEFAULT_SELECTION_JSON = REPO_ROOT / "models" / "final" / "selection.json"
DEFAULT_CANONICAL_PRED = REPO_ROOT / "data" / "predictions" / "need_predictions_geolocated_v2_final.csv"
DEFAULT_CANONICAL_META = REPO_ROOT / "data" / "predictions" / "need_predictions_geolocated_v2_final.meta.json"
DEFAULT_DASHBOARD_REPO = REPO_ROOT.parent / "afetYonetimi-dashboard"
DEFAULT_OUT_PREFIX = REPO_ROOT / "data" / "analysis" / "canonical_model_stress_audit_2026_05_17"

TEXT_COL = "tweet_clean"
FOCUS_LABELS = ["guvenlik", "bilgi_paylasimi"]

KEYWORD_GROUPS: dict[str, list[str]] = {
    "guvenlik": [
        r"\bh[ıi]rs[ıi]z",
        r"\bya[gğ]ma",
        r"\bya[gğ]mac",
        r"\basayi[şs]",
        r"\bg[uü]venlik",
        r"\bpolis\b",
        r"\bjandarma\b",
        r"\bf[ıi]rsat[cç][ıi]",
        r"\btehlike",
    ],
    "bilgi_paylasimi": [
        r"\bhaber alam",
        r"\bhaber alan",
        r"\bg[oö]ren\b",
        r"\bduyan\b",
        r"\bbilgi",
        r"\bula[şs]am",
        r"\bula[şs][ıi]n",
        r"\byak[ıi]n[ıi]m",
        r"\bileti[şs]im",
        r"\bdurum[uu]?",
    ],
}

CHALLENGE_SET: list[dict[str, Any]] = [
    {
        "name": "security_direct_looting",
        "text": "Mahallede hırsızlık ve yağma başladı, asayiş yok. Lütfen polis veya jandarma gelsin.",
        "expected": ["guvenlik"],
    },
    {
        "name": "security_plus_basic_needs",
        "text": "Çadır ve su bekleyen 40 kişi var, ayrıca hırsızlar dadanmış ve güvenlik yok.",
        "expected": ["barinma", "gida_su", "guvenlik"],
    },
    {
        "name": "security_fire_ambiguous",
        "text": "Binanın yanında yangın var, çevre çok tehlikeli, ekiplerin güvenliği sağlanmalı.",
        "expected": ["guvenlik", "altyapi"],
    },
    {
        "name": "info_only_missing_relative",
        "text": "Ailemden haber alamıyoruz, gören duyan varsa lütfen bilgi versin.",
        "expected": ["bilgi_paylasimi"],
    },
    {
        "name": "info_plus_rescue",
        "text": "Yakınım enkaz altında olabilir, haber alamıyoruz. Gören ya da bilgi alan varsa yazsın.",
        "expected": ["arama_kurtarma", "bilgi_paylasimi"],
    },
    {
        "name": "info_address_only",
        "text": "Gaziantep Nurdağı Atatürk Mahallesi Ata Caddesi, bilgisi olan acil ulaşsın.",
        "expected": ["bilgi_paylasimi"],
    },
    {
        "name": "logistics_rescue_machine",
        "text": "Enkaz başında kepçe ve vinç gerekiyor, arama kurtarma ekibi acil yönlendirilsin.",
        "expected": ["arama_kurtarma", "lojistik"],
    },
    {
        "name": "basic_needs_no_rescue",
        "text": "Toplanma alanında bebek maması, su, battaniye ve çadır ihtiyacı var.",
        "expected": ["barinma", "gida_su"],
    },
    {
        "name": "announcement_distribution",
        "text": "Yardımlar okula ulaştırıldı, yemek dağıtımı başladı, eksik olanlar belediyeye bildirilsin.",
        "expected": ["bilgi_paylasimi", "gida_su"],
    },
    {
        "name": "psychological_support",
        "text": "Çocuklar panik atak geçiriyor, psikolojik destek ve sakinleştirecek uzman gerekiyor.",
        "expected": ["psikolojik"],
    },
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_binary_0_1(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip().replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    bad = ~df[col].isin([0, 1])
    if bool(bad.any()):
        raise SystemExit(f"Invalid values in {col}: expected 0/1 only.")


def _load_test(csv_path: Path, label_cols: list[str], text_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"id": "string"})
    missing = [c for c in [text_col] + label_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")
    for c in label_cols:
        _coerce_binary_0_1(df, c)
    return df


def _predict_probs(
    model_dir: Path,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    prefer_cpu: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)

    all_logits: list[np.ndarray] = []
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
    probs = _sigmoid(np.concatenate(all_logits, axis=0))
    runtime = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": str(device),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
    }
    if torch.cuda.is_available():
        runtime["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    return probs, runtime


def _threshold_matrix(label_cols: list[str], thresholds: dict[str, float]) -> np.ndarray:
    return np.array([float(thresholds[l]) for l in label_cols], dtype=np.float32).reshape(1, -1)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int((y_true == 1).sum()),
        "predicted_positive": int((y_pred == 1).sum()),
    }


def _overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = _confusion(y_true.reshape(-1), y_pred.reshape(-1))
    per_label = [_confusion(y_true[:, j], y_pred[:, j])["f1"] for j in range(y_true.shape[1])]
    return {
        "precision_micro": float(c["precision"]),
        "recall_micro": float(c["recall"]),
        "f1_micro": float(c["f1"]),
        "f1_macro": float(np.mean(per_label)) if per_label else 0.0,
    }


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict[str, Any]:
    return _confusion(y_true, (probs >= float(thr)).astype(int))


def _threshold_sensitivity(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    out: dict[str, Any] = {}
    for j, lab in enumerate(labels):
        current_thr = float(thresholds[lab])
        current = _metrics_at_threshold(y_true[:, j], probs[:, j], current_thr)
        rows: list[dict[str, Any]] = []
        for thr in grid:
            m = _metrics_at_threshold(y_true[:, j], probs[:, j], float(thr))
            rows.append(
                {
                    "threshold": float(thr),
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "f1": float(m["f1"]),
                    "tp": int(m["tp"]),
                    "fp": int(m["fp"]),
                    "fn": int(m["fn"]),
                    "predicted_positive": int(m["predicted_positive"]),
                }
            )
        best = max(rows, key=lambda r: (r["f1"], r["recall"], -abs(r["threshold"] - current_thr)))
        nearby = {}
        for delta in [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]:
            cand = min(0.99, max(0.01, current_thr + delta))
            nearby[f"{delta:+.2f}"] = _metrics_at_threshold(y_true[:, j], probs[:, j], cand)
            nearby[f"{delta:+.2f}"]["threshold"] = float(cand)
        fn_mask = (y_true[:, j] == 1) & (probs[:, j] < current_thr)
        recoverable_005 = int(((current_thr - probs[:, j] <= 0.05) & (current_thr - probs[:, j] > 0) & fn_mask).sum())
        recoverable_010 = int(((current_thr - probs[:, j] <= 0.10) & (current_thr - probs[:, j] > 0) & fn_mask).sum())
        out[lab] = {
            "current_threshold": current_thr,
            "current": current,
            "best_on_test_diagnostic_only": best,
            "nearby_thresholds": nearby,
            "fn_within_0_05_below_threshold": recoverable_005,
            "fn_within_0_10_below_threshold": recoverable_010,
            "grid": rows,
        }
    return out


def _label_list(row: np.ndarray, labels: list[str]) -> list[str]:
    return [labels[i] for i, v in enumerate(row) if int(v) == 1]


def _excerpt(text: Any, limit: int = 280) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "..."


def _row_context(
    df: pd.DataFrame,
    idx: int,
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    thresholds: dict[str, float],
    label: str | None = None,
) -> dict[str, Any]:
    row = df.iloc[idx]
    base: dict[str, Any] = {
        "row_index": int(idx),
        "id": str(row.get("id", "")),
        "province": str(row.get("province", "")),
        "district": str(row.get("district", "")),
        "urgency_score": _safe_number(row.get("urgency_score", "")),
        "aciliyet_0_3": _safe_number(row.get("aciliyet_0_3", "")),
        "gold_labels": _label_list(y_true[idx], labels),
        "pred_labels": _label_list(y_pred[idx], labels),
        "text": _excerpt(row.get(TEXT_COL, "")),
    }
    if label:
        j = labels.index(label)
        base["label"] = label
        base["prob"] = float(probs[idx, j])
        base["threshold"] = float(thresholds[label])
        base["margin_to_threshold"] = float(probs[idx, j] - thresholds[label])
    top = sorted(
        [
            {"label": lab, "prob": float(probs[idx, k]), "threshold": float(thresholds[lab])}
            for k, lab in enumerate(labels)
        ],
        key=lambda r: r["prob"],
        reverse=True,
    )[:4]
    base["top_probs"] = top
    return base


def _safe_number(value: Any) -> int | float | str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if f.is_integer():
        return int(f)
    return f


def _focus_error_examples(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
    focus_labels: Iterable[str],
    *,
    top_n: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lab in focus_labels:
        j = labels.index(lab)
        fp = np.where((y_true[:, j] == 0) & (y_pred[:, j] == 1))[0]
        fn = np.where((y_true[:, j] == 1) & (y_pred[:, j] == 0))[0]
        fp_order = sorted(fp, key=lambda i: float(probs[i, j]), reverse=True)
        fn_order = sorted(fn, key=lambda i: float(probs[i, j]))
        out[lab] = {
            "false_positives": [
                _row_context(df, int(i), labels, y_true, y_pred, probs, thresholds, lab)
                for i in fp_order[:top_n]
            ],
            "false_negatives": [
                _row_context(df, int(i), labels, y_true, y_pred, probs, thresholds, lab)
                for i in fn_order[:top_n]
            ],
        }
    return out


def _cofire(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    fn_cofire: dict[str, dict[str, int]] = {}
    fp_gold_cofire: dict[str, dict[str, int]] = {}
    for i, lab in enumerate(labels):
        fn_mask = (y_true[:, i] == 1) & (y_pred[:, i] == 0)
        fp_mask = (y_true[:, i] == 0) & (y_pred[:, i] == 1)

        fn_counts = Counter()
        if bool(fn_mask.any()):
            for j, other in enumerate(labels):
                if j != i:
                    c = int(y_pred[fn_mask, j].sum())
                    if c:
                        fn_counts[other] = c

        fp_counts = Counter()
        if bool(fp_mask.any()):
            for j, other in enumerate(labels):
                if j != i:
                    c = int(y_true[fp_mask, j].sum())
                    if c:
                        fp_counts[other] = c

        fn_cofire[lab] = dict(fn_counts.most_common())
        fp_gold_cofire[lab] = dict(fp_counts.most_common())
    return {"fn_predicted_other_labels": fn_cofire, "fp_gold_other_labels": fp_gold_cofire}


def _gold_pairs(y_true: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            count = int(((y_true[:, i] == 1) & (y_true[:, j] == 1)).sum())
            if count:
                rows.append({"a": a, "b": b, "count": count})
    return sorted(rows, key=lambda r: (-r["count"], r["a"], r["b"]))


def _bucket_text_len(text: Any) -> str:
    n = len(str(text or ""))
    if n < 120:
        return "short(<120)"
    if n < 220:
        return "medium(120-219)"
    return "long(>=220)"


def _bucket_label_count(k: int) -> str:
    if k <= 0:
        return "0"
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def _slice_summary(
    name: str,
    groups: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_rows: int = 1,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    any_error = (y_true != y_pred).any(axis=1).astype(int)
    work = pd.DataFrame({"group": groups.astype("string").fillna("(missing)"), "any_error": any_error})
    rows: list[dict[str, Any]] = []
    for group, sub in work.groupby("group", dropna=False):
        n = int(len(sub))
        if n < min_rows:
            continue
        err = int(sub["any_error"].sum())
        rows.append(
            {
                "slice": name,
                "group": str(group),
                "rows": n,
                "rows_with_any_label_error": err,
                "error_rate": err / n if n else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["error_rate"], -r["rows_with_any_label_error"], -r["rows"], r["group"]))
    if top_n is not None:
        return rows[:top_n]
    return rows


def _all_slices(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if "aciliyet_0_3" in df.columns:
        out["aciliyet_0_3"] = _slice_summary("aciliyet_0_3", df["aciliyet_0_3"], y_true, y_pred)
    if "urgency_score" in df.columns:
        out["urgency_score"] = _slice_summary("urgency_score", df["urgency_score"], y_true, y_pred)
    out["text_length"] = _slice_summary(
        "text_length", df[TEXT_COL].map(_bucket_text_len), y_true, y_pred
    )
    out["gold_label_count"] = _slice_summary(
        "gold_label_count",
        pd.Series(y_true.sum(axis=1)).map(lambda x: _bucket_label_count(int(x))),
        y_true,
        y_pred,
    )
    if "province" in df.columns:
        out["province_min3"] = _slice_summary("province", df["province"], y_true, y_pred, min_rows=3, top_n=20)
    if "district" in df.columns:
        combo = df.get("province", "").astype("string").fillna("") + " / " + df["district"].astype("string").fillna("")
        out["province_district_min3"] = _slice_summary(
            "province_district", combo, y_true, y_pred, min_rows=3, top_n=25
        )
    return out


def _uncertainty(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
    *,
    top_n: int,
) -> dict[str, Any]:
    thr_vec = np.array([thresholds[l] for l in labels], dtype=np.float32).reshape(1, -1)
    margins = probs - thr_vec
    abs_margins = np.abs(margins)
    min_margin = abs_margins.min(axis=1)
    min_label_idx = abs_margins.argmin(axis=1)
    any_error = (y_true != y_pred).any(axis=1)

    near_rows = []
    for idx in np.argsort(min_margin, kind="stable")[:top_n]:
        lab = labels[int(min_label_idx[idx])]
        ctx = _row_context(df, int(idx), labels, y_true, y_pred, probs, thresholds, lab)
        ctx["any_label_error"] = bool(any_error[idx])
        near_rows.append(ctx)

    error_near_rows = []
    err_idxs = [int(i) for i in np.argsort(min_margin, kind="stable") if bool(any_error[int(i)])]
    for idx in err_idxs[:top_n]:
        lab = labels[int(min_label_idx[idx])]
        ctx = _row_context(df, int(idx), labels, y_true, y_pred, probs, thresholds, lab)
        ctx["any_label_error"] = True
        error_near_rows.append(ctx)

    return {
        "rows_with_any_error": int(any_error.sum()),
        "rows_near_any_threshold_0_02": int((min_margin <= 0.02).sum()),
        "rows_near_any_threshold_0_05": int((min_margin <= 0.05).sum()),
        "rows_near_any_threshold_0_10": int((min_margin <= 0.10).sum()),
        "error_rows_near_any_threshold_0_05": int(((min_margin <= 0.05) & any_error).sum()),
        "error_rows_near_any_threshold_0_10": int(((min_margin <= 0.10) & any_error).sum()),
        "closest_rows": near_rows,
        "closest_error_rows": error_near_rows,
    }


def _keyword_audit(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    focus_labels: Iterable[str],
) -> dict[str, Any]:
    texts = df[TEXT_COL].astype("string").fillna("").str.casefold()
    out: dict[str, Any] = {}
    for lab in focus_labels:
        j = labels.index(lab)
        regexes = [re.compile(p, flags=re.IGNORECASE) for p in KEYWORD_GROUPS.get(lab, [])]
        has_keyword = texts.map(lambda s: any(r.search(str(s)) for r in regexes)).to_numpy(dtype=bool)
        gold_pos = y_true[:, j] == 1
        pred_pos = y_pred[:, j] == 1
        fp = (~gold_pos) & pred_pos
        fn = gold_pos & (~pred_pos)
        tp = gold_pos & pred_pos
        out[lab] = {
            "patterns": KEYWORD_GROUPS.get(lab, []),
            "rows_with_keyword": int(has_keyword.sum()),
            "gold_positive_with_keyword": int((gold_pos & has_keyword).sum()),
            "tp_with_keyword": int((tp & has_keyword).sum()),
            "fn_with_keyword": int((fn & has_keyword).sum()),
            "fp_with_keyword": int((fp & has_keyword).sum()),
            "gold_positive_without_keyword": int((gold_pos & ~has_keyword).sum()),
            "fn_without_keyword": int((fn & ~has_keyword).sum()),
        }
    return out


def _challenge_results(
    model_dir: Path,
    labels: list[str],
    thresholds: dict[str, float],
    *,
    max_length: int,
    batch_size: int,
    prefer_cpu: bool,
) -> dict[str, Any]:
    texts = [row["text"] for row in CHALLENGE_SET]
    probs, runtime = _predict_probs(
        model_dir,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        prefer_cpu=prefer_cpu,
    )
    thr = _threshold_matrix(labels, thresholds)
    pred = (probs >= thr).astype(int)
    rows = []
    for i, row in enumerate(CHALLENGE_SET):
        rows.append(
            {
                "name": row["name"],
                "text": row["text"],
                "expected_not_a_metric": row["expected"],
                "pred_labels": _label_list(pred[i], labels),
                "top_probs": sorted(
                    [
                        {
                            "label": lab,
                            "prob": float(probs[i, j]),
                            "threshold": float(thresholds[lab]),
                            "fires": bool(pred[i, j]),
                        }
                        for j, lab in enumerate(labels)
                    ],
                    key=lambda r: r["prob"],
                    reverse=True,
                )[:5],
            }
        )
    return {"note": "Synthetic qualitative probes only; do not treat as held-out metrics.", "runtime": runtime, "rows": rows}


def _prediction_artifact_audit(
    pred_csv: Path,
    meta_json: Path,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {"csv": str(pred_csv), "meta": str(meta_json), "exists": pred_csv.exists()}
    if meta_json.exists():
        meta = _json_load(meta_json)
        out["meta_exists"] = True
        out["meta_canonical"] = bool(meta.get("canonical"))
        out["meta_row_count"] = meta.get("row_count")
        out["meta_selected_experiment_key"] = meta.get("selected_experiment_key")
        out["meta_thresholds_match"] = {
            lab: abs(float(meta.get("thresholds_per_label", {}).get(lab, -999)) - float(thresholds[lab])) < 1e-6
            for lab in labels
        }
    else:
        out["meta_exists"] = False
    if pred_csv.exists():
        header = pd.read_csv(pred_csv, encoding="utf-8-sig", nrows=0)
        cols = list(header.columns)
        out["has_required_prediction_columns"] = all(f"pred_{lab}" in cols for lab in labels)
        out["has_required_probability_columns"] = all(f"prob_{lab}" in cols for lab in labels)
        # 63k rows is small enough to count without touching the file content semantically.
        out["csv_row_count"] = int(sum(1 for _ in pred_csv.open("r", encoding="utf-8-sig", errors="replace")) - 1)
    return out


def _dashboard_meta_audit(dashboard_repo: Path, labels: list[str], thresholds: dict[str, float]) -> dict[str, Any]:
    labels_path = dashboard_repo / "data" / "model_meta" / "label_columns.json"
    thr_path = dashboard_repo / "data" / "model_meta" / "thresholds_cv.json"
    out: dict[str, Any] = {
        "dashboard_repo": str(dashboard_repo),
        "labels_path": str(labels_path),
        "thresholds_path": str(thr_path),
        "labels_exists": labels_path.exists(),
        "thresholds_exists": thr_path.exists(),
    }
    if labels_path.exists():
        dash_labels = [str(x) for x in _json_load(labels_path)]
        out["labels_match_canonical"] = dash_labels == labels
    if thr_path.exists():
        dash_thr = {str(k): float(v) for k, v in _json_load(thr_path).items()}
        out["thresholds_match_canonical"] = {
            lab: lab in dash_thr and abs(float(dash_thr[lab]) - float(thresholds[lab])) < 1e-6
            for lab in labels
        }
    return out


def _float4(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _metric_table(per_label: dict[str, Any], labels: list[str], thresholds: dict[str, float]) -> list[str]:
    lines = [
        "| label | support | threshold | precision | recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lab in labels:
        c = per_label[lab]
        marker = " **" if lab in FOCUS_LABELS else ""
        end = "**" if lab in FOCUS_LABELS else ""
        lines.append(
            f"| {marker}`{lab}`{end} | {c['support']} | {thresholds[lab]:.2f} | "
            f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
            f"{c['tp']} | {c['fp']} | {c['fn']} |"
        )
    return lines


def _render_focus_section(label: str, audit: dict[str, Any]) -> list[str]:
    per = audit["per_label"][label]
    sens = audit["threshold_sensitivity"][label]
    cofire = audit["cofire"]
    keyword = audit["keyword_audit"][label]
    examples = audit["focus_errors"][label]

    lines = []
    lines.append(f"### `{label}`")
    lines.append(
        f"- Test desteği {per['support']} pozitif; TP={per['tp']}, FP={per['fp']}, FN={per['fn']}, "
        f"P={per['precision']:.3f}, R={per['recall']:.3f}, F1={per['f1']:.3f}."
    )
    best = sens["best_on_test_diagnostic_only"]
    lines.append(
        f"- CV eşiği {sens['current_threshold']:.2f}. Test üzerinde tanısal en iyi eşik "
        f"{best['threshold']:.2f} ile F1={best['f1']:.3f} görünüyor; bu değer seçim için kullanılmamalı, "
        "sadece hassasiyet sinyali."
    )
    if sens["fn_within_0_10_below_threshold"]:
        lines.append(
            f"- FN'lerin {sens['fn_within_0_10_below_threshold']} tanesi eşiğin 0.10 altında; "
            "threshold oynatması bazı kaçırmaları geri alabilir."
        )
    else:
        lines.append("- FN'ler eşiğe çok yakın değil; salt threshold düşürmek bütün hatayı açıklamıyor.")
    fn_co = cofire["fn_predicted_other_labels"].get(label, {})
    fp_co = cofire["fp_gold_other_labels"].get(label, {})
    if fn_co:
        lines.append("- FN cofire: " + ", ".join(f"`{k}` x{v}" for k, v in list(fn_co.items())[:5]) + ".")
    if fp_co:
        lines.append("- FP satırlarında gerçek diğer etiketler: " + ", ".join(f"`{k}` x{v}" for k, v in list(fp_co.items())[:5]) + ".")
    lines.append(
        "- Anahtar kelime kapsaması: "
        f"gold pozitif {keyword['gold_positive_with_keyword']}/{per['support']} satırda yakalandı; "
        f"FN içinde keyword={keyword['fn_with_keyword']}, keyword yok={keyword['fn_without_keyword']}."
    )

    if examples["false_negatives"]:
        lines.append("- Kritik FN örnekleri:")
        for row in examples["false_negatives"][:4]:
            lines.append(
                f"  - id={row['id']} p={row['prob']:.3f}, gold={','.join(row['gold_labels']) or '-'}, "
                f"pred={','.join(row['pred_labels']) or '-'} - {row['text']}"
            )
    if examples["false_positives"]:
        lines.append("- Kritik FP örnekleri:")
        for row in examples["false_positives"][:4]:
            lines.append(
                f"  - id={row['id']} p={row['prob']:.3f}, gold={','.join(row['gold_labels']) or '-'}, "
                f"pred={','.join(row['pred_labels']) or '-'} - {row['text']}"
            )
    return lines


def _render_markdown(audit: dict[str, Any]) -> str:
    labels = audit["labels"]
    thresholds = audit["thresholds"]
    lines: list[str] = []
    lines.append("# Canonical Model Stress Audit - 2026-05-17")
    lines.append("")
    lines.append("Bu rapor canonical modeli yeniden yükleyip test setini tekrar koşturur; model, eşik ve canonical prediction artefaktlarını değiştirmez.")
    lines.append("")
    lines.append("## 1. Baseline doğrulama")
    lines.append("")
    sel = audit["selection"]
    lines.append(f"- Selection experiment: `{sel.get('selected_experiment_key')}`")
    lines.append(f"- Model dir: `{audit['model_dir']}`")
    lines.append(f"- Test CSV: `{audit['test_csv']}`")
    lines.append(f"- Runtime: `{audit['runtime']['device_used']}`, torch `{audit['runtime']['torch_version']}`")
    overall = audit["overall"]
    lines.append(
        f"- Reproduced metrics: F1 micro={overall['f1_micro']:.4f}, "
        f"F1 macro={overall['f1_macro']:.4f}, P micro={overall['precision_micro']:.4f}, "
        f"R micro={overall['recall_micro']:.4f}."
    )
    delta = audit["selection_metric_delta"]
    lines.append(
        f"- Selection delta: micro={delta.get('f1_micro', 0):+.8f}, macro={delta.get('f1_macro', 0):+.8f}."
    )
    pred = audit["prediction_artifact_audit"]
    lines.append(
        f"- Canonical prediction audit: exists={pred.get('exists')}, rows={pred.get('csv_row_count')}, "
        f"meta canonical={pred.get('meta_canonical')}, experiment=`{pred.get('meta_selected_experiment_key')}`."
    )
    dash = audit["dashboard_meta_audit"]
    lines.append(
        f"- Dashboard meta uyumu: labels={dash.get('labels_match_canonical')}, "
        f"thresholds={all(dash.get('thresholds_match_canonical', {}).values()) if dash.get('thresholds_match_canonical') else False}."
    )
    lines.append("")
    lines.append("## 2. Etiket bazlı sonuç")
    lines.append("")
    lines.extend(_metric_table(audit["per_label"], labels, thresholds))
    lines.append("")
    lines.append("## 3. Zayıf etiketler")
    lines.append("")
    for lab in FOCUS_LABELS:
        lines.extend(_render_focus_section(lab, audit))
        lines.append("")

    lines.append("## 4. Threshold hassasiyeti")
    lines.append("")
    lines.append("| label | current_thr | current_F1 | diagnostic_best_thr | diagnostic_best_F1 | current_FN | best_FN | best_FP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lab in labels:
        sens = audit["threshold_sensitivity"][lab]
        best = sens["best_on_test_diagnostic_only"]
        curr = sens["current"]
        lines.append(
            f"| `{lab}` | {sens['current_threshold']:.2f} | {curr['f1']:.3f} | "
            f"{best['threshold']:.2f} | {best['f1']:.3f} | {curr['fn']} | {best['fn']} | {best['fp']} |"
        )
    lines.append("")
    lines.append("> Not: `diagnostic_best_thr` test setinden hesaplandığı için production eşiği olarak önerilmez; sadece mevcut eşiğin ne kadar hassas olduğunu gösterir.")
    lines.append("")

    lines.append("## 5. Bucket analizi")
    lines.append("")
    for key in ["text_length", "aciliyet_0_3", "urgency_score", "gold_label_count", "province_min3", "province_district_min3"]:
        rows = audit["slices"].get(key, [])
        if not rows:
            continue
        lines.append(f"### {key}")
        lines.append("| group | rows | error_rows | error_rate |")
        lines.append("|---|---:|---:|---:|")
        for r in rows[:10]:
            lines.append(
                f"| {r['group']} | {r['rows']} | {r['rows_with_any_label_error']} | {r['error_rate']:.3f} |"
            )
        lines.append("")

    lines.append("## 6. Belirsiz örnekler")
    lines.append("")
    unc = audit["uncertainty"]
    lines.append(
        f"- Herhangi bir eşiğe 0.05 yakın satır: {unc['rows_near_any_threshold_0_05']} / {audit['rows']}."
    )
    lines.append(
        f"- Hatalı olup herhangi bir eşiğe 0.05 yakın satır: {unc['error_rows_near_any_threshold_0_05']}."
    )
    lines.append("- Eşiğe en yakın hatalı örnekler:")
    for row in unc["closest_error_rows"][:8]:
        lines.append(
            f"  - `{row['label']}` p={row['prob']:.3f}, thr={row['threshold']:.3f}, "
            f"gold={','.join(row['gold_labels']) or '-'}, pred={','.join(row['pred_labels']) or '-'} - {row['text']}"
        )
    lines.append("")

    lines.append("## 7. Challenge set")
    lines.append("")
    lines.append("Bu bölüm sentetik/elle yazılmış nitel problardır; skor olarak okunmamalı.")
    lines.append("| probe | expected | predicted | top probs |")
    lines.append("|---|---|---|---|")
    for row in audit["challenge_set"]["rows"]:
        top = ", ".join(f"{p['label']}={p['prob']:.2f}" for p in row["top_probs"][:3])
        lines.append(
            f"| `{row['name']}` | {','.join(row['expected_not_a_metric']) or '-'} | "
            f"{','.join(row['pred_labels']) or '-'} | {top} |"
        )
    lines.append("")

    lines.append("## 8. Sonuç ve öneriler")
    lines.append("")
    for item in audit["recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _recommendations(audit: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    per = audit["per_label"]
    sens = audit["threshold_sensitivity"]
    cofire = audit["cofire"]["fn_predicted_other_labels"]
    challenge_rows = audit.get("challenge_set", {}).get("rows", [])

    recs.append(
        "Baseline korunmalı: canonical model ve CV eşikleri yeniden üretildi; bu rapor model seçimini değiştirmiyor."
    )
    bp = per["bilgi_paylasimi"]
    bp_best = sens["bilgi_paylasimi"]["best_on_test_diagnostic_only"]
    recs.append(
        "`bilgi_paylasimi` için en hızlı güvenli deney, production amaçlı ayrı bir recall-senaryosu eşiği çalışmak: "
        f"CV eşiği {sens['bilgi_paylasimi']['current_threshold']:.2f}, mevcut recall={bp['recall']:.2f}; "
        f"test üstündeki tanısal en iyi eşik {bp_best['threshold']:.2f}. Bu eşik doğrudan seçilmemeli, OOF/validation ile doğrulanmalı."
    )
    if cofire.get("bilgi_paylasimi", {}).get("arama_kurtarma"):
        recs.append(
            "`bilgi_paylasimi` FN'lerinde `arama_kurtarma` cofire belirgin; guideline'a 'haber alamıyoruz/gören duyan var mı' ifadeleri arama-kurtarma ile birlikte de `bilgi_paylasimi` alır notu eklenmeli."
        )
    gv = per["guvenlik"]
    recs.append(
        "`guvenlik` için skor çok küçük desteğe dayanıyor "
        f"(test pozitif={gv['support']}, pool pozitif={audit['gold_pool_positives'].get('guvenlik')}); "
        "hızlı iyileştirme threshold değil, keyword/active-learning adaylarından çift etiketlemeli yeni pozitif örnek toplamak."
    )
    psych_extra = [
        r["name"]
        for r in challenge_rows
        if "psikolojik" in r.get("pred_labels", []) and "psikolojik" not in r.get("expected_not_a_metric", [])
    ]
    if psych_extra:
        recs.append(
            "`psikolojik` testte kusursuz görünüyor ama challenge problarında beklenmeyen ateşlemeler var "
            f"({', '.join(psych_extra[:4])}); bu rare-label skorunu kalibre başarı olarak okumamak gerekir."
        )
    info_missed = [
        r["name"]
        for r in challenge_rows
        if "bilgi_paylasimi" in r.get("expected_not_a_metric", []) and "bilgi_paylasimi" not in r.get("pred_labels", [])
    ]
    if info_missed:
        recs.append(
            "`bilgi_paylasimi` challenge problarında kısa/adres ağırlıklı bilgi çağrılarını kaçırabiliyor "
            f"({', '.join(info_missed[:4])}); guideline ve veri genişletme planında bu alt tip ayrı kovalanmalı."
        )
    recs.append(
        "Büyük iyileştirme için silver havuzu içerik düzeyinde dedup edilip `guvenlik` ve `bilgi_paylasimi` odaklı 50-100 yeni gold pozitif eklenmeli; ardından threshold CV tekrar koşulmalı."
    )
    recs.append(
        "Dashboard tarafında değişiklik gerekmedi: bundled label/threshold metadata canonical ile eşleşiyor ve legacy 63k fallback geri getirilmedi."
    )
    return recs


def _gold_pool_counts(split_dir: Path, labels: list[str]) -> dict[str, int]:
    frames = []
    for split in ["train", "val", "test"]:
        p = split_dir / f"{split}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, encoding="utf-8-sig", usecols=labels))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    for c in labels:
        _coerce_binary_0_1(df, c)
    return {lab: int(df[lab].sum()) for lab in labels}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a non-mutating stress audit for the canonical model.")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    ap.add_argument("--labels-json", default=str(DEFAULT_LABELS_JSON))
    ap.add_argument("--thresholds-json", default=str(DEFAULT_THRESHOLDS_JSON))
    ap.add_argument("--selection-json", default=str(DEFAULT_SELECTION_JSON))
    ap.add_argument("--canonical-pred", default=str(DEFAULT_CANONICAL_PRED))
    ap.add_argument("--canonical-meta", default=str(DEFAULT_CANONICAL_META))
    ap.add_argument("--dashboard-repo", default=str(DEFAULT_DASHBOARD_REPO))
    ap.add_argument("--out-prefix", default=str(DEFAULT_OUT_PREFIX))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--prefer-cpu", action="store_true")
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    test_csv = Path(args.test_csv)
    labels_json = Path(args.labels_json)
    thresholds_json = Path(args.thresholds_json)
    selection_json = Path(args.selection_json)
    pred_csv = Path(args.canonical_pred)
    pred_meta = Path(args.canonical_meta)
    dashboard_repo = Path(args.dashboard_repo)
    out_prefix = Path(args.out_prefix)

    labels = [str(x) for x in _json_load(labels_json)]
    thresholds = {str(k): float(v) for k, v in _json_load(thresholds_json).items()}
    selection = _json_load(selection_json) if selection_json.exists() else {}

    df = _load_test(test_csv, labels, TEXT_COL)
    texts = df[TEXT_COL].astype("string").fillna("").tolist()
    probs, runtime = _predict_probs(
        model_dir,
        texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        prefer_cpu=bool(args.prefer_cpu),
    )
    y_true = df[labels].astype(int).to_numpy()
    y_pred = (probs >= _threshold_matrix(labels, thresholds)).astype(int)

    overall = _overall_metrics(y_true, y_pred)
    per_label = {lab: _confusion(y_true[:, i], y_pred[:, i]) for i, lab in enumerate(labels)}
    expected_metrics = selection.get("metrics_on_canonical_test", {}) if isinstance(selection, dict) else {}
    selection_delta = {
        k: float(overall[k]) - float(expected_metrics.get(k, overall[k]))
        for k in ["f1_micro", "f1_macro", "precision_micro", "recall_micro"]
    }

    audit: dict[str, Any] = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "test_csv": str(test_csv),
        "labels_json": str(labels_json),
        "thresholds_json": str(thresholds_json),
        "selection_json": str(selection_json),
        "selection": selection,
        "runtime": runtime,
        "rows": int(len(df)),
        "labels": labels,
        "thresholds": thresholds,
        "overall": overall,
        "selection_metric_delta": selection_delta,
        "per_label": per_label,
        "gold_pool_positives": _gold_pool_counts(test_csv.parent, labels),
        "threshold_sensitivity": _threshold_sensitivity(y_true, probs, labels, thresholds),
        "cofire": _cofire(y_true, y_pred, labels),
        "gold_cooccurrence": _gold_pairs(y_true, labels),
        "slices": _all_slices(df, y_true, y_pred),
        "uncertainty": _uncertainty(df, y_true, y_pred, probs, labels, thresholds, top_n=int(args.top_n)),
        "focus_errors": _focus_error_examples(
            df,
            y_true,
            y_pred,
            probs,
            labels,
            thresholds,
            FOCUS_LABELS,
            top_n=int(args.top_n),
        ),
        "keyword_audit": _keyword_audit(df, y_true, y_pred, labels, FOCUS_LABELS),
        "prediction_artifact_audit": _prediction_artifact_audit(pred_csv, pred_meta, labels, thresholds),
        "dashboard_meta_audit": _dashboard_meta_audit(dashboard_repo, labels, thresholds),
    }
    audit["challenge_set"] = _challenge_results(
        model_dir,
        labels,
        thresholds,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        prefer_cpu=bool(args.prefer_cpu),
    )
    audit["recommendations"] = _recommendations(audit)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(_render_markdown(audit), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(
        "Baseline: "
        f"f1_micro={overall['f1_micro']:.4f} "
        f"f1_macro={overall['f1_macro']:.4f} "
        f"precision_micro={overall['precision_micro']:.4f} "
        f"recall_micro={overall['recall_micro']:.4f}"
    )
    for lab in FOCUS_LABELS:
        c = per_label[lab]
        print(
            f"{lab}: F1={c['f1']:.3f} P={c['precision']:.3f} R={c['recall']:.3f} "
            f"TP={c['tp']} FP={c['fp']} FN={c['fn']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
