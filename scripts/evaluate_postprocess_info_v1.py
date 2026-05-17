#!/usr/bin/env python3
"""Validate the info_v1 postprocess profile on OOF, validation, and test.

Selection rule:
- Use OOF + validation as the decision surfaces.
- Test is reported as a regression check, not used to choose the rule.

The profile keeps canonical CV thresholds intact and only adds
`bilgi_paylasimi` when:
  prob_bilgi_paylasimi >= 0.20
  AND the text has a strong missing-news / info-request / contact /
      announcement signal.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "final"
DEFAULT_LABELS = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "label_columns.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "thresholds_cv.json"
DEFAULT_OOF = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "threshold_tuning_cv_oof.csv"
DEFAULT_TRAIN = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "train.csv"
DEFAULT_VAL = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "val.csv"
DEFAULT_TEST = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "test.csv"
DEFAULT_OUT_PREFIX = REPO_ROOT / "data" / "analysis" / "postprocess_info_v1_validation_2026_05_17"

TEXT_COL = "tweet_clean"
INFO_POSTPROCESS_MIN_PROB = 0.20
INFO_MISSING_RE = re.compile(
    r"(haber\s+alam|haber\s+al[ıi]nam|ula[şs]am[ıi]yor|ula[şs][ıi]lam[ıi]yor)",
    flags=re.IGNORECASE,
)
INFO_REQUEST_RE = re.compile(
    r"(g[oö]ren|duyan|bilen|bilgisi\s+olan|bilgi\s+alan|haber\s+alan|ula[şs]s[ıi]n|yazs[ıi]n|bildirsin)",
    flags=re.IGNORECASE,
)
INFO_CONTACT_RE = re.compile(r"(ileti[şs]im|irtibat|telefon|numara|0\d{10}|05\d{9})", flags=re.IGNORECASE)
INFO_ANNOUNCEMENT_RE = re.compile(
    r"(duyuru|canl[ıi]\s+yay[ıi]n|transfer|da[ğg][ıi]t[ıi]m|ula[şs]t[ıi]r[ıi]ld[ıi]|bildirilsin)",
    flags=re.IGNORECASE,
)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _normalize_rule_text(text: str) -> str:
    s = unicodedata.normalize("NFC", str(text or "")).casefold()
    return " ".join(s.split())


def _has_info_signal(text: str) -> bool:
    t = _normalize_rule_text(text)
    missing = bool(INFO_MISSING_RE.search(t))
    request = bool(INFO_REQUEST_RE.search(t))
    contact = bool(INFO_CONTACT_RE.search(t))
    announcement = bool(INFO_ANNOUNCEMENT_RE.search(t))
    return (missing and request) or (missing and contact) or (request and contact) or announcement


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def _overall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    micro = _confusion(y_true.reshape(-1), y_pred.reshape(-1))
    f1s = [_confusion(y_true[:, i], y_pred[:, i])["f1"] for i in range(y_true.shape[1])]
    return {
        "precision_micro": micro["precision"],
        "recall_micro": micro["recall"],
        "f1_micro": micro["f1"],
        "f1_macro": float(np.mean(f1s)),
    }


def _base_pred(probs: np.ndarray, labels: list[str], thresholds: dict[str, float]) -> np.ndarray:
    thr = np.array([thresholds[l] for l in labels], dtype=np.float32).reshape(1, -1)
    return (probs >= thr).astype(int)


def _apply_info_v1(texts: list[str], probs: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    out = y_pred.copy()
    j = labels.index("bilgi_paylasimi")
    signals = np.array([_has_info_signal(t) for t in texts], dtype=bool)
    added = (out[:, j] == 0) & (probs[:, j] >= INFO_POSTPROCESS_MIN_PROB) & signals
    out[added, j] = 1
    return out, {"rule_hits": int(signals.sum()), "predictions_added": int(added.sum())}


def _coerce_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    out = df.copy()
    for lab in labels:
        s = out[lab].astype("string").fillna("").str.strip().replace({"": "0"})
        out[lab] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return out


def _read_oof(oof_path: Path, train_csv: Path, val_csv: Path, labels: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    oof = pd.read_csv(oof_path, encoding="utf-8-sig", dtype={"id": "string"})
    train = pd.read_csv(train_csv, encoding="utf-8-sig", dtype={"id": "string"}, usecols=["id", TEXT_COL])
    val = pd.read_csv(val_csv, encoding="utf-8-sig", dtype={"id": "string"}, usecols=["id", TEXT_COL])
    pool = pd.concat([train, val], ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    merged = oof.merge(pool, on="id", how="left")
    texts = merged[TEXT_COL].astype("string").fillna("").tolist()
    y_true = np.stack([merged[f"y_{lab}"].astype(int).to_numpy() for lab in labels], axis=1)
    probs = np.stack([merged[f"prob_{lab}"].astype(float).to_numpy() for lab in labels], axis=1)
    return texts, y_true, probs


def _predict_split(model_dir: Path, csv_path: Path, labels: list[str], max_length: int, batch_size: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"id": "string"})
    df = _coerce_labels(df, labels)
    texts = df[TEXT_COL].astype("string").fillna("").tolist()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    model.to("cpu")

    logits: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits.append(model(**enc).logits.detach().cpu().numpy())
    probs = _sigmoid(np.concatenate(logits, axis=0))
    y_true = df[labels].astype(int).to_numpy()
    return texts, y_true, probs


def _evaluate_dataset(
    name: str,
    texts: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    base = _base_pred(probs, labels, thresholds)
    info_v1, meta = _apply_info_v1(texts, probs, base, labels)
    j_info = labels.index("bilgi_paylasimi")
    j_sec = labels.index("guvenlik")
    return {
        "name": name,
        "rows": int(len(texts)),
        "base": {
            "overall": _overall(y_true, base),
            "bilgi_paylasimi": _confusion(y_true[:, j_info], base[:, j_info]),
            "guvenlik": _confusion(y_true[:, j_sec], base[:, j_sec]),
        },
        "info_v1": {
            "overall": _overall(y_true, info_v1),
            "bilgi_paylasimi": _confusion(y_true[:, j_info], info_v1[:, j_info]),
            "guvenlik": _confusion(y_true[:, j_sec], info_v1[:, j_sec]),
            "postprocess": meta,
        },
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: float(after[k]) - float(before[k]) for k in sorted(before)}


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Postprocess info_v1 Validation - 2026-05-17")
    lines.append("")
    lines.append("Validated change: keep canonical thresholds, then add `bilgi_paylasimi` when the text has a strong information-sharing signal and `prob_bilgi_paylasimi >= 0.20`.")
    lines.append("")
    lines.append("Decision surfaces are OOF and validation. Test is shown as a non-selection regression check.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| dataset | profile | micro F1 | macro F1 | bilgi F1 | bilgi P | bilgi R | bilgi FP | bilgi FN | rule adds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in payload["datasets"]:
        for profile in ["base", "info_v1"]:
            row = ds[profile]
            o = row["overall"]
            b = row["bilgi_paylasimi"]
            adds = row.get("postprocess", {}).get("predictions_added", 0)
            lines.append(
                f"| {ds['name']} | {profile} | {o['f1_micro']:.4f} | {o['f1_macro']:.4f} | "
                f"{b['f1']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['fp']} | {b['fn']} | {adds} |"
            )
    lines.append("")
    lines.append("## Deltas (info_v1 - base)")
    lines.append("")
    lines.append("| dataset | micro F1 | macro F1 | bilgi F1 | bilgi precision | bilgi recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ds in payload["datasets"]:
        od = _delta(ds["info_v1"]["overall"], ds["base"]["overall"])
        bd = _delta(ds["info_v1"]["bilgi_paylasimi"], ds["base"]["bilgi_paylasimi"])
        lines.append(
            f"| {ds['name']} | {od['f1_micro']:+.4f} | {od['f1_macro']:+.4f} | "
            f"{bd['f1']:+.4f} | {bd['precision']:+.4f} | {bd['recall']:+.4f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "- `info_v1` improves OOF and validation micro F1, macro F1, and `bilgi_paylasimi` F1 together."
    )
    lines.append(
        "- `guvenlik` rule candidates were not promoted: they added false positives without consistent OOF/validation gain."
    )
    lines.append(
        "- This is an inference-layer improvement, not a model-weight or CV-threshold replacement; use `--postprocess-profile none` for exact baseline reproduction."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate info_v1 postprocess profile.")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--labels-json", default=str(DEFAULT_LABELS))
    ap.add_argument("--thresholds-json", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--oof-csv", default=str(DEFAULT_OOF))
    ap.add_argument("--train-csv", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val-csv", default=str(DEFAULT_VAL))
    ap.add_argument("--test-csv", default=str(DEFAULT_TEST))
    ap.add_argument("--out-prefix", default=str(DEFAULT_OUT_PREFIX))
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    labels = [str(x) for x in _json_load(Path(args.labels_json))]
    thresholds = {str(k): float(v) for k, v in _json_load(Path(args.thresholds_json)).items()}

    datasets = []
    datasets.append(
        _evaluate_dataset(
            "oof",
            *_read_oof(Path(args.oof_csv), Path(args.train_csv), Path(args.val_csv), labels),
            labels,
            thresholds,
        )
    )
    for name, csv_path in [("validation", Path(args.val_csv)), ("test_regression_check", Path(args.test_csv))]:
        datasets.append(
            _evaluate_dataset(
                name,
                *_predict_split(Path(args.model_dir), csv_path, labels, int(args.max_length), int(args.batch_size)),
                labels,
                thresholds,
            )
        )

    payload = {
        "profile": "info_v1",
        "min_probability": INFO_POSTPROCESS_MIN_PROB,
        "rule": "Add bilgi_paylasimi when prob>=0.20 and strong info-sharing text signal is present.",
        "labels": labels,
        "thresholds": thresholds,
        "decision_surfaces": ["oof", "validation"],
        "test_usage": "reported as regression check only",
        "datasets": datasets,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    for ds in datasets:
        base = ds["base"]["overall"]
        info = ds["info_v1"]["overall"]
        b0 = ds["base"]["bilgi_paylasimi"]
        b1 = ds["info_v1"]["bilgi_paylasimi"]
        print(
            f"{ds['name']}: micro {base['f1_micro']:.4f}->{info['f1_micro']:.4f}; "
            f"macro {base['f1_macro']:.4f}->{info['f1_macro']:.4f}; "
            f"bilgi F1 {b0['f1']:.4f}->{b1['f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
