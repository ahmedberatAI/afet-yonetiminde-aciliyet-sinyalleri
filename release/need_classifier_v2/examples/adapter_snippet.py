"""
Minimal Python adapter for need_classifier_v2.

Release paketinden pivot eden bir sample. Repo kökü release/need_classifier_v2/
dizininin iki üstü kabul edilir. Bu dosya tek başına çalışır; kütüphane olarak
konumlandırılmamıştır.

Çalıştırma:
    python release/need_classifier_v2/examples/adapter_snippet.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RELEASE_DIR = Path(__file__).resolve().parents[1]          # release/need_classifier_v2/
REPO_ROOT = RELEASE_DIR.parents[1]                         # repo root

MODEL_DIR = REPO_ROOT / "models" / "exp3_silver_then_gold_v3_exgold" / "final"
LABELS_JSON = RELEASE_DIR / "label_columns.json"
THRESHOLDS_JSON = RELEASE_DIR / "thresholds" / "thresholds_cv.json"
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


def load_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, List[str], Dict[str, float]]:
    if not MODEL_DIR.exists():
        raise SystemExit(f"Model dir missing: {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    labels: List[str] = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    thresholds: Dict[str, float] = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
    return tok, mdl, labels, thresholds


def _normalize_rule_text(text: str) -> str:
    s = unicodedata.normalize("NFC", str(text or "")).casefold()
    return " ".join(s.split())


def _has_info_postprocess_signal(text: str) -> bool:
    t = _normalize_rule_text(text)
    missing = bool(INFO_MISSING_RE.search(t))
    request = bool(INFO_REQUEST_RE.search(t))
    contact = bool(INFO_CONTACT_RE.search(t))
    announcement = bool(INFO_ANNOUNCEMENT_RE.search(t))
    return (missing and request) or (missing and contact) or (request and contact) or announcement


def apply_info_v1_postprocess(text: str, labels: List[str], probs: np.ndarray, preds: Dict[str, int]) -> None:
    if "bilgi_paylasimi" not in labels:
        return
    j = labels.index("bilgi_paylasimi")
    if preds.get("bilgi_paylasimi", 0) == 0 and probs[j] >= INFO_POSTPROCESS_MIN_PROB:
        if _has_info_postprocess_signal(text):
            preds["bilgi_paylasimi"] = 1


def predict(
    texts: List[str],
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    labels: List[str],
    thresholds: Dict[str, float],
    max_length: int = 192,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    postprocess_profile: str = "info_v1",
) -> List[Dict[str, object]]:
    mdl = mdl.to(device)
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()

    out: List[Dict[str, object]] = []
    for i, text in enumerate(texts):
        row: Dict[str, object] = {"text": text, "probs": {}, "preds": {}}
        for j, lab in enumerate(labels):
            p = float(probs[i, j])
            row["probs"][lab] = p
            row["preds"][lab] = int(p >= thresholds[lab])
        if postprocess_profile == "info_v1":
            apply_info_v1_postprocess(text, labels, probs[i], row["preds"])
        elif postprocess_profile != "none":
            raise ValueError(f"Unsupported postprocess_profile: {postprocess_profile}")
        row["pred_label_count"] = int(sum(row["preds"].values()))
        row["pred_any_need"] = int(row["pred_label_count"] >= 1)
        out.append(row)
    return out


def main() -> int:
    tok, mdl, labels, thresholds = load_model()
    samples = [
        "Deprem sonrası enkaz altındayız, yardım edin lütfen Antakya",
        "Gıda ve su ulaşmadı, 3 gündür bekliyoruz",
        "Sağlık sorunu var, insülin lazım",
        "Haber alamıyoruz, gören var mı bu adresi",
    ]
    rows = predict(samples, tok, mdl, labels, thresholds)
    for r in rows:
        fired = [l for l, v in r["preds"].items() if v == 1]
        print(f"text: {r['text']}")
        print(f"  fired: {fired or '(none)'}")
        print(f"  pred_any_need: {r['pred_any_need']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
