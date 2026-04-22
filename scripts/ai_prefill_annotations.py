#!/usr/bin/env python3
"""
Rule-based prefill for annotation CSVs.

This is NOT a replacement for human annotation. It produces "pseudo-labels" that
can be used to:
- speed up manual labeling (human edits the prefill)
- bootstrap baselines / sanity checks

It fills:
- need labels (binary multi-label)
- aciliyet_0_3 (0-3)
- veracity_label (dogrulanmis / supheli / asilsiz) [mostly supheli]
"""

from __future__ import annotations

import argparse
import re
import unicodedata as ud
from pathlib import Path
from typing import Dict, List

import pandas as pd


NEED_COLS: List[str] = [
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

AUX_COLS: List[str] = ["aciliyet_0_3", "veracity_label", "notes"]


def norm(s: str) -> str:
    # Turkish-safe ASCII-ish normalization:
    # - casefold for better unicode case mapping
    # - map dotless i to i
    # - strip diacritics (NFKD + remove Mn)
    s = (s or "").strip().casefold()
    s = s.replace("\u0131", "i")  # dotless i
    s = ud.normalize("NFKD", s)
    s = "".join(ch for ch in s if ud.category(ch) != "Mn")
    return s


def compile_patterns(profile: str) -> Dict[str, List[re.Pattern]]:
    # Profile B is a bit more conservative for rescue cues.
    if profile not in {"A", "B"}:
        raise ValueError("profile must be A or B")

    rescue = (
        [
            r"\benkaz\w*\b",
            r"\bgocuk\w*\b",
            r"\bgocug\w*\b",  # gocuk -> gocugun
            r"\bmahsur\w*\b",
            r"\bkurtar\w*",
            r"\bses\w*\s+geliyor\b",
            r"\byardim\s+ses\w*\b",
            r"\bcanli\s+var\b",
            r"\bcan\s+var\b",
            r"\byikilmak\s+uzere\b",
            r"\bciglik\w*\b",
            r"\bimdat\b",
        ]
        if profile == "A"
        else [
            r"\benkaz\s+alt",
            r"\benkazalt\w*",
            r"\bgocuk\s+alt",
            r"\bgocukalt\w*",
            r"\bmahsur\w*\b",
            r"\bses\w*\s+geliyor\b",
            r"\byardim\s+ses\w*\b",
            r"\bcanli\s+var\b",
            r"\bcan\s+var\b",
            r"\byikilmak\s+uzere\b",
        ]
    )

    pat = {
        "arama_kurtarma": rescue,
        "saglik": [
            r"\byarali\w*\b",
            r"\bambulans\b",
            r"\bdoktor\b",
            r"\bhastane\b",
            r"\bilac\b",
            r"\bkan\b",
            r"\bserum\b",
            r"\bameli\w*\b",
            r"\bdogum\b",
        ],
        "barinma": [
            r"\bcadir\b",
            r"\bbattaniye\b",
            r"\bbarinma\b",
            r"\bisin\w*\b",
            r"\bsoguk\b",
            r"\bdonuyor\w*\b",
            r"\bkalacak\s+yer\b",
            r"\bisitici\b",
            r"\bsoba\b",
            r"\buyku\s+tulumu\b",
            r"\bodun\b",
        ],
        "gida_su": [
            r"\bsusuz\b",
            r"\bsu\s+yok\b",
            r"\bsu\s+kalmad\w*\b",
            r"\bsu\s+lazim\b",
            r"\bsu\s+ihtiyac\w*\b",
            r"\bgida\b",
            r"\byemek\b",
            r"\bekmek\b",
            r"\berzak\b",
            r"\bmama\b",
            r"\bbebek\s+mam\w*\b",
            r"\bicecek\b",
        ],
        "altyapi": [
            r"\belektrik\b",
            r"\bsu\s+kesint\w*\b",
            r"\binternet\b",
            r"\bsebeke\b",
            r"\byol\s+kapal\w*\b",
            r"\bdogalgaz\b",
        ],
        "guvenlik": [
            r"\byagma\b",
            r"\bhirsiz\w*\b",
            r"\bguvenlik\b",
            r"\basayis\b",
            r"\bsilah\b",
        ],
        "lojistik": [
            r"\bvinc\b",
            r"\bis\s+makinesi\b",
            r"\bbeton\s+kes\w*\b",
            r"\bbeton\s+kiric\w*\b",
            r"\bjenerator\b",
            r"\byakit\b",
            r"\bbenzin\b",
            r"\baku\b",
            r"\btermal\s+kamera\b",
            r"\bses\s+kayit\b",
            r"\bekip\w*\b",
            r"\bekipman\b",
            r"\bpersonel\b",
            r"\barac\b",
            r"\bkamyon\b",
            r"\btir\b",
            r"\bsevkiyat\b",
            r"\bkoordin\w*\b",
            r"\byonlendir\w*\b",
        ],
        # Psychological support cues. Use `\w*` to catch common Turkish suffixes
        # (e.g., "korkudan", "panikledim"). Keep it reasonably specific to avoid
        # false positives from surnames/addresses like "Korkmaz".
        "psikolojik": [
            r"\bpsikoloj\w*\b",  # psikoloji, psikolojik, psikolojisi...
            r"\btravma\w*\b",
            r"\bpanik\w*\b",
            r"\bkorku\w*\b",
            r"\bstres\w*\b",
        ],
        "bilgi_paylasimi": [
            r"\bduyuru\b",
            r"\bbilgi\b",
            r"\bnumara\b",
            r"\bpaylas\w*\b",
            r"\bafad\b\s+numara",
        ],
        "urgent_words": [r"\bacil\b", r"\bcok\s+acil\b", r"\bimdat\b", r"\bson\s+dakika\b"],
        "helpish_words": [r"\byardim\w*\b", r"\byardim\s+edin\b", r"\byardim\s+lazim\b", r"\bekip\s+lazim\b"],
        "verified": [r"\bteyitli\b", r"\bdogruland\w*\b", r"\bconfirmed\b", r"\bverified\b"],
    }

    return {k: [re.compile(x, flags=re.IGNORECASE) for x in v] for k, v in pat.items()}


def match_any(t: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(t) for p in pats)


def label_text(text: str, compiled: Dict[str, List[re.Pattern]]) -> Dict[str, str]:
    t = norm(text)

    labels = {}
    labels["arama_kurtarma"] = int(match_any(t, compiled["arama_kurtarma"]))
    labels["saglik"] = int(match_any(t, compiled["saglik"]))
    labels["barinma"] = int(match_any(t, compiled["barinma"]))
    labels["gida_su"] = int(match_any(t, compiled["gida_su"]))
    labels["altyapi"] = int(match_any(t, compiled["altyapi"]))
    labels["guvenlik"] = int(match_any(t, compiled["guvenlik"]))
    labels["lojistik"] = int(match_any(t, compiled["lojistik"]))
    labels["psikolojik"] = int(match_any(t, compiled["psikolojik"]))

    urgent = match_any(t, compiled["urgent_words"])
    helpish = match_any(t, compiled["helpish_words"])

    any_need = any(labels[k] == 1 for k in ["arama_kurtarma", "saglik", "barinma", "gida_su", "altyapi", "guvenlik", "lojistik", "psikolojik"])

    labels["bilgi_paylasimi"] = int((not any_need) and match_any(t, compiled["bilgi_paylasimi"]) and (not urgent) and (not helpish))

    # aciliyet_0_3 heuristic
    ac = 0
    if labels["arama_kurtarma"] == 1:
        ac = 3
    if labels["saglik"] == 1:
        ac = max(ac, 3)
    if labels["guvenlik"] == 1:
        ac = max(ac, 2)
    if labels["barinma"] == 1 or labels["gida_su"] == 1:
        ac = max(ac, 2 if urgent or re.search(r"\bdonuyor\w*\b|\b\d+\s+gundur\b", t) else 1)
    if labels["altyapi"] == 1 and ac < 2:
        ac = max(ac, 1)
    if labels["lojistik"] == 1 and labels["arama_kurtarma"] == 0 and ac < 2:
        ac = max(ac, 1)
    if (not any_need) and (urgent or helpish):
        ac = max(ac, 1)

    ver = "supheli"
    if match_any(t, compiled["verified"]):
        ver = "dogrulanmis"

    out: Dict[str, str] = {k: str(v) for k, v in labels.items()}
    out["aciliyet_0_3"] = str(ac)
    out["veracity_label"] = ver
    out["notes"] = ""
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Rule-based prefill for annotation CSVs.")
    p.add_argument("--input", required=True, help="Input CSV (template / annotator file).")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--profile", choices=["A", "B"], default="A", help="Prefill profile.")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", dtype=str)

    for c in NEED_COLS + AUX_COLS:
        if c not in df.columns:
            df[c] = ""

    compiled = compile_patterns(args.profile)

    for i, row in df.iterrows():
        text = row.get("tweet_clean") or row.get("tweet") or ""
        lab = label_text(text, compiled)
        for c in NEED_COLS:
            df.at[i, c] = lab.get(c, "0")
        df.at[i, "aciliyet_0_3"] = lab.get("aciliyet_0_3", "0")
        df.at[i, "veracity_label"] = lab.get("veracity_label", "supheli")
        df.at[i, "notes"] = lab.get("notes", "")

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
