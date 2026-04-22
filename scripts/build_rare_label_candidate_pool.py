#!/usr/bin/env python3
"""
Build a candidate pool for rare labels from the processed geolocated dataset.

Targets:
- guvenlik
- psikolojik
- bilgi_paylasimi

The script:
1. Deduplicates by tweet id.
2. Excludes ids already present in the combined gold set.
3. Exact-deduplicates on normalized tweet text to reduce repeated reposts.
4. Scores each row with keyword / pattern rules per rare label.
5. Selects the top-N candidates per label and exports their union as a CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata as ud
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


TARGET_LABELS: List[str] = ["guvenlik", "psikolojik", "bilgi_paylasimi"]

OUTPUT_COLUMNS: List[str] = [
    "id",
    "created_at",
    "date",
    "time",
    "neighborhood",
    "district",
    "province",
    "urgency_score",
    "total_engagement",
    "tweet",
    "tweet_clean",
    "candidate_guvenlik",
    "candidate_psikolojik",
    "candidate_bilgi_paylasimi",
    "candidate_labels",
    "selected_for_labels",
    "label_match_count",
    "guvenlik_score",
    "psikolojik_score",
    "bilgi_paylasimi_score",
    "guvenlik_rules",
    "psikolojik_rules",
    "bilgi_paylasimi_rules",
    "selection_reason",
]


def norm_text(s: str) -> str:
    s = (s or "").strip().casefold()
    s = s.replace("\u0131", "i")
    s = ud.normalize("NFKD", s)
    s = "".join(ch for ch in s if ud.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


LABEL_RULES: Dict[str, Dict[str, Sequence[Tuple[str, int, re.Pattern[str]]]]] = {
    "guvenlik": {
        "positive": [
            ("loot_terms", 4, _compile(r"\byagmac\w*\b|\byagma\s+var\b|\bhirsiz\w*\b|\bhirsizlik\b")),
            ("armed_terms", 4, _compile(r"\bsilahli\b|\bsilahlarla\b|\bsilah\b|\bgasp\b")),
            ("security_request", 3, _compile(r"\bguvenlik\s+lazim\b|\bguvenlik\s+yok\b|\basayis\b")),
            ("safety_risk", 3, _compile(r"\bcan\s+guvenlig\w*\b|\bkolluk\s+kuvvet\w*\b|\bpolis\s+asker\b")),
        ],
        "negative": [
            ("institution_noise", 4, _compile(r"\bsahil\s+guvenlik\b|\bsosyal\s+guvenlik\b|\bsilahli\s+kuvvet\w*\b")),
            ("address_noise", 3, _compile(r"\bguvenlik\s+caddesi\b|\bguvenlik\s+sistem\w*\b|\bguvenlik\s+gorevl\w*\b")),
            ("context_noise", 2, _compile(r"\bguvenlik\s+nedeniyle\b|\basayis\s+burosu\b|\basayis\s+ekip\w*\b")),
            ("name_noise", 2, _compile(r"\bsilah\s+arkadas\w*\b|\bsaldirgan\s+degil\w*\b")),
        ],
    },
    "psikolojik": {
        "positive": [
            ("explicit_support", 5, _compile(r"\bpsikolojik\s+destek\b|\bpsikolog\w*\b|\bmoral\s+desteg\w*\b")),
            ("trauma_terms", 4, _compile(r"\btravma\w*\b|\btravmatik\b|\bpanik\s+atak\b|\bstres\w*\b|\bdepresyon\b|\bsokta\b")),
            ("fear_terms", 2, _compile(r"\bkorkudan\b|\bcok\s+kork\w*\b|\bpanik\w*\b|\bciglik\s+ciglig\w*\b")),
            ("vulnerable_group", 1, _compile(r"\bcocuk\w*\b|\bkadin\w*\b|\byasli\w*\b|\banne\b|\banneler\b")),
        ],
        "negative": [
            ("profession_noise", 4, _compile(r"\bspor\s+psikolog\w*\b|\bpsikoloji\s+bolum\w*\b")),
        ],
    },
    "bilgi_paylasimi": {
        "positive": [
            ("announcement", 5, _compile(r"\bduyuru\b|\bbilgilendirme\b|\bduyurulur\b")),
            (
                "contact_request",
                5,
                _compile(
                    r"\bbilgi\s+vere\w*\b|\bgoren\s+duyan\b|\bhaber\s+alan\b|\bhaber\s+veren\b|"
                    r"\biletisime\s+gecsin\b|\bulasabilen\s+olursa\b|\bpaylasabilecek\b|"
                    r"\bbu\s+konumla\s+ilgili\s+bir\s+bilgi\b|\bbilgisi\s+olan\b"
                ),
            ),
            (
                "aid_ops",
                5,
                _compile(
                    r"\bbagis\b|\byardim\s+toplama\b|\bteslim\s+nokta\w*\b|\bdagitim\s+nokta\w*\b|"
                    r"\birtibat\b|\bkoordinasyon\b|\bdepo\b|\bdestek\s+noktasi\b"
                ),
            ),
            ("contact_line", 3, _compile(r"\btelefon\b|\bnumara\b|\bcep\s+numara\w*\b|\biletisim\b")),
            ("share_call", 1, _compile(r"\bpaylas\w*\b|\byayalim\b|\brt\s+yap\w*\b|\btweet\s+atabilir\w*\b")),
            ("address_list", 2, _compile(r"\badresler\b|\bliste\b|\badres\s+listesi\b")),
            ("missing_info", 4, _compile(r"\bhaber\s+alamiyor\w*\b|\bkayip\b|\bnerede\s+oldug\w*\b")),
            ("phone_like", 2, _compile(r"\b(?:\+?90\s*)?\d{10,11}\b")),
        ],
        "negative": [],
    },
}


def evaluate_label(label: str, text_norm: str) -> Tuple[int, List[str]]:
    rules = LABEL_RULES[label]
    positive_hits: List[str] = []
    negative_hits: List[str] = []
    score = 0

    for rule_name, weight, pattern in rules["positive"]:
        if pattern.search(text_norm):
            positive_hits.append(rule_name)
            score += int(weight)

    for rule_name, penalty, pattern in rules["negative"]:
        if pattern.search(text_norm):
            negative_hits.append(f"-{rule_name}")
            score -= int(penalty)

    if label == "guvenlik":
        is_candidate = score >= 4 and any(hit in positive_hits for hit in ["loot_terms", "armed_terms", "security_request", "safety_risk"])
    elif label == "psikolojik":
        is_candidate = (
            "explicit_support" in positive_hits
            or "trauma_terms" in positive_hits
            or ("fear_terms" in positive_hits and "vulnerable_group" in positive_hits)
        )
    elif label == "bilgi_paylasimi":
        is_candidate = (
            "announcement" in positive_hits
            or "contact_request" in positive_hits
            or "aid_ops" in positive_hits
            or "missing_info" in positive_hits
            or (("share_call" in positive_hits) and ("contact_line" in positive_hits or "phone_like" in positive_hits))
            or score >= 6
        )
    else:
        is_candidate = score > 0

    all_hits = positive_hits + negative_hits
    return (max(score, 0) if is_candidate else 0, all_hits)


def selection_reason(row: pd.Series) -> str:
    parts: List[str] = []
    for label in TARGET_LABELS:
        score = int(row[f"{label}_score"])
        rules = str(row[f"{label}_rules"]).strip()
        if score > 0:
            parts.append(f"{label}:{score} [{rules}]")
    return " | ".join(parts)


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["text_norm"] = work["tweet_clean"].fillna(work["tweet"]).fillna("").astype("string").map(lambda x: norm_text(str(x)))

    for label in TARGET_LABELS:
        scores: List[int] = []
        rules_text: List[str] = []
        flags: List[int] = []
        for text in work["text_norm"].tolist():
            score, hits = evaluate_label(label, text)
            scores.append(int(score))
            rules_text.append(", ".join(hits))
            flags.append(int(score > 0))
        work[f"{label}_score"] = scores
        work[f"{label}_rules"] = rules_text
        work[f"candidate_{label}"] = flags

    work["candidate_labels"] = work.apply(
        lambda r: ",".join([label for label in TARGET_LABELS if int(r[f"candidate_{label}"]) == 1]),
        axis=1,
    )
    work["label_match_count"] = work.apply(lambda r: int(sum(int(r[f"candidate_{label}"]) for label in TARGET_LABELS)), axis=1)
    work = work[work["label_match_count"] > 0].copy()
    return work


def _top_by_label(df: pd.DataFrame, label: str, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return df.iloc[0:0].copy()
    sub = df[df[f"candidate_{label}"] == 1].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(
        by=[f"{label}_score", "label_match_count", "urgency_score", "total_engagement"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return sub.head(limit).copy()


def write_report(
    *,
    report_path: Path,
    summary_path: Path,
    stats: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("RARE LABEL CANDIDATE POOL REPORT")
    lines.append("")
    lines.append(f"- processed rows read: {stats['rows_read']}")
    lines.append(f"- unique ids after id dedup: {stats['rows_after_id_dedup']}")
    lines.append(f"- rows after excluding current gold ids: {stats['rows_after_gold_exclusion']}")
    lines.append(f"- rows after normalized-text dedup: {stats['rows_after_text_dedup']}")
    lines.append(f"- candidate rows before per-label caps: {stats['candidate_rows_before_caps']}")
    lines.append(f"- exported pool rows: {stats['exported_rows']}")
    lines.append("")
    lines.append("LABEL COUNTS BEFORE CAPS")
    for label, value in stats["label_counts_before_caps"].items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append("LABEL COUNTS AFTER CAPS")
    for label, value in stats["label_counts_after_caps"].items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append("MULTI-LABEL CANDIDATES")
    lines.append(f"- rows matching 2+ target labels: {stats['multi_label_rows']}")
    lines.append("")
    lines.append("TOP RULE USAGE")
    for label, counts in stats["top_rule_usage"].items():
        lines.append(f"- {label}: {counts}")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate pool for rare-label manual annotation.")
    parser.add_argument("--input", default="data/processed/emergency_geolocated_96k.csv", help="Processed input CSV.")
    parser.add_argument("--gold", default="data/need_classification_gold_combined.csv", help="Combined gold CSV.")
    parser.add_argument(
        "--output",
        default="data/labeling/need_classification_rare_label_candidates.csv",
        help="Output candidate CSV path.",
    )
    parser.add_argument(
        "--report",
        default="data/analysis/need_classification_rare_label_candidates.report.txt",
        help="Human-readable report path.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/analysis/need_classification_rare_label_candidates.summary.json",
        help="Machine-readable summary path.",
    )
    parser.add_argument("--max-guvenlik", type=int, default=120, help="Max exported candidates for guvenlik.")
    parser.add_argument("--max-psikolojik", type=int, default=120, help="Max exported candidates for psikolojik.")
    parser.add_argument(
        "--max-bilgi-paylasimi",
        type=int,
        default=250,
        help="Max exported candidates for bilgi_paylasimi.",
    )
    parser.add_argument("--min-text-len", type=int, default=20, help="Minimum normalized text length.")
    args = parser.parse_args()

    input_path = Path(args.input)
    gold_path = Path(args.gold)
    output_path = Path(args.output)
    report_path = Path(args.report)
    summary_path = Path(args.summary_json)

    usecols = [
        "id",
        "created_at",
        "date",
        "time",
        "tweet",
        "tweet_clean",
        "urgency_score",
        "total_engagement",
        "neighborhood",
        "district",
        "province",
    ]
    df = pd.read_csv(input_path, encoding="utf-8-sig", usecols=lambda c: c in usecols)
    rows_read = int(len(df))

    for col in usecols:
        if col not in df.columns:
            df[col] = ""

    df["id"] = df["id"].astype("string").fillna("").str.strip()
    df = df[df["id"] != ""].copy()
    df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0).astype(int)
    df["total_engagement"] = pd.to_numeric(df["total_engagement"], errors="coerce").fillna(0).astype(int)
    df["tweet_clean"] = df["tweet_clean"].astype("string")
    df["tweet"] = df["tweet"].astype("string")

    df = df.sort_values(by=["urgency_score", "total_engagement"], ascending=[False, False], kind="mergesort")
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    rows_after_id_dedup = int(len(df))

    gold_ids = set(pd.read_csv(gold_path, encoding="utf-8-sig", usecols=["id"])["id"].astype("string").fillna("").str.strip().tolist())
    df = df[~df["id"].isin(gold_ids)].copy()
    rows_after_gold_exclusion = int(len(df))

    df["text_norm"] = df["tweet_clean"].fillna(df["tweet"]).fillna("").astype("string").map(lambda x: norm_text(str(x)))
    df = df[df["text_norm"].str.len() >= int(args.min_text_len)].copy()
    df = df.drop_duplicates(subset=["text_norm"], keep="first").copy()
    rows_after_text_dedup = int(len(df))

    candidates = build_candidates(df)
    candidate_rows_before_caps = int(len(candidates))

    selected_parts = [
        _top_by_label(candidates, "guvenlik", int(args.max_guvenlik)),
        _top_by_label(candidates, "psikolojik", int(args.max_psikolojik)),
        _top_by_label(candidates, "bilgi_paylasimi", int(args.max_bilgi_paylasimi)),
    ]
    selected = pd.concat(selected_parts, ignore_index=False).drop_duplicates(subset=["id"], keep="first").copy()

    selected["selected_for_labels"] = selected.apply(
        lambda r: ",".join(
            [
                label
                for label, limit in [
                    ("guvenlik", int(args.max_guvenlik)),
                    ("psikolojik", int(args.max_psikolojik)),
                    ("bilgi_paylasimi", int(args.max_bilgi_paylasimi)),
                ]
                if int(r[f"candidate_{label}"]) == 1
            ]
        ),
        axis=1,
    )
    selected["selection_reason"] = selected.apply(selection_reason, axis=1)

    selected = selected.sort_values(
        by=["label_match_count", "guvenlik_score", "psikolojik_score", "bilgi_paylasimi_score", "urgency_score", "total_engagement"],
        ascending=[False, False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected[OUTPUT_COLUMNS].to_csv(output_path, index=False, encoding="utf-8-sig")

    top_rule_usage: Dict[str, Dict[str, int]] = {}
    for label in TARGET_LABELS:
        counts: Dict[str, int] = {}
        for rule_blob in selected.loc[selected[f"candidate_{label}"] == 1, f"{label}_rules"].fillna("").astype(str).tolist():
            for item in [x.strip() for x in rule_blob.split(",") if x.strip() and not x.strip().startswith("-")]:
                counts[item] = counts.get(item, 0) + 1
        top_rule_usage[label] = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8])

    stats: Dict[str, object] = {
        "rows_read": rows_read,
        "rows_after_id_dedup": rows_after_id_dedup,
        "rows_after_gold_exclusion": rows_after_gold_exclusion,
        "rows_after_text_dedup": rows_after_text_dedup,
        "candidate_rows_before_caps": candidate_rows_before_caps,
        "exported_rows": int(len(selected)),
        "label_counts_before_caps": {
            label: int(candidates[f"candidate_{label}"].sum()) for label in TARGET_LABELS
        },
        "label_counts_after_caps": {
            label: int(selected[f"candidate_{label}"].sum()) for label in TARGET_LABELS
        },
        "multi_label_rows": int((selected["label_match_count"] >= 2).sum()),
        "top_rule_usage": top_rule_usage,
        "output_csv": str(output_path.as_posix()),
        "caps": {
            "guvenlik": int(args.max_guvenlik),
            "psikolojik": int(args.max_psikolojik),
            "bilgi_paylasimi": int(args.max_bilgi_paylasimi),
        },
    }
    write_report(report_path=report_path, summary_path=summary_path, stats=stats)

    print(f"Wrote candidate CSV: {output_path}")
    print(f"Wrote report TXT: {report_path}")
    print(f"Wrote summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
