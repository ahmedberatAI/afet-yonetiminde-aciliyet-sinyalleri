#!/usr/bin/env python3
"""
Content-level overlap audit between the LEAK-FREE silver pool and the gold
splits that were used to evaluate/fine-tune the canonical winner.

Purpose
-------
Step 7 v3 closed the id-level silver→gold leak (all 1934 gold_combined ids
were removed from the silver pool). This audit checks what is left:
  - exact normalized-text overlap
  - high-Jaccard near-duplicate overlap (5-char shingles)

If exact overlap is non-zero even after id-exclusion, that means the silver
pool contains tweets with the same cleaned text as gold rows — usually
retweets or alternate ids of the same content. Near-duplicate overlap catches
copy-paste variants (hashtag reorder, handle differences, etc.).

Outputs
-------
  - data/analysis/content_overlap_audit_v2_leakfree.json
  - data/analysis/content_overlap_audit_v2_leakfree.md

Scope
-----
This is a bounded audit meant for the step-9 selection rationale, not a full
near-duplicate study. A gold→silver candidate-retrieval pass is performed
using an inverted index over 5-char shingles; only gold rows with nonzero
candidate signals are scored, keeping runtime manageable on ~60k silver rows.

Deterministic: no random sampling, fixed seed (not strictly needed here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

SILVER_CSV = REPO_ROOT / "data" / "labeling" / "need_classification_silver_63k_profileA_exgold.csv"
GOLD_TRAIN = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "train.csv"
GOLD_VAL = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "val.csv"
GOLD_TEST = REPO_ROOT / "data" / "modeling" / "need_classification_gold_combined" / "test.csv"

OUT_JSON = REPO_ROOT / "data" / "analysis" / "content_overlap_audit_v2_leakfree.json"
OUT_MD = REPO_ROOT / "data" / "analysis" / "content_overlap_audit_v2_leakfree.md"

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#")
WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = re.sub(r"[^\w\sçğıöşüâîû]", " ", t, flags=re.UNICODE)
    t = WS_RE.sub(" ", t).strip()
    return t


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def _shingles(s: str, k: int = 5) -> Set[str]:
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def _read_csv(path: Path, text_col: str = "tweet_clean") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=["id", text_col])
    df[text_col] = df[text_col].fillna("")
    df["id"] = df["id"].fillna("")
    df["_norm"] = df[text_col].map(_normalize)
    df["_hash"] = df["_norm"].map(_sha1)
    return df


def _build_shingle_index(
    silver_df: pd.DataFrame,
    k: int = 5,
    min_token_hits: int = 3,
) -> Tuple[Dict[str, List[int]], List[Set[str]]]:
    """Inverted index: shingle -> list of silver row indices."""
    ix: Dict[str, List[int]] = {}
    shingle_sets: List[Set[str]] = []
    for i, t in enumerate(silver_df["_norm"].tolist()):
        shg = _shingles(t, k)
        shingle_sets.append(shg)
        for s in shg:
            ix.setdefault(s, []).append(i)
    return ix, shingle_sets


def _audit_split(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    silver_hashes: Set[str],
    shingle_index: Dict[str, List[int]],
    silver_shingle_sets: List[Set[str]],
    k: int,
    jaccard_thresh: float,
    top_n_pairs: int,
    max_candidates: int = 200,
) -> Dict[str, Any]:
    exact_overlap: List[Dict[str, Any]] = []
    near_dup_hits: List[Dict[str, Any]] = []
    gold_hashes = set(gold_df["_hash"])
    exact_ids = gold_df.loc[gold_df["_hash"].isin(silver_hashes), "id"].tolist()

    # Summarize exact overlap rows.
    for i, row in gold_df.iterrows():
        h = row["_hash"]
        if h in silver_hashes:
            exact_overlap.append(
                {"gold_id": row["id"], "norm_preview": row["_norm"][:160]}
            )

    # Near-dup scoring: for each gold row, find candidate silver rows via
    # shingle inverted index, then compute Jaccard.
    for _, row in gold_df.iterrows():
        shg = _shingles(row["_norm"], k)
        if not shg:
            continue
        cand_counts: Dict[int, int] = {}
        for s in shg:
            for idx in shingle_index.get(s, ()):
                cand_counts[idx] = cand_counts.get(idx, 0) + 1
        if not cand_counts:
            continue
        # Top-candidate cap to bound runtime.
        top_cands = sorted(cand_counts.items(), key=lambda kv: -kv[1])[:max_candidates]
        best: Optional[Tuple[float, int]] = None
        for idx, _c in top_cands:
            j = _jaccard(shg, silver_shingle_sets[idx])
            if j < jaccard_thresh:
                continue
            if best is None or j > best[0]:
                best = (j, idx)
        if best is not None and best[0] >= jaccard_thresh:
            j, sidx = best
            near_dup_hits.append(
                {
                    "gold_id": row["id"],
                    "silver_id": silver_df.iloc[sidx]["id"],
                    "jaccard": float(j),
                    "gold_norm": row["_norm"][:160],
                    "silver_norm": silver_df.iloc[sidx]["_norm"][:160],
                }
            )

    near_dup_hits.sort(key=lambda r: -r["jaccard"])
    return {
        "gold_rows": int(len(gold_df)),
        "silver_rows": int(len(silver_df)),
        "exact_overlap_count": int(len(exact_overlap)),
        "exact_overlap_samples": exact_overlap[:top_n_pairs],
        "near_dup_threshold": jaccard_thresh,
        "near_dup_hit_count": int(len(near_dup_hits)),
        "near_dup_top": near_dup_hits[:top_n_pairs],
    }


def _render_md(payload: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Content-overlap audit (v2, leak-free silver vs gold splits)\n")
    L.append(
        "Bu audit step-9 selection rationale'ı destekler. Step 7 v3 id-seviyesindeki "
        "silver→gold leakage'ı kapattı (1934/1934 gold id silver havuzundan çıkarıldı). "
        "Burada id ayıklamasından sonra **içerik** (cleaned-text) seviyesinde kalıntı "
        "örtüşme olup olmadığını ölçüyoruz.\n"
    )
    L.append("## Kaynaklar\n")
    L.append(f"- Silver (leak-free): `{payload['silver_csv']}` — {payload['silver_rows']} satır")
    L.append("- Gold splits:")
    for split, info in payload["splits"].items():
        L.append(f"  - `{info['csv']}` — {info['gold_rows']} satır")
    L.append(
        f"- Normalizasyon: lowercase + URL/mention/`#` ayıklama + ASCII-dışı dengeleme + whitespace squash.\n"
        f"- Near-dup metriği: {payload['shingle_k']}-char shingle Jaccard ≥ {payload['jaccard_thresh']}"
    )
    L.append("")
    L.append("## Özet tablo\n")
    L.append("| split | gold satır | exact overlap | near-dup (J≥thr) | ratio_exact | ratio_near |")
    L.append("|---|---|---|---|---|---|")
    for split, info in payload["splits"].items():
        n = info["gold_rows"]
        e = info["exact_overlap_count"]
        nd = info["near_dup_hit_count"]
        L.append(
            f"| {split} | {n} | {e} | {nd} | {e/n:.4f} | {nd/n:.4f} |"
            if n
            else f"| {split} | {n} | {e} | {nd} | - | - |"
        )
    L.append("")
    for split, info in payload["splits"].items():
        if info["exact_overlap_count"] == 0 and info["near_dup_hit_count"] == 0:
            continue
        L.append(f"## {split} — örnekler\n")
        if info["exact_overlap_count"]:
            L.append(f"### Exact overlap (ilk {min(10, len(info['exact_overlap_samples']))})\n")
            for r in info["exact_overlap_samples"][:10]:
                L.append(f"- `id={r['gold_id']}` — {r['norm_preview']}")
            L.append("")
        if info["near_dup_hit_count"]:
            L.append(f"### Near-dup (ilk {min(10, len(info['near_dup_top']))}, J≥{payload['jaccard_thresh']})\n")
            for r in info["near_dup_top"][:10]:
                L.append(
                    f"- J={r['jaccard']:.3f} `gold_id={r['gold_id']}` ↔ `silver_id={r['silver_id']}`"
                )
                L.append(f"  - gold : {r['gold_norm']}")
                L.append(f"  - silver: {r['silver_norm']}")
            L.append("")
    L.append("## Yorum\n")
    L.append(payload["interpretation"])
    L.append("")
    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-csv", default=str(SILVER_CSV.as_posix()))
    ap.add_argument("--gold-train", default=str(GOLD_TRAIN.as_posix()))
    ap.add_argument("--gold-val", default=str(GOLD_VAL.as_posix()))
    ap.add_argument("--gold-test", default=str(GOLD_TEST.as_posix()))
    ap.add_argument("--jaccard-thresh", type=float, default=0.85)
    ap.add_argument("--shingle-k", type=int, default=5)
    ap.add_argument("--top-n-pairs", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading silver ...", flush=True)
    silver = _read_csv(Path(args.silver_csv))
    print(f"  silver rows: {len(silver)}", flush=True)

    print("Building silver shingle index ...", flush=True)
    ix, silver_shingles = _build_shingle_index(silver, k=int(args.shingle_k))
    silver_hashes = set(silver["_hash"])
    print(f"  unique shingles: {len(ix)}", flush=True)

    splits: Dict[str, Dict[str, Any]] = {}
    for name, path in (("train", args.gold_train), ("val", args.gold_val), ("test", args.gold_test)):
        print(f"Auditing {name} ({path}) ...", flush=True)
        gold = _read_csv(Path(path))
        result = _audit_split(
            gold_df=gold,
            silver_df=silver,
            silver_hashes=silver_hashes,
            shingle_index=ix,
            silver_shingle_sets=silver_shingles,
            k=int(args.shingle_k),
            jaccard_thresh=float(args.jaccard_thresh),
            top_n_pairs=int(args.top_n_pairs),
        )
        result["csv"] = str(Path(path).as_posix())
        splits[name] = result
        print(
            f"  {name}: exact={result['exact_overlap_count']} / "
            f"near_dup(J>={args.jaccard_thresh})={result['near_dup_hit_count']} "
            f"/ rows={result['gold_rows']}",
            flush=True,
        )

    total_exact = sum(s["exact_overlap_count"] for s in splits.values())
    total_nd = sum(s["near_dup_hit_count"] for s in splits.values())
    total_gold = sum(s["gold_rows"] for s in splits.values())

    if total_exact == 0 and total_nd == 0:
        interp = (
            "İçerik-seviyesinde ölçülebilir kalıntı örtüşme bulunmadı. "
            "Silver pool id-dışlaması (step 7 v3) sonrası normalize-edilmiş metin "
            "ve 5-char shingle Jaccard≥%.2f hiçbir gold satırıyla eşleşmedi. "
            "Selection'ı durduracak ölçüde bir sinyal yok; ancak bu audit "
            "nihai bir yakınlık çalışması değildir (truncated candidate listesi kullandı)."
        ) % float(args.jaccard_thresh)
    elif total_exact == 0 and total_nd > 0:
        interp = (
            f"Exact overlap yok (id-ayıklaması etkili), ama near-dup (J≥{args.jaccard_thresh}) "
            f"{total_nd}/{total_gold} gold satırında var. Bu örnekler retweet / alıntı / "
            "küçük yeniden-yazım olabilir. Selection'ı durduracak sinyal bulunmamakla birlikte, "
            "'known residual risk' olarak final selection dokümanına yazılmalı."
        )
    elif total_exact > 0:
        interp = (
            f"UYARI: id-ayıklamasına rağmen {total_exact}/{total_gold} gold satır "
            "silver'da normalize-eşdeğer metinle kaldı (muhtemelen aynı tweet'in farklı id'si). "
            "Bu sinyal selection'ı DURDURACAK düzeyde değilse (örn. yalnızca gold_train'i etkiliyorsa) "
            "'known residual risk' olarak doküman et ve step 10'da kapsamlı bir dedup yap."
        )
    else:
        interp = "(beklenmeyen durum)"

    payload: Dict[str, Any] = {
        "silver_csv": str(Path(args.silver_csv).as_posix()),
        "silver_rows": int(len(silver)),
        "shingle_k": int(args.shingle_k),
        "jaccard_thresh": float(args.jaccard_thresh),
        "splits": splits,
        "totals": {
            "gold_rows": int(total_gold),
            "exact_overlap": int(total_exact),
            "near_dup": int(total_nd),
        },
        "interpretation": interp,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")
    print(f"Totals: exact={total_exact}, near_dup={total_nd}, rows={total_gold}, elapsed={payload['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
