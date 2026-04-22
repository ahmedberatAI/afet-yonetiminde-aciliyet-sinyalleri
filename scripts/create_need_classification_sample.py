#!/usr/bin/env python3
"""
Create a 1,000-tweet stratified sample for manual need classification labeling.

Notes on tiering and de-duplication:
- The source dataset contains many duplicate rows for the same tweet `id`, so we
  de-duplicate by `id` before sampling to avoid wasting labeling effort.
- After de-duplication + basic quality filters (tweet_clean length > 30 and a
  non-empty neighborhood), there are only 30 unique tweets with urgency 10-11
  and only 128 unique tweets with urgency 9-11. That makes the original
  "150 tweets from 10-11" tier infeasible without duplicates.
- To keep a "most-critical-heavy" distribution while still producing 1,000
  unique tweets, this script samples:
  - 150 tweets from urgency 8-11 (including ALL 9-11, topped up with 8)
  - 250 tweets from urgency 7
  - 200 tweets from urgency 5-6
  - 250 tweets from urgency 3-4
  - 150 tweets from urgency 0-2

Outputs (under data/labeling/):
- need_classification_sample_1000.csv
- labeling_guide.txt
- sampling_statistics.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NEED_LABEL_COLUMNS: List[str] = [
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

EXTRA_LABEL_COLUMNS: List[str] = [
    # Document schema alignment (project program): urgency 0-3 + veracity tri-class.
    "aciliyet_0_3",
    "veracity_label",  # dogrulanmis / supheli / asilsiz
]

NOTES_COLUMN = "notes"

LABEL_COLUMNS: List[str] = [*NEED_LABEL_COLUMNS, *EXTRA_LABEL_COLUMNS, NOTES_COLUMN]

OUTPUT_COLUMNS: List[str] = [
    "id",
    "created_at",
    "date",
    "time",
    "neighborhood",
    "district",
    "province",
    "urgency_score",
    "tweet",
    "tweet_clean",
    *LABEL_COLUMNS,
]


LabelTier = Tuple[str, int, int, int]  # (name, min_score, max_score, n)


def proportional_allocation(sizes: pd.Series, n: int, min_per_group: int = 0) -> Dict[str, int]:
    """Allocate n items across groups proportionally to group sizes."""
    if n < 0:
        raise ValueError("n must be >= 0")

    sizes = sizes.copy()
    sizes = sizes[sizes > 0]
    if sizes.empty:
        return {}

    total = int(sizes.sum())
    raw = (sizes / total) * n
    alloc = np.floor(raw).astype(int)

    if min_per_group > 0:
        alloc = np.maximum(alloc, min_per_group)

    # Cap at availability.
    alloc = np.minimum(alloc, sizes).astype(int)

    diff = int(n - int(alloc.sum()))
    if diff > 0:
        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        order = list(frac.index)
        i = 0
        while diff > 0 and order and i < 1_000_000:
            g = order[i % len(order)]
            if alloc[g] < sizes[g]:
                alloc[g] += 1
                diff -= 1
            i += 1
        if diff > 0:
            # Fallback: any group with remaining capacity.
            for g in sizes.sort_values(ascending=False).index:
                if diff <= 0:
                    break
                cap = int(sizes[g] - alloc[g])
                if cap <= 0:
                    continue
                take = min(diff, cap)
                alloc[g] += take
                diff -= take
        if diff != 0:
            raise RuntimeError("Allocation failed to reach target size.")

    elif diff < 0:
        # Too many allocated (due to min_per_group); remove from largest allocations.
        for g in alloc.sort_values(ascending=False).index:
            if diff == 0:
                break
            removable = int(alloc[g])
            if removable <= 0:
                continue
            take = min(removable, -diff)
            alloc[g] -= take
            diff += take
        if diff != 0:
            raise RuntimeError("Allocation failed to reduce to target size.")

    return {str(k): int(v) for k, v in alloc.to_dict().items()}


def stratified_sample_by_date(df: pd.DataFrame, n: int, seed: int, date_col: str = "date") -> pd.DataFrame:
    """Sample n rows, roughly proportional per-date, with at least 1 per date where possible."""
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).copy()

    sizes = df[date_col].value_counts(dropna=False)
    min_per = 1 if n >= sizes.size else 0
    alloc = proportional_allocation(sizes, n, min_per_group=min_per)

    parts: List[pd.DataFrame] = []
    date_as_str = df[date_col].astype("string")
    for date_value, k in alloc.items():
        if k <= 0:
            continue
        g = df[date_as_str == str(date_value)]
        if g.empty:
            continue
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))

    out = pd.concat(parts, ignore_index=False)
    if len(out) < n:
        remaining = df.drop(index=out.index, errors="ignore")
        need = n - len(out)
        if need > 0 and not remaining.empty:
            out = pd.concat(
                [out, remaining.sample(n=min(need, len(remaining)), random_state=seed)],
                ignore_index=False,
            )

    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.copy()


def write_labeling_guide(path: Path) -> None:
    # NOTE: This guide is a template to bootstrap annotation.
    # If you later calibrate/improve `data/labeling/labeling_guide.txt`,
    # prefer NOT overwriting it unless you explicitly pass --overwrite-guide.
    content = """\
NEED CLASSIFICATION LABELING GUIDE

File: need_classification_sample_1000.csv
Task: Multi-label classification of needs expressed in earthquake-related tweets.

IMPORTANT
- A tweet can have MULTIPLE categories (multi-label).
- Mark ALL applicable categories with 1.
- Leave as 0/blank if not applicable.
- Also fill the auxiliary fields:
  - aciliyet_0_3: 0,1,2,3 (overall urgency)
  - veracity_label: dogrulanmis / supheli / asilsiz (credibility)
- If unsure, write a short explanation in the 'notes' column.
- Focus on the PRIMARY need expressed in the tweet.

CROSS-CUTTING FIELDS

aciliyet_0_3 (0-3)
- 3: Immediate life-threatening (e.g., trapped under rubble, active rescue/medical emergency).
- 2: Urgent need affecting safety/health soon (e.g., no shelter in cold, no water/food for days, medical supplies).
- 1: Lower-priority / coordination / non-immediate logistics (e.g., requesting equipment dispatch, transport coordination).
- 0: Not a concrete need / general info sharing / unclear.

veracity_label (dogrulanmis / supheli / asilsiz)
- dogrulanmis: Confirmed/verified information (only use if you have a clear verification signal).
- supheli: Cannot verify / missing details / looks questionable (default if uncertain).
- asilsiz: Clearly false/misleading/spam.

COMMON RULES / EDGE CASES

- If the tweet indicates people trapped / under rubble / hearing voices / cannot reach the family / "enkaz" / "gocuk" / "mahsur" / "ses geliyor":
  - set `arama_kurtarma=1`
  - set `aciliyet_0_3=3` (default for rescue calls)

- If the tweet requests equipment/personnel/vehicles (e.g., vinc, is makinesi, beton kesici/kirici, termal kamera, ekip):
  - set `lojistik=1`
  - if it is clearly for an ongoing rescue case (mentions enkaz/gocuk/mahsur/ses), also set `arama_kurtarma=1`

- Tweets that say "lutfen yayalim / RT / duyurun" but are about a concrete need (especially rescue) are NOT `bilgi_paylasimi`.
  - `bilgi_paylasimi=1` is for general announcements/phone numbers/info that do NOT express a concrete need.

- Avoid literal substring logic when labeling:
  - Example: a place name containing "su" does NOT automatically mean `gida_su=1`.
  - Label based on meaning (need expressed), not only on keywords.

CATEGORIES

1) ARAMA_KURTARMA (Search & Rescue)
   Description: People trapped in rubble, need immediate rescue.
   Keywords: enkaz, gocuk, mahsur, kurtarma, sikismis, ses geliyor, canli var, enkaz altinda
   Example: \"Akevler mahallesi 5 kişi enkaz altında yardım edin\"

2) SAGLIK (Medical)
   Description: Injured people, need medical attention, medicine.
   Keywords: yaralı, ambulans, doktor, hastane, ilaç, kan
   Example: \"Odabaşı'nda 3 yaralı var ambulans lazım acil\"

3) BARINMA (Shelter)
   Description: Need shelter, blankets, heating.
   Keywords: çadır, battaniye, barınma, ısınma, soğuk, donuyoruz
   Example: \"Cebrail mahallesi battaniye lazım çok soğuk\"

4) GIDA_SU (Food & Water)
   Description: Hunger, thirst, need food/water.
   Keywords: aç, susuz, yemek, su, gıda, içecek
   Example: \"Hayrullah'ta 2 gündür su yok\"

5) ALTYAPI (Infrastructure)
   Description: Power outage, water cut, road blocked.
   Keywords: elektrik, jeneratör, su kesintisi, yol kapalı
   Example: \"Kanatlı mahallesi 48 saattir elektrik yok\"

6) GUVENLIK (Security)
   Description: Looting, security concerns.
   Keywords: yağma, güvenlik, asayiş, hırsızlık
   Example: \"Ekinci'de yağma var güvenlik lazım\"

7) LOJISTIK (Logistics)
   Description: Need vehicles, equipment, personnel.
   Keywords: araç, ekipman, personel, iş makinesi, vinç
   Example: \"Kemal mahallesi iş makinesi lazım yıkık bina için\"

8) PSIKOLOJIK (Psychological)
   Description: Trauma, panic, psychological support needed.
   Keywords: travma, panik, psikolojik destek, korku
   Example: \"Çocuklar çok korkmuş psikolojik destek gerek\"

9) BILGI_PAYLASIMI (Information Sharing)
   Description: General info, phone numbers, announcements (not urgent needs).
   Keywords: bilgi, duyuru, numara, paylaşım
   Example: \"AFAD numarası 122 herkes arasın\"
"""
    # Use BOM for best compatibility with Windows editors.
    path.write_text(content, encoding="utf-8-sig")


def tier_name_for_score(score: int) -> str:
    if 8 <= score <= 11:
        return "8-11_ultra_critical"
    if score == 7:
        return "7_high_priority"
    if 5 <= score <= 6:
        return "5-6_medium_priority"
    if 3 <= score <= 4:
        return "3-4_standard_priority"
    return "0-2_low_priority"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a 1,000-tweet need labeling sample.")
    parser.add_argument("--input", default="data/processed/emergency_geolocated_96k.csv", help="Input CSV path.")
    parser.add_argument("--outdir", default="data/labeling", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overwrite-guide",
        action="store_true",
        help="Overwrite data/labeling/labeling_guide.txt if it already exists.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tiers: List[LabelTier] = [
        ("8-11_ultra_critical", 8, 11, 150),
        ("7_high_priority", 7, 7, 250),
        ("5-6_medium_priority", 5, 6, 200),
        ("3-4_standard_priority", 3, 4, 250),
        ("0-2_low_priority", 0, 2, 150),
    ]

    usecols = [
        "id",
        "created_at",
        "date",
        "time",
        "tweet",
        "tweet_clean",
        "urgency_score",
        "neighborhood",
        "district",
        "province",
    ]
    df = pd.read_csv(input_path, usecols=usecols)

    df["id"] = df["id"].astype("string")
    for col in ["created_at", "date", "time", "tweet", "tweet_clean", "neighborhood", "district", "province"]:
        df[col] = df[col].astype("string")
    # Strip whitespace in key text fields to reduce accidental category inflation.
    for col in ["neighborhood", "district", "province"]:
        df[col] = df[col].str.strip()
    df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0).astype(int)

    # Filters: clear-enough content + usable geolocation.
    df["tweet_clean_len"] = df["tweet_clean"].fillna("").str.len()
    base = df[(df["tweet_clean_len"] > 30) & (df["neighborhood"].fillna("").str.strip() != "")].copy()
    base_rows_before_dedup = int(len(base))
    base = base.drop_duplicates(subset=["id"], keep="first").copy()
    base_rows_after_dedup = int(len(base))
    # Drop obvious extraction noise (single-letter "neighborhoods" like "l", "k", ...).
    base = base[base["neighborhood"].fillna("").str.len() >= 2].copy()
    base_rows_after_neighborhood_len = int(len(base))

    samples: List[pd.DataFrame] = []
    tier_seed = int(args.seed)

    for tier_name, lo, hi, n in tiers:
        cand = base[(base["urgency_score"] >= lo) & (base["urgency_score"] <= hi)].copy()
        if cand.empty:
            raise RuntimeError(f"No candidates for tier {tier_name} ({lo}-{hi}).")

        if tier_name == "8-11_ultra_critical":
            # Maximize critical content: include ALL 9-11, then top up from 8.
            crit_9_11 = cand[cand["urgency_score"] >= 9].copy()
            if len(crit_9_11) >= n:
                picked = stratified_sample_by_date(crit_9_11, n=n, seed=tier_seed)
                samples.append(picked)
                continue

            remain = n - len(crit_9_11)
            cand_8 = cand[cand["urgency_score"] == 8].copy()
            if len(cand_8) < remain:
                raise RuntimeError(
                    f"Not enough urgency=8 candidates to top up ultra critical tier: need {remain}, have {len(cand_8)}."
                )
            picked_8 = stratified_sample_by_date(cand_8, n=remain, seed=tier_seed + 1) if remain > 0 else cand_8.iloc[0:0]
            picked = pd.concat([crit_9_11, picked_8], ignore_index=False).sample(frac=1.0, random_state=tier_seed)
            samples.append(picked)
            continue

        if len(cand) < n:
            raise RuntimeError(f"Not enough candidates for tier {tier_name}: need {n}, have {len(cand)}.")
        samples.append(stratified_sample_by_date(cand, n=n, seed=tier_seed))
        tier_seed += 10

    sample = pd.concat(samples, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if int(sample["id"].duplicated().sum()) != 0:
        raise RuntimeError("Sample contains duplicate tweet ids; de-duplication failed.")

    unique_neighborhoods = int(sample["neighborhood"].nunique(dropna=True))
    unique_dates = int(sample["date"].nunique(dropna=True))
    if unique_neighborhoods < 10:
        raise RuntimeError(f"Neighborhood diversity too low: {unique_neighborhoods} unique neighborhoods.")
    if unique_dates < 8:
        raise RuntimeError(f"Temporal diversity too low: {unique_dates} unique dates (need >= 8).")

    for col in LABEL_COLUMNS:
        sample[col] = ""

    out_csv = outdir / "need_classification_sample_1000.csv"
    guide_path = outdir / "labeling_guide.txt"
    stats_path = outdir / "sampling_statistics.txt"

    sample[OUTPUT_COLUMNS].to_csv(out_csv, index=False, encoding="utf-8-sig")
    if guide_path.exists() and not args.overwrite_guide:
        print(f"Guide exists, not overwriting (pass --overwrite-guide to overwrite): {guide_path}")
    else:
        write_labeling_guide(guide_path)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sample_tier = sample["urgency_score"].apply(tier_name_for_score)

    tier_counts = sample_tier.value_counts().to_dict()
    urgency_counts = sample["urgency_score"].value_counts().sort_index().to_dict()
    neighborhood_top = sample["neighborhood"].value_counts().head(15).to_dict()
    province_counts = sample["province"].value_counts(dropna=False).to_dict()
    date_counts = sample["date"].value_counts().sort_index().to_dict()

    lengths = sample["tweet_clean"].fillna("").str.len()
    length_min = int(lengths.min())
    length_max = int(lengths.max())
    length_mean = float(lengths.mean())

    # Validation examples: 2 from each tier.
    tmp = sample.copy()
    tmp["tier"] = sample_tier
    rs = np.random.RandomState(args.seed)
    validation_lines: List[str] = []
    for tier_label in ["8-11_ultra_critical", "7_high_priority", "5-6_medium_priority", "3-4_standard_priority", "0-2_low_priority"]:
        subset = tmp[tmp["tier"] == tier_label]
        if subset.empty:
            continue
        take_n = 2 if len(subset) >= 2 else 1
        picked = subset.sample(n=take_n, random_state=int(rs.randint(0, 1_000_000)))
        for _, row in picked.iterrows():
            tweet = str(row["tweet"]) if pd.notna(row["tweet"]) else ""
            tweet = tweet.replace("\r", " ").replace("\n", " ").strip()
            if len(tweet) > 240:
                tweet = tweet[:237] + "..."
            validation_lines.append(
                f"- [{tier_label}] urgency={int(row['urgency_score'])} date={row['date']} time={row['time']} "
                f"neighborhood={row['neighborhood']} province={row['province']} id={row['id']} | {tweet}"
            )

    lines: List[str] = []
    lines.append("SAMPLING STATISTICS\n")
    lines.append(f"Generated: {now}")
    lines.append(f"Input: {input_path.as_posix()}")
    lines.append(f"Output: {out_csv.as_posix()}")
    lines.append(f"Seed: {args.seed}")
    lines.append("")
    lines.append("FILTERS")
    lines.append("- tweet_clean length > 30")
    lines.append("- neighborhood not empty")
    lines.append("- neighborhood length >= 2 (removes single-letter extraction noise)")
    lines.append(f"- de-duplicated by tweet id: {base_rows_before_dedup} rows -> {base_rows_after_dedup} unique ids")
    lines.append(f"- after neighborhood length filter: {base_rows_after_neighborhood_len} unique ids")
    lines.append("")
    lines.append("TIERS (ADJUSTED)")
    lines.append("- 8-11 (ultra critical): 150  (includes ALL urgency 9-11, topped up with urgency 8)")
    lines.append("- 7    (high priority): 250")
    lines.append("- 5-6  (medium priority): 200")
    lines.append("- 3-4  (standard priority): 250")
    lines.append("- 0-2  (low priority): 150")
    lines.append("")
    lines.append("DISTRIBUTION BY TIER")
    for k in ["8-11_ultra_critical", "7_high_priority", "5-6_medium_priority", "3-4_standard_priority", "0-2_low_priority"]:
        lines.append(f"- {k}: {tier_counts.get(k, 0)}")
    lines.append("")
    lines.append("DISTRIBUTION BY URGENCY SCORE")
    for k, v in urgency_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("TOP 15 NEIGHBORHOODS")
    for k, v in neighborhood_top.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("DISTRIBUTION BY PROVINCE")
    for k, v in sorted(province_counts.items(), key=lambda x: (-x[1], str(x[0]))):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("DISTRIBUTION BY DATE")
    for k, v in date_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("TWEET LENGTH (tweet_clean) STATISTICS")
    lines.append(f"- min: {length_min}")
    lines.append(f"- max: {length_max}")
    lines.append(f"- mean: {length_mean:.2f}")
    lines.append("")
    lines.append("VALIDATION SAMPLES (2 PER TIER)")
    lines.extend(validation_lines)
    # Use BOM for best compatibility with Windows editors.
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    # Console output: summary + validation lines.
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {guide_path}")
    print(f"Wrote: {stats_path}")
    print("")
    print(f"Sample size: {len(sample)}")
    print(f"Unique neighborhoods: {unique_neighborhoods}")
    print(f"Unique dates: {unique_dates} (range {sample['date'].min()} .. {sample['date'].max()})")
    print(f"Tweet_clean length: min/max/mean = {length_min} / {length_max} / {length_mean:.2f}")
    print("")
    print("Validation samples:")
    for line in validation_lines:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
