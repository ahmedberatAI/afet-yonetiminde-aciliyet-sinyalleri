"""
Match Tweets with Neighborhood Gazetteer (Optimized Version)
Combines keyword extraction and gazetteer matching for maximum location coverage
Uses vectorized operations and efficient matching for large datasets
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
INPUT_TWEETS = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\processed\emergency_with_location.csv"
INPUT_GAZETTEER = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\gazetteer\earthquake_region_neighborhoods.csv"
OUTPUT_DIR = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\processed"
REPORT_DIR = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 80)
print("NEIGHBORHOOD MATCHING - FINAL GEOPROCESSING (OPTIMIZED)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data...")

# Load tweets
tweets_df = pd.read_csv(INPUT_TWEETS, low_memory=False)
print(f"  Loaded {len(tweets_df):,} tweets")

# Load gazetteer
gazetteer_df = pd.read_csv(INPUT_GAZETTEER)
print(f"  Loaded {len(gazetteer_df):,} neighborhoods from gazetteer")

# ============================================================================
# STEP 2: Prepare for Matching
# ============================================================================
print("\n[STEP 2] Preparing for matching...")

def normalize_turkish(text):
    """Normalize Turkish characters for matching"""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Normalize tweet text once (vectorized)
print("  Normalizing tweet text...")
tweets_df['tweet_norm'] = tweets_df['tweet'].fillna('').apply(normalize_turkish)

# Create efficient gazetteer lookup
print("  Building gazetteer lookup...")
neighborhood_lookup = {}
for _, row in gazetteer_df.iterrows():
    name_clean = str(row['neighborhood_clean']).strip()
    name_normalized = normalize_turkish(name_clean)

    if len(name_normalized) <= 2:  # Skip very short names
        continue

    if name_normalized not in neighborhood_lookup:
        neighborhood_lookup[name_normalized] = {
            'name': row['neighborhood_name'],
            'name_clean': name_clean,
            'district': row['district'],
            'province': row['province']
        }

# Sort patterns by length (longer first)
sorted_patterns = sorted(neighborhood_lookup.keys(), key=len, reverse=True)

# Filter to patterns >= 4 characters for better precision
sorted_patterns = [p for p in sorted_patterns if len(p) >= 4]
print(f"  {len(sorted_patterns):,} patterns ready for matching")

# ============================================================================
# STEP 3: Keyword-based Extraction (Vectorized)
# ============================================================================
print("\n[STEP 3] Keyword-based neighborhood extraction...")

# Combined regex for all keyword patterns
keyword_pattern = r'(\b[\wğüşöçıİĞÜŞÖÇ]+)\s+(?:mahallesi|mah\.|mah\s)'

def extract_keyword(text):
    if pd.isna(text):
        return None
    match = re.search(keyword_pattern, str(text).lower())
    return match.group(1) if match else None

tweets_df['keyword_neighborhood'] = tweets_df['tweet'].apply(extract_keyword)
keyword_count = tweets_df['keyword_neighborhood'].notna().sum()
print(f"  Keyword matches: {keyword_count:,} ({keyword_count/len(tweets_df)*100:.1f}%)")

# ============================================================================
# STEP 4: Gazetteer Matching (Optimized)
# ============================================================================
print("\n[STEP 4] Gazetteer-based matching...")

# For gazetteer matching, we use a more efficient approach:
# Pre-compile regex patterns for top neighborhoods only
TOP_N_PATTERNS = 500  # Only use most common/important patterns

# Get top patterns (longer ones and those known from tweets)
priority_patterns = sorted_patterns[:TOP_N_PATTERNS]

# Pre-compile regex patterns
compiled_patterns = []
for pattern in priority_patterns:
    try:
        # Whole word matching with word boundaries
        regex = re.compile(r'(?:^|[\s,\.\-/\(\)\'\"])' + re.escape(pattern) + r"(?:[\s,\.\-/\(\)\'\"']|$)")
        compiled_patterns.append((pattern, regex))
    except:
        pass

print(f"  Using {len(compiled_patterns)} compiled patterns")

# Match function
def match_gazetteer(text_norm, has_keyword):
    """Match against gazetteer patterns"""
    if has_keyword or not text_norm:
        return None

    for pattern, regex in compiled_patterns:
        if regex.search(text_norm):
            return pattern
    return None

# Apply matching in batches
print("  Matching tweets against gazetteer...")
batch_size = 50000
results = []

for i in range(0, len(tweets_df), batch_size):
    batch_end = min(i + batch_size, len(tweets_df))
    print(f"    Processing {i:,} - {batch_end:,}...")

    batch_results = []
    for idx in range(i, batch_end):
        text_norm = tweets_df.iloc[idx]['tweet_norm']
        has_keyword = pd.notna(tweets_df.iloc[idx]['keyword_neighborhood'])
        match = match_gazetteer(text_norm, has_keyword)
        batch_results.append(match)

    results.extend(batch_results)

tweets_df['gazetteer_neighborhood'] = results

gazetteer_count = tweets_df['gazetteer_neighborhood'].notna().sum()
print(f"  Gazetteer matches: {gazetteer_count:,} ({gazetteer_count/len(tweets_df)*100:.1f}%)")

# ============================================================================
# STEP 5: Combine Methods
# ============================================================================
print("\n[STEP 5] Combining extraction methods...")

# Assign final neighborhood
def get_final_neighborhood(row):
    if pd.notna(row['keyword_neighborhood']):
        return row['keyword_neighborhood']
    elif pd.notna(row['gazetteer_neighborhood']):
        return row['gazetteer_neighborhood']
    return None

def get_extraction_method(row):
    if pd.notna(row['keyword_neighborhood']):
        return 'keyword'
    elif pd.notna(row['gazetteer_neighborhood']):
        return 'gazetteer'
    return None

tweets_df['neighborhood'] = tweets_df.apply(get_final_neighborhood, axis=1)
tweets_df['extraction_method'] = tweets_df.apply(get_extraction_method, axis=1)
tweets_df['neighborhood_normalized'] = tweets_df['neighborhood'].apply(
    lambda x: normalize_turkish(x) if pd.notna(x) else None
)
tweets_df['has_neighborhood'] = tweets_df['neighborhood'].notna()

# Lookup district/province from gazetteer
def get_district_province(neigh_norm):
    if pd.isna(neigh_norm):
        return None, None
    if neigh_norm in neighborhood_lookup:
        info = neighborhood_lookup[neigh_norm]
        return info['district'], info['province']
    return None, None

print("  Looking up district/province information...")
district_province = tweets_df['neighborhood_normalized'].apply(get_district_province)
tweets_df['district'] = district_province.apply(lambda x: x[0])
tweets_df['province'] = district_province.apply(lambda x: x[1])

# ============================================================================
# STEP 6: Quality Metrics
# ============================================================================
print("\n" + "=" * 80)
print("COVERAGE METRICS")
print("=" * 80)

total_tweets = len(tweets_df)
keyword_matches = (tweets_df['extraction_method'] == 'keyword').sum()
gazetteer_matches = (tweets_df['extraction_method'] == 'gazetteer').sum()
total_with_neighborhood = tweets_df['has_neighborhood'].sum()
no_match = total_tweets - total_with_neighborhood

print(f"\n[Coverage Summary]")
print(f"  Total tweets: {total_tweets:,}")
print(f"  With neighborhood: {total_with_neighborhood:,} ({total_with_neighborhood/total_tweets*100:.1f}%)")
print(f"  Without neighborhood: {no_match:,} ({no_match/total_tweets*100:.1f}%)")

print(f"\n[Method Breakdown]")
print(f"  Keyword-based: {keyword_matches:,} ({keyword_matches/total_tweets*100:.1f}%)")
print(f"  Gazetteer-based: {gazetteer_matches:,} ({gazetteer_matches/total_tweets*100:.1f}%)")

print(f"\n[Coverage Improvement]")
print(f"  Before (keyword only): {keyword_matches:,} tweets")
print(f"  After (keyword + gazetteer): {total_with_neighborhood:,} tweets")
print(f"  Increase: +{gazetteer_matches:,} tweets (+{gazetteer_matches/keyword_matches*100:.1f}% improvement)")

# Top neighborhoods
print(f"\n[Top 30 Neighborhoods]")
neighborhood_counts = tweets_df['neighborhood'].value_counts().head(30)
for i, (name, count) in enumerate(neighborhood_counts.items(), 1):
    method_df = tweets_df[tweets_df['neighborhood'] == name]['extraction_method']
    k_n = (method_df == 'keyword').sum()
    g_n = (method_df == 'gazetteer').sum()
    print(f"  {i:2}. {name}: {count:,} (K:{k_n:,} G:{g_n:,})")

# Province distribution
print(f"\n[Province Distribution]")
province_counts = tweets_df[tweets_df['province'].notna()]['province'].value_counts()
for province, count in province_counts.head(12).items():
    pct = count / total_with_neighborhood * 100
    print(f"  {province}: {count:,} ({pct:.1f}%)")

# Top districts
print(f"\n[Top 15 Districts]")
district_counts = tweets_df[tweets_df['district'].notna()].groupby(
    ['province', 'district']
).size().sort_values(ascending=False).head(15)
for (province, district), count in district_counts.items():
    print(f"  {district}, {province}: {count:,}")

# ============================================================================
# STEP 7: Validation Examples
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION EXAMPLES")
print("=" * 80)

def show_example(row, num):
    tweet_preview = str(row['tweet'])[:120].replace('\n', ' ')
    print(f"\n  Example {num}:")
    print(f"    Tweet: {tweet_preview}...")
    print(f"    Neighborhood: {row['neighborhood']}")
    print(f"    District: {row['district']}, Province: {row['province']}")
    print(f"    Urgency: {row['urgency_score']}")

print("\n[5 Tweets - KEYWORD Method]")
kw_sample = tweets_df[tweets_df['extraction_method'] == 'keyword'].sample(n=min(5, keyword_matches), random_state=42)
for i, (_, row) in enumerate(kw_sample.iterrows(), 1):
    show_example(row, i)

print("\n[5 Tweets - GAZETTEER Method]")
gz_sample = tweets_df[tweets_df['extraction_method'] == 'gazetteer'].sample(n=min(5, gazetteer_matches), random_state=42)
for i, (_, row) in enumerate(gz_sample.iterrows(), 1):
    show_example(row, i)

print("\n[5 Tweets - NO NEIGHBORHOOD]")
no_sample = tweets_df[tweets_df['neighborhood'].isna()].sample(n=min(5, no_match), random_state=42)
for i, (_, row) in enumerate(no_sample.iterrows(), 1):
    tweet_preview = str(row['tweet'])[:150].replace('\n', ' ')
    print(f"\n  Example {i}:")
    print(f"    Tweet: {tweet_preview}...")
    print(f"    Urgency: {row['urgency_score']}")

# ============================================================================
# STEP 8: Save Output
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Select columns
output_columns = [
    'id', 'created_at', 'date', 'time', 'user_id', 'username',
    'tweet', 'tweet_clean', 'tweet_normalized',
    'urgency_score', 'has_emergency_keywords', 'has_location_keywords',
    'has_affected_area', 'total_engagement', 'mentions_official',
    'hashtags', 'mentions', 'photos', 'urls',
    'neighborhood', 'neighborhood_normalized', 'district', 'province',
    'extraction_method', 'has_neighborhood'
]

available_cols = [c for c in output_columns if c in tweets_df.columns]
output_df = tweets_df[available_cols].copy()

# Sort chronologically
output_df = output_df.sort_values(['date', 'time'])

# Save
output_path = os.path.join(OUTPUT_DIR, 'emergency_with_neighborhoods_final.csv')
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {output_path}")
print(f"  Rows: {len(output_df):,}")
print(f"  With neighborhood: {output_df['has_neighborhood'].sum():,}")

# Save report
report_path = os.path.join(REPORT_DIR, 'neighborhood_matching_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("NEIGHBORHOOD MATCHING REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total tweets: {total_tweets:,}\n")
    f.write(f"With neighborhood: {total_with_neighborhood:,} ({total_with_neighborhood/total_tweets*100:.1f}%)\n")
    f.write(f"Without neighborhood: {no_match:,} ({no_match/total_tweets*100:.1f}%)\n\n")

    f.write("METHOD BREAKDOWN\n")
    f.write(f"Keyword: {keyword_matches:,} ({keyword_matches/total_tweets*100:.1f}%)\n")
    f.write(f"Gazetteer: {gazetteer_matches:,} ({gazetteer_matches/total_tweets*100:.1f}%)\n\n")

    f.write("TOP 50 NEIGHBORHOODS\n")
    for i, (name, count) in enumerate(tweets_df['neighborhood'].value_counts().head(50).items(), 1):
        f.write(f"{i:2}. {name}: {count:,}\n")

    f.write("\nPROVINCE DISTRIBUTION\n")
    for prov, cnt in province_counts.items():
        f.write(f"{prov}: {cnt:,}\n")

print(f"Saved: {report_path}")

print("\n" + "=" * 80)
print("NEIGHBORHOOD MATCHING COMPLETE")
print("=" * 80)
print(f"\nFinal: {total_with_neighborhood:,}/{total_tweets:,} tweets have neighborhood ({total_with_neighborhood/total_tweets*100:.1f}%)")
print("Dataset ready for simulation!")
