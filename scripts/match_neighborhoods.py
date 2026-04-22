"""
Match Tweets with Neighborhood Gazetteer
Combines keyword extraction and gazetteer matching for maximum location coverage
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from collections import Counter
from datetime import datetime

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
print("NEIGHBORHOOD MATCHING - FINAL GEOPROCESSING")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data...")

# Load tweets
tweets_df = pd.read_csv(INPUT_TWEETS)
print(f"  Loaded {len(tweets_df):,} tweets")
print(f"  Unique tweets: {tweets_df['id'].nunique():,}")

# Load gazetteer
gazetteer_df = pd.read_csv(INPUT_GAZETTEER)
print(f"  Loaded {len(gazetteer_df):,} neighborhoods from gazetteer")

# ============================================================================
# STEP 2: Prepare Gazetteer for Matching
# ============================================================================
print("\n[STEP 2] Preparing gazetteer for matching...")

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

# Create gazetteer lookup dictionary
# Sort by name length (longer first) to prioritize multi-word names
gazetteer_df['name_length'] = gazetteer_df['neighborhood_clean'].str.len()
gazetteer_df = gazetteer_df.sort_values('name_length', ascending=False)

# Create lookup structures
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

# Also add variations with "mahallesi" suffix removed
additional_lookups = {}
for name, info in list(neighborhood_lookup.items()):
    # Remove common suffixes if present
    for suffix in ['mahallesi', 'mah', 'köyü', 'koy']:
        if name.endswith(suffix):
            short_name = name[:-len(suffix)].strip()
            if len(short_name) > 2 and short_name not in neighborhood_lookup:
                additional_lookups[short_name] = info

neighborhood_lookup.update(additional_lookups)

print(f"  Created lookup with {len(neighborhood_lookup):,} unique neighborhood patterns")

# Sort by length for matching (longer patterns first)
sorted_patterns = sorted(neighborhood_lookup.keys(), key=len, reverse=True)
print(f"  Longest pattern: '{sorted_patterns[0]}' ({len(sorted_patterns[0])} chars)")
print(f"  Shortest pattern: '{sorted_patterns[-1]}' ({len(sorted_patterns[-1])} chars)")

# ============================================================================
# STEP 3: Keyword-based Extraction
# ============================================================================
print("\n[STEP 3] Keyword-based neighborhood extraction...")

def extract_keyword_neighborhood(text):
    """Extract neighborhood using mahalle/mah keywords"""
    if pd.isna(text):
        return None, None

    text_lower = str(text).lower()

    # Pattern: [Name] Mahallesi
    match = re.search(r'(\b[\wğüşöçıİĞÜŞÖÇ]+)\s+mahallesi\b', text_lower)
    if match:
        return match.group(1), 'mahallesi'

    # Pattern: [Name] Mah.
    match = re.search(r'(\b[\wğüşöçıİĞÜŞÖÇ]+)\s+mah\.\s', text_lower)
    if match:
        return match.group(1), 'mah.'

    # Pattern: [Name] Mah (space after)
    match = re.search(r'(\b[\wğüşöçıİĞÜŞÖÇ]+)\s+mah\s', text_lower)
    if match:
        return match.group(1), 'mah'

    return None, None

# Apply keyword extraction
print("  Extracting neighborhoods using keywords...")
keyword_results = tweets_df['tweet'].apply(extract_keyword_neighborhood)
tweets_df['keyword_neighborhood'] = keyword_results.apply(lambda x: x[0])
tweets_df['keyword_pattern'] = keyword_results.apply(lambda x: x[1])

keyword_count = tweets_df['keyword_neighborhood'].notna().sum()
print(f"  Tweets with keyword-based neighborhood: {keyword_count:,} ({keyword_count/len(tweets_df)*100:.1f}%)")

# ============================================================================
# STEP 4: Gazetteer-based Matching
# ============================================================================
print("\n[STEP 4] Gazetteer-based neighborhood matching...")

def match_gazetteer_neighborhood(text, sorted_patterns, neighborhood_lookup):
    """Match neighborhood from gazetteer using whole word matching"""
    if pd.isna(text):
        return None, None, None

    text_normalized = normalize_turkish(str(text))

    # Try matching each pattern (longest first)
    for pattern in sorted_patterns:
        # Build whole word regex pattern
        # Allow for Turkish word boundaries and common punctuation
        regex_pattern = r'(?:^|[\s,\.\-/\(\)])' + re.escape(pattern) + r"(?:[\s,\.\-/\(\)'']|$)"

        if re.search(regex_pattern, text_normalized):
            info = neighborhood_lookup[pattern]
            return pattern, info['district'], info['province']

    return None, None, None

# Apply gazetteer matching only to tweets without keyword match
print("  Matching against gazetteer (this may take a few minutes)...")

# Process in batches for progress reporting
batch_size = 10000
total_batches = len(tweets_df) // batch_size + 1

gazetteer_neighborhood = []
gazetteer_district = []
gazetteer_province = []

for i in range(0, len(tweets_df), batch_size):
    batch = tweets_df.iloc[i:i+batch_size]
    batch_num = i // batch_size + 1

    if batch_num % 5 == 0 or batch_num == 1:
        print(f"    Processing batch {batch_num}/{total_batches}...")

    for _, row in batch.iterrows():
        # Skip if already has keyword match
        if pd.notna(row['keyword_neighborhood']):
            gazetteer_neighborhood.append(None)
            gazetteer_district.append(None)
            gazetteer_province.append(None)
        else:
            match, district, province = match_gazetteer_neighborhood(
                row['tweet'], sorted_patterns, neighborhood_lookup
            )
            gazetteer_neighborhood.append(match)
            gazetteer_district.append(district)
            gazetteer_province.append(province)

tweets_df['gazetteer_neighborhood'] = gazetteer_neighborhood
tweets_df['gazetteer_district'] = gazetteer_district
tweets_df['gazetteer_province'] = gazetteer_province

gazetteer_count = tweets_df['gazetteer_neighborhood'].notna().sum()
print(f"  Tweets matched via gazetteer: {gazetteer_count:,} ({gazetteer_count/len(tweets_df)*100:.1f}%)")

# ============================================================================
# STEP 5: Combine Methods and Assign Final Neighborhoods
# ============================================================================
print("\n[STEP 5] Combining extraction methods...")

def assign_final_neighborhood(row):
    """Assign final neighborhood from keyword or gazetteer match"""
    if pd.notna(row['keyword_neighborhood']):
        neighborhood = row['keyword_neighborhood']
        method = 'keyword'
        # Look up district/province from gazetteer
        norm_name = normalize_turkish(neighborhood)
        if norm_name in neighborhood_lookup:
            info = neighborhood_lookup[norm_name]
            district = info['district']
            province = info['province']
        else:
            district = None
            province = None
    elif pd.notna(row['gazetteer_neighborhood']):
        neighborhood = row['gazetteer_neighborhood']
        method = 'gazetteer'
        district = row['gazetteer_district']
        province = row['gazetteer_province']
    else:
        neighborhood = None
        method = None
        district = None
        province = None

    return pd.Series({
        'neighborhood': neighborhood,
        'extraction_method': method,
        'district': district,
        'province': province
    })

# Apply final assignment
print("  Assigning final neighborhoods...")
final_results = tweets_df.apply(assign_final_neighborhood, axis=1)
tweets_df['neighborhood'] = final_results['neighborhood']
tweets_df['extraction_method'] = final_results['extraction_method']
tweets_df['district_final'] = final_results['district']
tweets_df['province_final'] = final_results['province']

# Create normalized neighborhood name
tweets_df['neighborhood_normalized'] = tweets_df['neighborhood'].apply(normalize_turkish)

# Boolean flag
tweets_df['has_neighborhood'] = tweets_df['neighborhood'].notna()

# ============================================================================
# STEP 6: Calculate Quality Metrics
# ============================================================================
print("\n[STEP 6] Calculating quality metrics...")

total_tweets = len(tweets_df)
keyword_matches = (tweets_df['extraction_method'] == 'keyword').sum()
gazetteer_matches = (tweets_df['extraction_method'] == 'gazetteer').sum()
no_match = tweets_df['neighborhood'].isna().sum()
total_with_neighborhood = tweets_df['has_neighborhood'].sum()

print("\n" + "=" * 80)
print("COVERAGE METRICS")
print("=" * 80)

print(f"\n[Before/After Comparison]")
print(f"  Before (keyword only): {keyword_matches:,} tweets ({keyword_matches/total_tweets*100:.1f}%)")
print(f"  After (keyword + gazetteer): {total_with_neighborhood:,} tweets ({total_with_neighborhood/total_tweets*100:.1f}%)")
print(f"  Coverage increase: +{gazetteer_matches:,} tweets (+{gazetteer_matches/total_tweets*100:.1f}%)")

print(f"\n[Method Breakdown]")
print(f"  Keyword-based: {keyword_matches:,} ({keyword_matches/total_tweets*100:.1f}%)")
print(f"  Gazetteer-based: {gazetteer_matches:,} ({gazetteer_matches/total_tweets*100:.1f}%)")
print(f"  No neighborhood: {no_match:,} ({no_match/total_tweets*100:.1f}%)")

# Top neighborhoods
print(f"\n[Top 30 Neighborhoods (Final)]")
neighborhood_counts = tweets_df['neighborhood'].value_counts().head(30)
for i, (name, count) in enumerate(neighborhood_counts.items(), 1):
    method_counts = tweets_df[tweets_df['neighborhood'] == name]['extraction_method'].value_counts()
    keyword_n = method_counts.get('keyword', 0)
    gazetteer_n = method_counts.get('gazetteer', 0)
    print(f"  {i:2}. {name}: {count:,} (K:{keyword_n:,} G:{gazetteer_n:,})")

# Province distribution
print(f"\n[Province Distribution]")
province_counts = tweets_df[tweets_df['province_final'].notna()]['province_final'].value_counts()
for province, count in province_counts.head(15).items():
    print(f"  {province}: {count:,}")

# District distribution
print(f"\n[Top 20 Districts]")
district_counts = tweets_df[tweets_df['district_final'].notna()].groupby(
    ['province_final', 'district_final']
).size().sort_values(ascending=False).head(20)
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
    print(f"    District: {row['district_final']}")
    print(f"    Province: {row['province_final']}")
    print(f"    Urgency: {row['urgency_score']}")

print("\n[5 Tweets Matched by KEYWORD Method]")
keyword_examples = tweets_df[tweets_df['extraction_method'] == 'keyword'].sample(n=min(5, keyword_matches), random_state=42)
for i, (_, row) in enumerate(keyword_examples.iterrows(), 1):
    show_example(row, i)

print("\n[5 Tweets Matched by GAZETTEER Method]")
gazetteer_examples = tweets_df[tweets_df['extraction_method'] == 'gazetteer'].sample(n=min(5, gazetteer_matches), random_state=42)
for i, (_, row) in enumerate(gazetteer_examples.iterrows(), 1):
    show_example(row, i)

print("\n[5 Tweets with NO Neighborhood Found]")
no_match_examples = tweets_df[tweets_df['neighborhood'].isna()].sample(n=min(5, no_match), random_state=42)
for i, (_, row) in enumerate(no_match_examples.iterrows(), 1):
    tweet_preview = str(row['tweet'])[:150].replace('\n', ' ')
    print(f"\n  Example {i}:")
    print(f"    Tweet: {tweet_preview}...")
    print(f"    Urgency: {row['urgency_score']}")

# ============================================================================
# STEP 8: Save Final Dataset
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Select columns for final output
output_columns = [
    'id', 'created_at', 'date', 'time', 'user_id', 'username',
    'tweet', 'tweet_clean', 'tweet_normalized',
    'urgency_score', 'has_emergency_keywords', 'has_location_keywords',
    'has_affected_area', 'total_engagement', 'mentions_official',
    'hashtags', 'mentions', 'photos', 'urls',
    # New neighborhood columns
    'neighborhood', 'neighborhood_normalized', 'district_final', 'province_final',
    'extraction_method', 'has_neighborhood'
]

# Ensure columns exist
available_columns = [c for c in output_columns if c in tweets_df.columns]
output_df = tweets_df[available_columns].copy()

# Rename for consistency
output_df = output_df.rename(columns={
    'district_final': 'district',
    'province_final': 'province'
})

# Sort by date and time for simulation
output_df = output_df.sort_values(['date', 'time'])

# Save final dataset
output_path = os.path.join(OUTPUT_DIR, 'emergency_with_neighborhoods_final.csv')
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved final dataset to: {output_path}")
print(f"Total rows: {len(output_df):,}")
print(f"Rows with neighborhood: {output_df['has_neighborhood'].sum():,}")

# ============================================================================
# STEP 9: Generate Report
# ============================================================================
report_path = os.path.join(REPORT_DIR, 'neighborhood_matching_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("NEIGHBORHOOD MATCHING REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("COVERAGE SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total tweets: {total_tweets:,}\n")
    f.write(f"Tweets with neighborhood (final): {total_with_neighborhood:,} ({total_with_neighborhood/total_tweets*100:.1f}%)\n")
    f.write(f"Tweets without neighborhood: {no_match:,} ({no_match/total_tweets*100:.1f}%)\n\n")

    f.write("METHOD BREAKDOWN\n")
    f.write("-" * 40 + "\n")
    f.write(f"Keyword extraction: {keyword_matches:,} ({keyword_matches/total_tweets*100:.1f}%)\n")
    f.write(f"Gazetteer matching: {gazetteer_matches:,} ({gazetteer_matches/total_tweets*100:.1f}%)\n\n")

    f.write("COVERAGE IMPROVEMENT\n")
    f.write("-" * 40 + "\n")
    f.write(f"Before (keyword only): {keyword_matches:,} ({keyword_matches/total_tweets*100:.1f}%)\n")
    f.write(f"After (keyword + gazetteer): {total_with_neighborhood:,} ({total_with_neighborhood/total_tweets*100:.1f}%)\n")
    f.write(f"Improvement: +{gazetteer_matches:,} tweets (+{gazetteer_matches/total_tweets*100:.1f}%)\n\n")

    f.write("TOP 50 NEIGHBORHOODS\n")
    f.write("-" * 40 + "\n")
    top_50 = tweets_df['neighborhood'].value_counts().head(50)
    for i, (name, count) in enumerate(top_50.items(), 1):
        f.write(f"{i:2}. {name}: {count:,}\n")

    f.write("\nPROVINCE DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    for province, count in province_counts.items():
        f.write(f"{province}: {count:,}\n")

    f.write("\nTOP 30 DISTRICTS\n")
    f.write("-" * 40 + "\n")
    district_counts_30 = tweets_df[tweets_df['district_final'].notna()].groupby(
        ['province_final', 'district_final']
    ).size().sort_values(ascending=False).head(30)
    for (province, district), count in district_counts_30.items():
        f.write(f"{district}, {province}: {count:,}\n")

print(f"Saved report to: {report_path}")

print("\n" + "=" * 80)
print("NEIGHBORHOOD MATCHING COMPLETE")
print("=" * 80)
print(f"\nFinal coverage: {total_with_neighborhood:,}/{total_tweets:,} tweets ({total_with_neighborhood/total_tweets*100:.1f}%)")
print("Ready for simulation!")
