"""
Phase 1: Data Quality Assessment for Neighborhood Extraction
Comprehensive analysis of neighborhood patterns in emergency tweets
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import os
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
INPUT_PATH = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\processed\emergency_with_location.csv"
OUTPUT_DIR = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\analysis"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 1: NEIGHBORHOOD DATA QUALITY ASSESSMENT")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv(INPUT_PATH)
print(f"Total tweets loaded: {len(df):,}")
print(f"Unique tweets (by ID): {df['id'].nunique():,}")

# Check urgency score distribution
print("\n[2] Urgency Score Distribution:")
urgency_dist = df['urgency_score'].value_counts().sort_index(ascending=False)
for score, count in urgency_dist.items():
    print(f"  Score {score}: {count:,} tweets ({count/len(df)*100:.1f}%)")

# ============================================================================
# TASK 1: Create Stratified Sample of 200 Tweets
# ============================================================================
print("\n" + "=" * 80)
print("TASK 1: CREATING STRATIFIED VALIDATION SAMPLE (200 tweets)")
print("=" * 80)

# Get unique tweets to avoid duplicates
df_unique = df.drop_duplicates(subset=['id'])
print(f"Unique tweets for sampling: {len(df_unique):,}")

# Define strata
strata = [
    (10, 11, 50),  # urgency 10-11, 50 samples
    (7, 9, 50),    # urgency 7-9, 50 samples
    (5, 6, 50),    # urgency 5-6, 50 samples
    (3, 4, 50),    # urgency 3-4, 50 samples
]

samples = []
for low, high, n in strata:
    stratum = df_unique[(df_unique['urgency_score'] >= low) & (df_unique['urgency_score'] <= high)]
    sample_n = min(n, len(stratum))
    if sample_n > 0:
        sample = stratum.sample(n=sample_n, random_state=42)
        samples.append(sample)
        print(f"  Urgency {low}-{high}: Sampled {sample_n} from {len(stratum):,} available")

validation_sample = pd.concat(samples, ignore_index=True)

# Select relevant columns for validation
validation_cols = ['id', 'urgency_score', 'tweet', 'address_components', 'date', 'time', 'hashtags']
validation_df = validation_sample[validation_cols].copy()
validation_df = validation_df.sort_values('urgency_score', ascending=False)

# Save validation sample
validation_path = os.path.join(OUTPUT_DIR, 'validation_sample_200.csv')
validation_df.to_csv(validation_path, index=False, encoding='utf-8-sig')
print(f"\nSaved validation sample to: {validation_path}")
print(f"Total samples: {len(validation_df)}")

# ============================================================================
# TASK 2.1: Address Components Analysis
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2.1: ADDRESS COMPONENTS ANALYSIS")
print("=" * 80)

# Count tweets with address_components
has_address = df['address_components'].notna() & (df['address_components'] != '')
print(f"\nTweets with address_components: {has_address.sum():,} ({has_address.sum()/len(df)*100:.1f}%)")
print(f"Tweets without address_components: {(~has_address).sum():,} ({(~has_address).sum()/len(df)*100:.1f}%)")

# Parse and analyze address components
print("\n[Top 50 Most Common Address Component Patterns]")
all_components = []
for addr in df['address_components'].dropna():
    if addr and str(addr) != 'nan':
        parts = str(addr).split(' | ')
        all_components.extend(parts)

component_counts = Counter(all_components)
print(f"Total unique component patterns: {len(component_counts):,}")

print("\nTop 50 patterns:")
for i, (pattern, count) in enumerate(component_counts.most_common(50), 1):
    print(f"  {i:2}. [{count:5,}] {pattern[:80]}")

# Identify neighborhood-like patterns vs false positives
print("\n[Pattern Classification]")
neighborhood_patterns = []
false_positive_patterns = []

for pattern, count in component_counts.most_common(200):
    pattern_lower = pattern.lower()
    # Check if it looks like a neighborhood
    if any(kw in pattern_lower for kw in ['mahalle', 'mah ', 'mah.']):
        neighborhood_patterns.append((pattern, count))
    elif any(kw in pattern_lower for kw in ['mahsur', 'enkaz', 'yardım', 'acil', 'kurtarma', 'var', 'kaldı']):
        false_positive_patterns.append((pattern, count))

print(f"\nNeighborhood-like patterns (containing 'mahalle/mah'): {len(neighborhood_patterns)}")
for pattern, count in neighborhood_patterns[:20]:
    print(f"  [{count:5,}] {pattern[:70]}")

print(f"\nFalse positive patterns (emergency keywords, not locations): {len(false_positive_patterns)}")
for pattern, count in false_positive_patterns[:20]:
    print(f"  [{count:5,}] {pattern[:70]}")

# ============================================================================
# TASK 2.2: Neighborhood Keyword Detection
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2.2: NEIGHBORHOOD KEYWORD DETECTION")
print("=" * 80)

# Search for neighborhood keywords in tweet text
mahalle_patterns = [
    r'(\w+)\s*mahallesi',
    r'(\w+)\s*mah\.',
    r'(\w+)\s*mah\s',
    r'mahallesi\s*(\w+)',
]

# Find all tweets with mahalle keywords
mahalle_regex = r'mahalle|mah\.|mah\s'
has_mahalle = df['tweet'].str.lower().str.contains(mahalle_regex, regex=True, na=False)
print(f"\nTweets with neighborhood keywords: {has_mahalle.sum():,} ({has_mahalle.sum()/len(df)*100:.1f}%)")

# Extract neighborhood names
def extract_neighborhood(text):
    if pd.isna(text):
        return []
    text_lower = str(text).lower()
    neighborhoods = []

    # Pattern: [Name] Mahallesi
    matches = re.findall(r'(\b\w+)\s*mahallesi', text_lower)
    neighborhoods.extend(matches)

    # Pattern: [Name] Mah.
    matches = re.findall(r'(\b\w+)\s*mah\.', text_lower)
    neighborhoods.extend(matches)

    # Pattern: [Name] Mah (space after)
    matches = re.findall(r'(\b\w+)\s*mah\s', text_lower)
    neighborhoods.extend(matches)

    return neighborhoods

print("\nExtracting neighborhood names from tweets...")
all_neighborhoods = []
for tweet in df['tweet']:
    neighborhoods = extract_neighborhood(tweet)
    all_neighborhoods.extend(neighborhoods)

neighborhood_counts = Counter(all_neighborhoods)
print(f"Total neighborhood mentions extracted: {len(all_neighborhoods):,}")
print(f"Unique neighborhood names: {len(neighborhood_counts):,}")

# Top 100 neighborhoods
print("\n[Top 100 Most Mentioned Neighborhoods]")
top_neighborhoods = neighborhood_counts.most_common(100)
for i, (name, count) in enumerate(top_neighborhoods, 1):
    print(f"  {i:3}. [{count:5,}] {name}")

# Save top neighborhoods
top_neigh_df = pd.DataFrame(top_neighborhoods, columns=['neighborhood_name', 'mention_count'])
top_neigh_path = os.path.join(OUTPUT_DIR, 'top_neighborhoods.csv')
top_neigh_df.to_csv(top_neigh_path, index=False, encoding='utf-8-sig')
print(f"\nSaved top neighborhoods to: {top_neigh_path}")

# Sample tweets for top 20 neighborhoods
print("\n[Sample Tweets for Top 20 Neighborhoods]")
for name, count in top_neighborhoods[:20]:
    print(f"\n--- {name.upper()} MAHALLESI ({count} mentions) ---")
    # Find sample tweets
    pattern = f'{name}\\s*(mahallesi|mah\\.|mah\\s)'
    matching = df[df['tweet'].str.lower().str.contains(pattern, regex=True, na=False)].head(2)
    for _, row in matching.iterrows():
        tweet_preview = str(row['tweet'])[:150].replace('\n', ' ')
        print(f"  [{row['urgency_score']}] {tweet_preview}...")

# ============================================================================
# TASK 2.3: Geographic Entity Recognition
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2.3: GEOGRAPHIC ENTITY RECOGNITION")
print("=" * 80)

# Known Turkish cities affected by earthquake
affected_cities = [
    'hatay', 'kahramanmaraş', 'kahramanmaras', 'gaziantep', 'adıyaman', 'adiyaman',
    'malatya', 'diyarbakır', 'diyarbakir', 'şanlıurfa', 'sanliurfa', 'osmaniye',
    'adana', 'kilis', 'elazığ', 'elazig'
]

# Known districts
known_districts = [
    'antakya', 'iskenderun', 'islahiye', 'nurdağı', 'nurdagi', 'pazarcık', 'pazarcik',
    'elbistan', 'gölbaşı', 'golbasi', 'defne', 'samandağ', 'samandag', 'reyhanlı', 'reyhanli',
    'arsuz', 'payas', 'dörtyol', 'dortyol', 'hassa', 'kırıkhan', 'kirikhan', 'belen',
    'yayladağı', 'yayladagi', 'altınözü', 'altinozu', 'kumlu', 'kırıkhan', 'besni',
    'çelikhan', 'celikhan', 'gerger', 'gölbaşı', 'kahta', 'samsat', 'sincik', 'tut'
]

print("\n[City Mentions in Tweets]")
city_counts = {}
for city in affected_cities:
    count = df['tweet'].str.lower().str.contains(city, na=False).sum()
    if count > 0:
        city_counts[city] = count

for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
    print(f"  {city.title()}: {count:,} tweets")

print("\n[District Mentions in Tweets]")
district_counts = {}
for district in known_districts:
    count = df['tweet'].str.lower().str.contains(district, na=False).sum()
    if count > 0:
        district_counts[district] = count

for district, count in sorted(district_counts.items(), key=lambda x: -x[1])[:30]:
    print(f"  {district.title()}: {count:,} tweets")

# ============================================================================
# TASK 2.4: Address Structure Analysis
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2.4: ADDRESS STRUCTURE ANALYSIS")
print("=" * 80)

# Define address patterns
patterns = {
    'Pattern 1: [Name] Mahallesi [Sokak]': r'\w+\s+mahallesi\s+\d*\.?\s*sokak',
    'Pattern 2: [Name] Mah. [Sokak]': r'\w+\s+mah\.\s+\d*\.?\s*sokak',
    'Pattern 3: [Name] Mahallesi [Cadde]': r'\w+\s+mahallesi\s+\w+\s+cadde',
    'Pattern 4: [City]/[District]': r'(hatay|gaziantep|adıyaman|kahramanmaraş|malatya|adana|osmaniye)\s*/\s*\w+',
    'Pattern 5: [District]/[City]': r'\w+\s*/\s*(hatay|gaziantep|adıyaman|kahramanmaraş|malatya|adana|osmaniye)',
    'Pattern 6: [Name] Apt/Apartman': r'\w+\s+(apartmanı|apt\.?|apt\s)',
    'Pattern 7: Sokak No': r'sokak\s+(no\.?|no\s*)\s*\d+',
    'Pattern 8: [Name] Sitesi': r'\w+\s+sitesi',
}

print("\n[Address Pattern Detection]")
for pattern_name, regex in patterns.items():
    matches = df['tweet'].str.lower().str.contains(regex, regex=True, na=False).sum()
    print(f"  {pattern_name}: {matches:,} tweets ({matches/len(df)*100:.1f}%)")

# ============================================================================
# TASK 3: Quality Metrics
# ============================================================================
print("\n" + "=" * 80)
print("TASK 3: QUALITY METRICS")
print("=" * 80)

# Calculate metrics
has_neighborhood = has_mahalle
has_city_only = df['tweet'].str.lower().str.contains('|'.join(affected_cities), regex=True, na=False) & ~has_neighborhood
has_district = df['tweet'].str.lower().str.contains('|'.join(known_districts), regex=True, na=False)
has_any_location = has_neighborhood | has_city_only | has_district

print("\n[Location Extraction Quality Metrics]")
print(f"  Tweets with identifiable neighborhood (mahalle): {has_neighborhood.sum():,} ({has_neighborhood.sum()/len(df)*100:.1f}%)")
print(f"  Tweets with city but no neighborhood: {has_city_only.sum():,} ({has_city_only.sum()/len(df)*100:.1f}%)")
print(f"  Tweets with district mention: {has_district.sum():,} ({has_district.sum()/len(df)*100:.1f}%)")
print(f"  Tweets with any location info: {has_any_location.sum():,} ({has_any_location.sum()/len(df)*100:.1f}%)")
print(f"  Tweets with NO clear location: {(~has_any_location).sum():,} ({(~has_any_location).sum()/len(df)*100:.1f}%)")

# Current extraction errors analysis
print("\n[Current Address Components Issues]")
# Check for common extraction errors
error_patterns = [
    ('Contains "mahsur" (false positive)', 'mahsur'),
    ('Contains "yardım" (false positive)', 'yardım'),
    ('Contains "acil" (false positive)', 'acil'),
    ('Contains "enkaz" (false positive)', 'enkaz'),
    ('Contains only numbers', r'^\d+$'),
]

print("  Common false positive patterns in address_components:")
for desc, pattern in error_patterns:
    if pattern.startswith('^'):
        count = df['address_components'].str.contains(pattern, regex=True, na=False).sum()
    else:
        count = df['address_components'].str.contains(pattern, case=False, na=False).sum()
    print(f"    {desc}: {count:,} occurrences")

# ============================================================================
# Generate Reports
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING REPORTS")
print("=" * 80)

# Report 1: Neighborhood Patterns Report
report_path = os.path.join(OUTPUT_DIR, 'neighborhood_patterns_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("NEIGHBORHOOD PATTERNS ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total tweets: {len(df):,}\n")
    f.write(f"Unique tweets: {df['id'].nunique():,}\n")
    f.write(f"Tweets with address_components: {has_address.sum():,} ({has_address.sum()/len(df)*100:.1f}%)\n\n")

    f.write("NEIGHBORHOOD KEYWORD ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Tweets with mahalle/mah keywords: {has_mahalle.sum():,} ({has_mahalle.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Unique neighborhood names extracted: {len(neighborhood_counts):,}\n\n")

    f.write("TOP 100 NEIGHBORHOODS\n")
    f.write("-" * 40 + "\n")
    for i, (name, count) in enumerate(top_neighborhoods, 1):
        f.write(f"{i:3}. {name}: {count:,} mentions\n")

    f.write("\nCITY MENTIONS\n")
    f.write("-" * 40 + "\n")
    for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
        f.write(f"  {city.title()}: {count:,}\n")

    f.write("\nDISTRICT MENTIONS\n")
    f.write("-" * 40 + "\n")
    for district, count in sorted(district_counts.items(), key=lambda x: -x[1])[:30]:
        f.write(f"  {district.title()}: {count:,}\n")

print(f"Saved neighborhood patterns report to: {report_path}")

# Report 2: Quality Metrics
metrics_path = os.path.join(OUTPUT_DIR, 'extraction_quality_metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXTRACTION QUALITY METRICS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("LOCATION EXTRACTION COVERAGE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Tweets with neighborhood mention: {has_neighborhood.sum():,} ({has_neighborhood.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Tweets with city only (no neighborhood): {has_city_only.sum():,} ({has_city_only.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Tweets with district mention: {has_district.sum():,} ({has_district.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Tweets with ANY location: {has_any_location.sum():,} ({has_any_location.sum()/len(df)*100:.1f}%)\n")
    f.write(f"Tweets with NO location: {(~has_any_location).sum():,} ({(~has_any_location).sum()/len(df)*100:.1f}%)\n\n")

    f.write("URGENCY SCORE DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    for score, count in urgency_dist.items():
        f.write(f"Score {score}: {count:,} ({count/len(df)*100:.1f}%)\n")

    f.write("\nADDRESS PATTERN COVERAGE\n")
    f.write("-" * 40 + "\n")
    for pattern_name, regex in patterns.items():
        matches = df['tweet'].str.lower().str.contains(regex, regex=True, na=False).sum()
        f.write(f"{pattern_name}: {matches:,} ({matches/len(df)*100:.1f}%)\n")

print(f"Saved quality metrics to: {metrics_path}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\n[Key Findings]")
print(f"1. {has_neighborhood.sum():,} tweets ({has_neighborhood.sum()/len(df)*100:.1f}%) contain neighborhood keywords")
print(f"2. {len(neighborhood_counts)} unique neighborhood names identified")
print(f"3. Top 5 neighborhoods: {', '.join([n for n,c in top_neighborhoods[:5]])}")
print(f"4. Current address_components have many false positives (emergency keywords)")

print("\n[Recommended Extraction Strategy]")
print("1. Use regex patterns to extract neighborhood names before 'mahallesi/mah.'")
print("2. Build a validated neighborhood dictionary from top extracted names")
print("3. Cross-reference with city/district to improve accuracy")
print("4. Remove false positives (mahsur, yardım, acil, etc.) from address_components")
print("5. Consider NER model for complex cases")

print("\n[Output Files Created]")
print(f"  1. {os.path.join(OUTPUT_DIR, 'validation_sample_200.csv')}")
print(f"  2. {os.path.join(OUTPUT_DIR, 'neighborhood_patterns_report.txt')}")
print(f"  3. {os.path.join(OUTPUT_DIR, 'top_neighborhoods.csv')}")
print(f"  4. {os.path.join(OUTPUT_DIR, 'extraction_quality_metrics.txt')}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
