"""
Extract Emergency Tweets with Help Request + Location
======================================================
Filters preprocessed data to find tweets containing BOTH:
1. Help/emergency request keywords
2. Location information
"""

import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Configuration
INPUT_FILE = 'data/processed/deprem_cleaned_no_dup_removal.csv'
OUTPUT_FILE = 'data/processed/emergency_with_location.csv'
LOG_FILE = 'data/processed/extraction_log.txt'

def log_and_print(message, log_entries):
    """Print and store log message."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'replace').decode('ascii'))
    log_entries.append(message)

def main():
    log_entries = []

    log_and_print("=" * 60, log_entries)
    log_and_print("EMERGENCY + LOCATION TWEET EXTRACTION", log_entries)
    log_and_print("=" * 60, log_entries)
    log_and_print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", log_entries)

    # Load data
    log_and_print(f"Loading data from {INPUT_FILE}...", log_entries)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    log_and_print(f"Loaded {len(df):,} tweets\n", log_entries)

    # Filtering Logic
    log_and_print("APPLYING FILTERS:", log_entries)
    log_and_print("-" * 40, log_entries)

    # Filter 1: has_emergency_keywords AND has_location_keywords
    filter1 = (df['has_emergency_keywords'] == True) & (df['has_location_keywords'] == True)
    count1 = filter1.sum()
    log_and_print(f"Filter 1: emergency_keywords AND location_keywords = {count1:,}", log_entries)

    # Filter 2: urgency_score >= 3 AND has address_components
    filter2 = (df['urgency_score'] >= 3) & (df['address_components'].notna()) & (df['address_components'] != '')
    count2 = filter2.sum()
    log_and_print(f"Filter 2: urgency_score >= 3 AND has address_components = {count2:,}", log_entries)

    # Combined filter (OR)
    combined_filter = filter1 | filter2
    log_and_print(f"Combined (Filter1 OR Filter2) = {combined_filter.sum():,}", log_entries)

    # Apply filter
    df_filtered = df[combined_filter].copy()
    log_and_print(f"\nExtracted: {len(df_filtered):,} tweets ({len(df_filtered)/len(df)*100:.2f}%)\n", log_entries)

    # Select columns
    output_columns = [
        'id', 'created_at', 'date', 'time',
        'user_id', 'username',
        'tweet', 'tweet_clean', 'tweet_normalized',
        'urgency_score',
        'address_components',
        'has_emergency_keywords', 'has_location_keywords', 'has_affected_area',
        'total_engagement',
        'mentions_official',
        'hashtags', 'mentions', 'photos', 'urls'
    ]

    # Keep only available columns
    available_cols = [c for c in output_columns if c in df_filtered.columns]
    df_filtered = df_filtered[available_cols]

    # Sort by urgency_score (desc), then total_engagement (desc)
    df_filtered = df_filtered.sort_values(
        ['urgency_score', 'total_engagement'],
        ascending=[False, False]
    )

    # ========== SUMMARY STATISTICS ==========
    log_and_print("=" * 60, log_entries)
    log_and_print("SUMMARY STATISTICS", log_entries)
    log_and_print("=" * 60, log_entries)

    # 1. Total tweets
    log_and_print(f"\n1. TOTAL TWEETS EXTRACTED: {len(df_filtered):,}\n", log_entries)

    # 2. Urgency score distribution
    urgency_dist = df_filtered['urgency_score'].value_counts().sort_index()
    log_and_print("2. URGENCY SCORE DISTRIBUTION:", log_entries)
    for score, count in urgency_dist.items():
        log_and_print(f"   Score {score}: {count:,} tweets", log_entries)

    # 3. Top 10 address patterns
    log_and_print("\n3. TOP 10 ADDRESS PATTERNS:", log_entries)
    all_addresses = df_filtered['address_components'].dropna().str.lower()

    # Extract individual components
    address_parts = []
    for addr in all_addresses:
        if addr:
            parts = [p.strip() for p in str(addr).split('|') if p.strip()]
            address_parts.extend(parts)

    # Find common patterns (first word of each component)
    pattern_counter = Counter()
    for part in address_parts:
        # Extract the key pattern (e.g., "mahallesi", "sokak", etc.)
        words = part.split()
        if words:
            pattern_counter[words[0]] += 1

    for pattern, count in pattern_counter.most_common(10):
        log_and_print(f"   '{pattern}': {count:,} occurrences", log_entries)

    # 4. Temporal distribution
    log_and_print("\n4. TEMPORAL DISTRIBUTION (tweets per day):", log_entries)
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    daily_counts = df_filtered.groupby(df_filtered['date'].dt.date).size().sort_index()
    for date, count in daily_counts.items():
        log_and_print(f"   {date}: {count:,} tweets", log_entries)

    # 5. Top users
    log_and_print("\n5. TOP 10 USERS (most emergency+location tweets):", log_entries)
    top_users = df_filtered['username'].value_counts().head(10)
    for username, count in top_users.items():
        log_and_print(f"   @{username}: {count:,} tweets", log_entries)

    # ========== SAMPLE OUTPUT ==========
    log_and_print("\n" + "=" * 60, log_entries)
    log_and_print("SAMPLE EMERGENCY TWEETS (Top 20 by urgency)", log_entries)
    log_and_print("=" * 60, log_entries)

    sample_df = df_filtered.head(20)
    for idx, row in sample_df.iterrows():
        log_and_print(f"\n--- Tweet #{sample_df.index.get_loc(idx) + 1} ---", log_entries)
        log_and_print(f"Urgency Score: {row['urgency_score']}", log_entries)
        log_and_print(f"User: @{row['username']}", log_entries)
        log_and_print(f"Time: {row['created_at']}", log_entries)
        log_and_print(f"Engagement: {row['total_engagement']}", log_entries)
        log_and_print(f"Tweet: {str(row['tweet_clean'])[:300]}", log_entries)
        log_and_print(f"Address Components: {str(row['address_components'])[:200]}", log_entries)

    # ========== SAVE OUTPUTS ==========
    log_and_print("\n" + "=" * 60, log_entries)
    log_and_print("SAVING FILES", log_entries)
    log_and_print("=" * 60, log_entries)

    # Save filtered dataset
    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    log_and_print(f"Saved: {OUTPUT_FILE} ({len(df_filtered):,} rows)", log_entries)

    # Save log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_entries))
    log_and_print(f"Saved: {LOG_FILE}", log_entries)

    log_and_print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_entries)

    return df_filtered

if __name__ == "__main__":
    main()
