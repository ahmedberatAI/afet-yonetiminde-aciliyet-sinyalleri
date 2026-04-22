"""
Emergency Call Detection - Data Preprocessing Pipeline
========================================================
Kahramanmaras Earthquake Twitter Data (February 6, 2023)

This script preprocesses raw Twitter data to prepare it for:
1. Emergency/help request detection
2. Location information extraction

Author: Data Preprocessing Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'deprem.csv'
OUTPUT_FILE = 'data/processed/deprem_cleaned_no_dup_removal.csv'
LOG_FILE = 'data/processed/preprocessing_log_no_dup_removal.txt'

class PreprocessingLogger:
    """Logger for documenting all preprocessing decisions and statistics."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.log_entries = []

    def log(self, message, section=None):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if section:
            entry = f"\n{'='*60}\n{section}\n{'='*60}\n{message}"
        else:
            entry = f"[{timestamp}] {message}"
        self.log_entries.append(entry)
        # Safe print for Windows console encoding
        try:
            print(entry)
        except UnicodeEncodeError:
            # Replace problematic characters for console output only
            safe_entry = entry.encode('ascii', 'replace').decode('ascii')
            print(safe_entry)

    def save(self):
        """Save log to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("EMERGENCY DATA PREPROCESSING LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write('\n'.join(self.log_entries))


def analyze_raw_data(df, logger):
    """Analyze the raw dataset to understand its structure and content."""

    logger.log("""
OBJECTIVE: Analyze raw data to inform preprocessing decisions.
We need to understand what we're working with before making any changes.
""", "STEP 1: RAW DATA ANALYSIS")

    # Basic statistics
    logger.log(f"""
BASIC STATISTICS:
- Total rows: {len(df):,}
- Total columns: {len(df.columns)}
- Columns: {list(df.columns)}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
""")

    # Date range
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    logger.log(f"""
TEMPORAL COVERAGE:
- Earliest tweet: {df['created_at'].min()}
- Latest tweet: {df['created_at'].max()}
- Duration: {(df['created_at'].max() - df['created_at'].min()).days} days
""")

    # Language distribution
    lang_dist = df['language'].value_counts().head(10)
    logger.log(f"""
LANGUAGE DISTRIBUTION (Top 10):
{lang_dist.to_string()}

DECISION: Keep all languages - emergency calls may be in any language,
and Turkish speakers abroad might tweet in other languages.
""")

    # Tweet vs Retweet analysis
    retweet_counts = df['retweet'].value_counts()
    logger.log(f"""
RETWEET ANALYSIS:
{retweet_counts.to_string()}

CRITICAL DECISION: Retweets are VALUABLE for emergency detection!
- Retweets help spread emergency calls to wider audience
- A retweeted emergency call indicates it got attention
- We will KEEP retweets but mark them for downstream filtering if needed
""")

    # Quote tweet analysis
    quote_counts = df['quote_url'].notna().sum()
    logger.log(f"""
QUOTE TWEET ANALYSIS:
- Tweets with quote URLs: {quote_counts:,}
- Original tweets: {len(df) - quote_counts:,}
""")

    # Missing values analysis
    missing_analysis = df.isnull().sum()
    missing_pct = (missing_analysis / len(df) * 100).round(2)
    logger.log(f"""
MISSING VALUES ANALYSIS:
{pd.DataFrame({'Missing': missing_analysis, 'Percentage': missing_pct}).to_string()}

DECISION: Missing values in optional fields (geo, near, place) are expected.
These fields only populate when user enables location services.
""")

    # Tweet length analysis
    df['tweet_length'] = df['tweet'].astype(str).str.len()
    logger.log(f"""
TWEET LENGTH ANALYSIS:
- Min length: {df['tweet_length'].min()}
- Max length: {df['tweet_length'].max()}
- Mean length: {df['tweet_length'].mean():.2f}
- Median length: {df['tweet_length'].median():.2f}

DECISION: Do NOT remove short tweets - emergency calls can be very short:
"YARDIM: [address]" or "ENKAZ: [location]"
""")

    return df


def detect_emergency_patterns(df, logger):
    """Analyze tweets for emergency-related patterns."""

    logger.log("""
OBJECTIVE: Identify patterns that indicate emergency calls.
This analysis will help us understand what we're looking for.
""", "STEP 2: EMERGENCY PATTERN ANALYSIS")

    # Turkish emergency keywords
    emergency_keywords = [
        'yardım', 'yardim', 'imdat', 'acil', 'aci̇l',
        'enkaz', 'göçük', 'gocuk', 'kurtarın', 'kurtarin',
        'mahsur', 'altında', 'altinda', 'sesler', 'ses geliyor',
        'kurtarma', 'ambulans', 'can var', 'hayat var'
    ]

    # Location keywords
    location_keywords = [
        'mahalle', 'mah.', 'mah ', 'sokak', 'sok.', 'sok ',
        'cadde', 'cad.', 'cad ', 'apartman', 'apt.', 'apt ',
        'bina', 'kat', 'blok', 'site', 'adres'
    ]

    # City/district names (earthquake affected areas)
    affected_areas = [
        'kahramanmaraş', 'kahramanmaras', 'hatay', 'antakya',
        'gaziantep', 'adıyaman', 'adiyaman', 'malatya',
        'diyarbakır', 'diyarbakir', 'osmaniye', 'kilis',
        'şanlıurfa', 'sanliurfa', 'adana', 'iskenderun',
        'pazarcık', 'pazarcik', 'nurdağı', 'nurdagi'
    ]

    tweet_lower = df['tweet'].astype(str).str.lower()

    # Count emergency keyword matches
    emergency_mask = tweet_lower.str.contains('|'.join(emergency_keywords), na=False, regex=True)
    location_mask = tweet_lower.str.contains('|'.join(location_keywords), na=False, regex=True)
    area_mask = tweet_lower.str.contains('|'.join(affected_areas), na=False, regex=True)

    # Combined potential emergency tweets
    potential_emergency = emergency_mask & (location_mask | area_mask)
    high_priority = emergency_mask & location_mask & area_mask

    logger.log(f"""
EMERGENCY PATTERN DETECTION:

1. Tweets containing emergency keywords: {emergency_mask.sum():,} ({emergency_mask.mean()*100:.2f}%)
   Keywords: {emergency_keywords[:5]}...

2. Tweets containing location indicators: {location_mask.sum():,} ({location_mask.mean()*100:.2f}%)
   Keywords: {location_keywords[:5]}...

3. Tweets mentioning affected areas: {area_mask.sum():,} ({area_mask.mean()*100:.2f}%)
   Areas: {affected_areas[:5]}...

4. POTENTIAL EMERGENCY CALLS (emergency + location/area): {potential_emergency.sum():,} ({potential_emergency.mean()*100:.2f}%)

5. HIGH PRIORITY (emergency + location + area): {high_priority.sum():,} ({high_priority.mean()*100:.2f}%)
""")

    # Sample emergency tweets
    emergency_samples = df[high_priority]['tweet'].head(5).tolist()
    logger.log(f"""
SAMPLE HIGH-PRIORITY EMERGENCY TWEETS:
""")
    for i, sample in enumerate(emergency_samples, 1):
        logger.log(f"  {i}. {sample[:200]}...")

    # Add temporary flags for analysis
    df['has_emergency_keywords'] = emergency_mask
    df['has_location_keywords'] = location_mask
    df['has_affected_area'] = area_mask

    return df


def analyze_duplicates(df, logger):
    """Carefully analyze duplicates before removing any."""

    logger.log("""
OBJECTIVE: Understand duplicate tweets BEFORE removing them.
We must be careful not to remove valuable emergency signals.
""", "STEP 3: DUPLICATE ANALYSIS")

    # Exact text duplicates
    exact_text_dups = df.duplicated(subset=['tweet'], keep=False)
    exact_text_groups = df[exact_text_dups].groupby('tweet').size()

    logger.log(f"""
EXACT TEXT DUPLICATE ANALYSIS:
- Tweets that appear more than once: {exact_text_dups.sum():,}
- Number of unique duplicate texts: {len(exact_text_groups):,}
- Max repetitions of single tweet: {exact_text_groups.max() if len(exact_text_groups) > 0 else 0}
""")

    # Check if duplicates are from same user (spammy) or different users (spreading info)
    text_user_combos = df.groupby('tweet')['username'].nunique()
    multi_user_texts = text_user_combos[text_user_combos > 1]

    logger.log(f"""
DUPLICATE SPREADING ANALYSIS:
- Unique tweets shared by MULTIPLE users: {len(multi_user_texts):,}
  (These are likely being shared to spread important information)

- Duplicates from SAME user: {(text_user_combos == 1).sum():,}
  (These might be spam or emphasis)
""")

    # ID duplicates (exact same tweet entry)
    id_dups = df.duplicated(subset=['id'], keep=False)
    logger.log(f"""
ID-BASED DUPLICATES:
- Exact duplicate tweet IDs: {id_dups.sum():,}
- These are true duplicates (same tweet scraped multiple times)
""")

    # Check emergency tweets in duplicates
    dup_emergency = df[exact_text_dups & df['has_emergency_keywords']]
    logger.log(f"""
EMERGENCY CONTENT IN DUPLICATES:
- Duplicate tweets with emergency keywords: {len(dup_emergency):,}

CRITICAL DECISION:
1. REMOVE exact ID duplicates (same tweet entry)
2. KEEP text duplicates from different users (spreading emergency info)
3. MARK but keep text duplicates from same user (might be emphasis)
""")

    return df


def clean_text(text):
    """Clean tweet text while preserving critical information."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode characters (Turkish characters)
    text = unicodedata.normalize('NFC', text)

    # Keep URLs but mark them (they might contain location info)
    # Don't remove them yet

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_turkish_text(text):
    """Create a normalized version for NLP processing."""
    if pd.isna(text) or text == "":
        return ""

    text = str(text).lower()

    # Turkish character normalization mapping
    # Keep both original and normalized for matching
    turkish_map = {
        'ı': 'i',  # dotless i to i
        'ğ': 'g',
        'ü': 'u',
        'ş': 's',
        'ö': 'o',
        'ç': 'c',
        'İ': 'i',
        'Ğ': 'g',
        'Ü': 'u',
        'Ş': 's',
        'Ö': 'o',
        'Ç': 'c'
    }

    for turkish_char, latin_char in turkish_map.items():
        text = text.replace(turkish_char, latin_char)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_features(df, logger):
    """Extract features useful for emergency detection."""

    logger.log("""
OBJECTIVE: Create features that help identify emergency calls.
These features will be used for filtering and ML models.
""", "STEP 4: FEATURE EXTRACTION")

    # Text cleaning
    logger.log("Cleaning tweet text...")
    df['tweet_clean'] = df['tweet'].apply(clean_text)

    # Normalized text for NLP
    logger.log("Creating normalized text for NLP...")
    df['tweet_normalized'] = df['tweet_clean'].apply(normalize_turkish_text)

    # Extract potential addresses (pattern-based)
    address_patterns = [
        r'(?:mahalle|mah\.?)\s*[:\s]?\s*[\w\s]+',
        r'(?:sokak|sok\.?|sk\.?)\s*[:\s]?\s*[\w\s]+',
        r'(?:cadde|cad\.?|cd\.?)\s*[:\s]?\s*[\w\s]+',
        r'(?:apartman|apt\.?)\s*[:\s]?\s*[\w\s]+',
        r'(?:no|numara)[:\s]?\s*\d+',
        r'(?:kat)[:\s]?\s*\d+',
        r'(?:blok)[:\s]?\s*[a-zA-Z\d]+'
    ]

    def extract_address_components(text):
        """Extract potential address components from text."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        matches = []
        for pattern in address_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)
        return ' | '.join(matches) if matches else ""

    logger.log("Extracting address components...")
    df['address_components'] = df['tweet_clean'].apply(extract_address_components)

    # Emergency urgency indicators
    urgency_patterns = {
        'critical': ['acil', 'aci̇l', 'çok acil', 'imdat', 'ölüyor', 'son dakika'],
        'trapped': ['enkaz altında', 'göçük altında', 'mahsur', 'altında kaldı'],
        'rescue_needed': ['kurtarın', 'yardım', 'ses geliyor', 'can var', 'canlı var'],
        'location_given': ['adres', 'lokasyon', 'konum', 'mahalle', 'sokak']
    }

    def calculate_urgency_score(text):
        """Calculate urgency score based on keywords."""
        if pd.isna(text):
            return 0
        text = str(text).lower()
        score = 0
        for category, keywords in urgency_patterns.items():
            for keyword in keywords:
                if keyword in text:
                    if category == 'critical':
                        score += 3
                    elif category == 'trapped':
                        score += 3
                    elif category == 'rescue_needed':
                        score += 2
                    elif category == 'location_given':
                        score += 1
        return score

    logger.log("Calculating urgency scores...")
    df['urgency_score'] = df['tweet_normalized'].apply(calculate_urgency_score)

    # AFAD/official mention indicator
    official_accounts = ['afad', 'afadbaskanlik', 'afadturkiye', 'depremdairesi',
                         'haluklevent', 'ahbap', 'kizilay']

    def check_official_mention(mentions):
        """Check if tweet mentions official accounts."""
        if pd.isna(mentions) or mentions == '[]':
            return False
        mentions_str = str(mentions).lower()
        return any(acc in mentions_str for acc in official_accounts)

    df['mentions_official'] = df['mentions'].apply(check_official_mention)

    # Has media (photos might show location/damage)
    df['has_photos'] = df['photos'].apply(lambda x: x != '[]' and pd.notna(x))
    df['has_video'] = df['video'].apply(lambda x: x == 1 if pd.notna(x) else False)

    # Tweet engagement (high engagement might indicate important info)
    df['total_engagement'] = df['replies_count'].fillna(0) + df['retweets_count'].fillna(0) + df['likes_count'].fillna(0)

    logger.log(f"""
FEATURES CREATED:
1. tweet_clean: Cleaned original text
2. tweet_normalized: Lowercase, Turkish chars normalized
3. address_components: Extracted address parts (if any)
4. urgency_score: 0-10+ score based on emergency keywords
5. mentions_official: Boolean - mentions AFAD/official accounts
6. has_photos: Boolean - has attached photos
7. has_video: Boolean - has attached video
8. total_engagement: replies + retweets + likes

URGENCY SCORE DISTRIBUTION:
{df['urgency_score'].value_counts().head(10).to_string()}

DECISION: Keep urgency_score > 0 tweets prioritized for review.
Urgency score > 3 indicates likely emergency call.
""")

    # High priority tweet stats
    high_urgency = df[df['urgency_score'] >= 3]
    logger.log(f"""
HIGH URGENCY TWEETS (score >= 3): {len(high_urgency):,}
Sample high urgency tweets:
""")
    for i, row in high_urgency.head(3).iterrows():
        logger.log(f"  Score {row['urgency_score']}: {row['tweet_clean'][:150]}...")

    return df


def remove_duplicates(df, logger):
    """Remove duplicates intelligently. DISABLED - keeping all duplicates."""

    logger.log("""
OBJECTIVE: Duplicate removal DISABLED for this run.
All duplicates (ID and text-based) are being kept.
""", "STEP 5: DUPLICATE REMOVAL (DISABLED)")

    initial_count = len(df)

    # # Step 1: Remove exact ID duplicates (keep first occurrence) - DISABLED
    # df = df.drop_duplicates(subset=['id'], keep='first')
    # after_id_dedup = len(df)

    logger.log(f"""
STEP 5a - Remove ID duplicates: SKIPPED
- Total tweets: {initial_count:,}
- All ID duplicates retained
""")

    # # Step 2: For same-user same-text tweets - DISABLED
    # Mark tweets that are duplicates from same user (for analysis only)
    df['is_same_user_dup'] = df.duplicated(subset=['username', 'tweet_clean'], keep=False)

    # For non-emergency duplicates from same user, keep highest engagement
    non_emergency_same_user = df[df['is_same_user_dup'] & (df['urgency_score'] == 0)]
    emergency_same_user = df[df['is_same_user_dup'] & (df['urgency_score'] > 0)]

    logger.log(f"""
STEP 5b - Analyze same-user duplicates (NO REMOVAL):
- Same-user duplicate tweets: {df['is_same_user_dup'].sum():,}
- Of which are emergency-related: {len(emergency_same_user):,}
- Non-emergency same-user dups: {len(non_emergency_same_user):,}

All duplicates are being KEPT (removal disabled).
""")

    # # Remove non-emergency same-user duplicates - DISABLED
    # idx_to_remove = []
    # for (username, tweet), group in non_emergency_same_user.groupby(['username', 'tweet_clean']):
    #     if len(group) > 1:
    #         keep_idx = group['total_engagement'].idxmax()
    #         remove_idx = group.index[group.index != keep_idx].tolist()
    #         idx_to_remove.extend(remove_idx)
    # df = df.drop(idx_to_remove)

    logger.log(f"""
STEP 5c - Remove non-emergency same-user duplicates: SKIPPED
- All same-user duplicates retained
""")

    # Clean up temporary columns
    df = df.drop(columns=['is_same_user_dup'])

    logger.log(f"""
DUPLICATE REMOVAL SUMMARY:
- Initial count: {initial_count:,}
- Final count: {initial_count:,}
- Total removed: 0 (0.00%)

NOTE: Duplicate removal is DISABLED. All tweets retained.
""")

    return df


def final_cleanup(df, logger):
    """Final cleanup and column organization."""

    logger.log("""
OBJECTIVE: Final cleanup and organize columns for output.
""", "STEP 6: FINAL CLEANUP")

    # Remove truly empty tweets
    initial = len(df)
    df = df[df['tweet_clean'].str.len() > 0]
    after_empty = len(df)

    logger.log(f"""
Empty tweet removal:
- Removed {initial - after_empty:,} empty tweets
""")

    # Select and reorder columns
    core_columns = [
        # IDs and metadata
        'id', 'conversation_id', 'created_at', 'date', 'time',
        # User info
        'user_id', 'username', 'name',
        # Tweet content
        'tweet', 'tweet_clean', 'tweet_normalized',
        # Language
        'language',
        # Engagement
        'replies_count', 'retweets_count', 'likes_count', 'total_engagement',
        # Features
        'address_components', 'urgency_score',
        'has_emergency_keywords', 'has_location_keywords', 'has_affected_area',
        'mentions_official', 'has_photos', 'has_video',
        # Original fields that might be useful
        'hashtags', 'mentions', 'urls', 'photos',
        'retweet', 'quote_url',
        'link',
        # Location fields (often empty but valuable when present)
        'place', 'near', 'geo'
    ]

    # Keep only columns that exist
    available_columns = [col for col in core_columns if col in df.columns]
    df = df[available_columns]

    logger.log(f"""
FINAL COLUMNS SELECTED:
{available_columns}

Total columns: {len(available_columns)}
""")

    # Sort by urgency score (high priority first) then by date
    df = df.sort_values(['urgency_score', 'created_at'], ascending=[False, True])

    return df


def generate_summary(df, logger):
    """Generate final summary statistics."""

    logger.log("""
FINAL DATASET SUMMARY
""", "PREPROCESSING COMPLETE")

    logger.log(f"""
DATASET STATISTICS:
==================
Total tweets: {len(df):,}
Date range: {df['created_at'].min()} to {df['created_at'].max()}

LANGUAGE DISTRIBUTION:
{df['language'].value_counts().head(5).to_string()}

URGENCY SCORE DISTRIBUTION:
{df['urgency_score'].value_counts().sort_index().to_string()}

HIGH PRIORITY TWEETS (urgency >= 3): {(df['urgency_score'] >= 3).sum():,}
MEDIUM PRIORITY (urgency 1-2): {((df['urgency_score'] >= 1) & (df['urgency_score'] < 3)).sum():,}

TWEETS WITH ADDRESS COMPONENTS: {(df['address_components'] != '').sum():,}
TWEETS MENTIONING OFFICIALS: {df['mentions_official'].sum():,}
TWEETS WITH PHOTOS: {df['has_photos'].sum():,}
TWEETS WITH VIDEO: {df['has_video'].sum():,}

ENGAGEMENT STATISTICS:
- Max engagement: {df['total_engagement'].max():,}
- Mean engagement: {df['total_engagement'].mean():.2f}
- Median engagement: {df['total_engagement'].median():.2f}
""")

    # Sample high-priority tweets for manual review
    high_priority = df[df['urgency_score'] >= 3].head(10)
    logger.log("""
SAMPLE HIGH-PRIORITY EMERGENCY TWEETS (for validation):
""")
    for i, row in high_priority.iterrows():
        logger.log(f"""
Tweet ID: {row['id']}
User: @{row['username']}
Urgency: {row['urgency_score']}
Text: {row['tweet_clean'][:300]}
Address components: {row['address_components'][:100] if row['address_components'] else 'None'}
---
""")


def main():
    """Main preprocessing pipeline."""

    # Initialize logger
    logger = PreprocessingLogger(LOG_FILE)

    logger.log("""
EMERGENCY CALL DETECTION - DATA PREPROCESSING PIPELINE
======================================================
Project: Kahramanmaras Earthquake Twitter Analysis
Goal: Prepare data for emergency call detection and location extraction

PHILOSOPHY:
- PRESERVE tweets that might contain emergency calls or location info
- DOCUMENT every decision and its rationale
- EXTRACT features that help identify urgent content
- AVOID aggressive cleaning that might remove critical information
""", "PIPELINE START")

    # Load data
    logger.log(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        logger.log(f"Successfully loaded {len(df):,} rows")
    except Exception as e:
        logger.log(f"Error loading data: {e}")
        return

    # Step 1: Analyze raw data
    df = analyze_raw_data(df, logger)

    # Step 2: Detect emergency patterns (before any cleaning)
    df = detect_emergency_patterns(df, logger)

    # Step 3: Analyze duplicates
    df = analyze_duplicates(df, logger)

    # Step 4: Extract features
    df = extract_features(df, logger)

    # Step 5: Remove duplicates (carefully)
    df = remove_duplicates(df, logger)

    # Step 6: Final cleanup
    df = final_cleanup(df, logger)

    # Generate summary
    generate_summary(df, logger)

    # Save outputs
    logger.log(f"\nSaving cleaned dataset to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    logger.log(f"Successfully saved {len(df):,} rows")

    # Save log
    logger.save()
    logger.log(f"Log saved to {LOG_FILE}")

    logger.log("""
RECOMMENDED NEXT STEPS:
=======================
1. EMERGENCY FILTERING: Filter tweets with urgency_score >= 3 for immediate review
2. LOCATION EXTRACTION: Use address_components and NER on high-priority tweets
3. CLASSIFICATION: Train a model using urgency features to classify emergency calls
4. CROWDSOURCING: High-engagement tweets often contain verified information
5. TIMELINE ANALYSIS: Analyze tweet volume over time to identify peak crisis periods

IMPORTANT NOTES:
- The 'tweet' column contains original text (preserve for reference)
- The 'tweet_normalized' column is for NLP processing
- urgency_score is a heuristic - manual validation recommended
- Different-user text duplicates are retained (information spreading)
""", "PIPELINE COMPLETE")

    return df


if __name__ == "__main__":
    main()
