"""
Build Earthquake Region Neighborhood Gazetteer
Collects neighborhoods from OSM Nominatim API for earthquake-affected provinces
"""

import requests
import pandas as pd
import time
import os
import re
import json
from datetime import datetime
from collections import defaultdict
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
OUTPUT_DIR = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\gazetteer"
ANALYSIS_DIR = r"C:\Users\HUAWE\OneDrive - Ankara Üniversitesi\Masaüstü\afetYonetimi\data\analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file
log_file = os.path.join(OUTPUT_DIR, 'gazetteer_collection_log.txt')
log_messages = []

def log(msg):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_messages.append(full_msg)

def save_log():
    """Save log to file"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_messages))

# Define earthquake-affected regions with districts
REGIONS = {
    'Hatay': {
        'districts': [
            'Antakya', 'İskenderun', 'Kırıkhan', 'Dörtyol', 'Samandağ',
            'Defne', 'Arsuz', 'Payas', 'Hassa', 'Belen', 'Yayladağı',
            'Altınözü', 'Kumlu', 'Reyhanlı', 'Erzin'
        ],
        'priority': 1
    },
    'Kahramanmaraş': {
        'districts': [
            'Onikişubat', 'Dulkadiroğlu', 'Pazarcık', 'Elbistan', 'Afşin',
            'Türkoğlu', 'Göksun', 'Andırın', 'Nurhak', 'Çağlayancerit', 'Ekinözü'
        ],
        'priority': 1
    },
    'Adıyaman': {
        'districts': [
            'Merkez', 'Gölbaşı', 'Besni', 'Kahta', 'Çelikhan', 'Gerger',
            'Samsat', 'Sincik', 'Tut'
        ],
        'priority': 1
    },
    'Gaziantep': {
        'districts': [
            'Şahinbey', 'Şehitkamil', 'İslahiye', 'Nurdağı', 'Oğuzeli',
            'Nizip', 'Araban', 'Yavuzeli', 'Karkamış'
        ],
        'priority': 1
    },
    'Malatya': {
        'districts': [
            'Battalgazi', 'Yeşilyurt', 'Doğanşehir', 'Akçadağ', 'Darende',
            'Hekimhan', 'Arguvan', 'Arapgir', 'Yazıhan', 'Pütürge', 'Kale', 'Kuluncak', 'Doğanyol'
        ],
        'priority': 2
    },
    'Osmaniye': {
        'districts': [
            'Merkez', 'Kadirli', 'Düziçi', 'Bahçe', 'Toprakkale',
            'Hasanbeyli', 'Sumbas'
        ],
        'priority': 2
    },
    'Diyarbakır': {
        'districts': [
            'Bağlar', 'Kayapınar', 'Sur', 'Yenişehir', 'Bismil',
            'Ergani', 'Çınar', 'Silvan', 'Çermik', 'Dicle', 'Eğil',
            'Hani', 'Hazro', 'Kocaköy', 'Kulp', 'Lice'
        ],
        'priority': 2
    },
    'Şanlıurfa': {
        'districts': [
            'Eyyübiye', 'Haliliye', 'Karaköprü', 'Akçakale', 'Birecik',
            'Bozova', 'Ceylanpınar', 'Halfeti', 'Harran', 'Hilvan',
            'Siverek', 'Suruç', 'Viranşehir'
        ],
        'priority': 2
    },
    'Kilis': {
        'districts': [
            'Merkez', 'Elbeyli', 'Musabeyli', 'Polateli'
        ],
        'priority': 2
    },
    'Adana': {
        'districts': [
            'Seyhan', 'Yüreğir', 'Çukurova', 'Sarıçam', 'Ceyhan',
            'Kozan', 'İmamoğlu', 'Karaisalı', 'Yumurtalık', 'Karataş',
            'Pozantı', 'Aladağ', 'Feke', 'Saimbeyli', 'Tufanbeyli'
        ],
        'priority': 2
    }
}

# Known neighborhoods from our tweet analysis (for validation)
KNOWN_NEIGHBORHOODS = [
    'akevler', 'odabaşı', 'cumhuriyet', 'cebrail', 'hayrullah', 'kanatlı',
    'ekinci', 'şazibey', 'atatürk', 'bahçelievler', 'ürgenpaşa', 'alitaşı',
    'akasya', 'armutlu', 'yurt', 'sümerler', 'emek', 'gazi', 'fatih',
    'harbiye', 'sakarya', 'dumlupınar', 'esentepe', 'yeşilyurt', 'aksaray'
]

class NeighborhoodCollector:
    def __init__(self):
        self.neighborhoods = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DisasterManagement/1.0 (Earthquake Response Research; contact@research.edu)'
        })
        self.rate_limit_delay = 1.5  # seconds between requests

    def normalize_name(self, name):
        """Normalize neighborhood name for matching"""
        if not name:
            return ''
        # Lowercase
        name = name.lower()
        # Remove common suffixes
        name = re.sub(r'\s*(mahallesi|mah\.?|köyü|köy)\s*$', '', name)
        # Normalize Turkish characters for matching
        replacements = {
            'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
        }
        normalized = name
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return name.strip(), normalized.strip()

    def search_osm_neighborhoods(self, province, district=None):
        """Search OSM for neighborhoods in a district/province"""
        base_url = "https://nominatim.openstreetmap.org/search"

        # Build search query
        if district:
            query = f"{district}, {province}, Turkey"
        else:
            query = f"{province}, Turkey"

        params = {
            'q': query,
            'format': 'json',
            'addressdetails': 1,
            'limit': 50,
            'accept-language': 'tr'
        }

        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except Exception as e:
            log(f"  Error searching OSM for {query}: {e}")
            return []

    def search_osm_by_type(self, province, district, place_type='suburb'):
        """Search OSM for specific place types (suburb = mahalle)"""
        base_url = "https://nominatim.openstreetmap.org/search"

        # Search for suburbs (neighborhoods) in the district
        query = f"{district}, {province}, Turkey"

        params = {
            'q': query,
            'format': 'json',
            'addressdetails': 1,
            'limit': 100,
            'featuretype': place_type,
            'accept-language': 'tr'
        }

        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except Exception as e:
            log(f"  Error in type search for {district}: {e}")
            return []

    def search_overpass_neighborhoods(self, province, district):
        """Use Overpass API to find neighborhoods (mahalle) in a district"""
        overpass_url = "https://overpass-api.de/api/interpreter"

        # Overpass query for places tagged as neighbourhoods/suburbs in the area
        query = f"""
        [out:json][timeout:60];
        area["name"="{province}"]["admin_level"="4"]->.province;
        area["name"="{district}"]->.district;
        (
          node["place"~"neighbourhood|suburb|quarter"](area.province);
          node["place"~"neighbourhood|suburb|quarter"](area.district);
          way["place"~"neighbourhood|suburb|quarter"](area.province);
          way["place"~"neighbourhood|suburb|quarter"](area.district);
        );
        out body;
        """

        try:
            response = self.session.post(overpass_url, data={'data': query}, timeout=120)
            response.raise_for_status()
            time.sleep(2)  # Longer delay for Overpass
            data = response.json()
            return data.get('elements', [])
        except Exception as e:
            log(f"  Overpass error for {district}, {province}: {e}")
            return []

    def collect_from_osm(self, province, districts):
        """Collect neighborhoods from OSM for a province"""
        log(f"\nCollecting from OSM: {province}")
        collected = []

        for district in districts:
            log(f"  District: {district}")

            # Method 1: Direct search
            results = self.search_osm_neighborhoods(province, district)
            for item in results:
                addr = item.get('address', {})
                neighborhood = addr.get('suburb') or addr.get('neighbourhood') or addr.get('quarter')
                if neighborhood:
                    name_clean, name_normalized = self.normalize_name(neighborhood)
                    collected.append({
                        'neighborhood_name': neighborhood,
                        'neighborhood_clean': name_clean,
                        'neighborhood_normalized': name_normalized,
                        'district': district,
                        'province': province,
                        'source': 'osm_nominatim',
                        'osm_type': item.get('type', ''),
                        'lat': item.get('lat'),
                        'lon': item.get('lon')
                    })

            # Method 2: Overpass API for more complete coverage
            overpass_results = self.search_overpass_neighborhoods(province, district)
            for item in overpass_results:
                name = item.get('tags', {}).get('name')
                if name:
                    name_clean, name_normalized = self.normalize_name(name)
                    collected.append({
                        'neighborhood_name': name,
                        'neighborhood_clean': name_clean,
                        'neighborhood_normalized': name_normalized,
                        'district': district,
                        'province': province,
                        'source': 'osm_overpass',
                        'osm_type': item.get('tags', {}).get('place', ''),
                        'lat': item.get('lat'),
                        'lon': item.get('lon')
                    })

            log(f"    Found {len([c for c in collected if c['district'] == district])} neighborhoods")

        return collected

    def add_known_neighborhoods_from_tweets(self, top_neighborhoods_path):
        """Add neighborhoods extracted from tweets that aren't in OSM"""
        log("\nAdding neighborhoods from tweet analysis...")

        if not os.path.exists(top_neighborhoods_path):
            log(f"  Tweet neighborhoods file not found: {top_neighborhoods_path}")
            return []

        tweet_df = pd.read_csv(top_neighborhoods_path)
        added = []

        # Common district mappings based on our knowledge of the data
        district_mappings = {
            'akevler': ('Antakya', 'Hatay'),
            'odabaşı': ('Antakya', 'Hatay'),
            'cumhuriyet': ('Antakya', 'Hatay'),
            'cebrail': ('Antakya', 'Hatay'),
            'hayrullah': ('Antakya', 'Hatay'),
            'kanatlı': ('Antakya', 'Hatay'),
            'ekinci': ('Antakya', 'Hatay'),
            'ürgenpaşa': ('Antakya', 'Hatay'),
            'alitaşı': ('Merkez', 'Adıyaman'),
            'şazibey': ('Onikişubat', 'Kahramanmaraş'),
            'bahçelievler': ('Dulkadiroğlu', 'Kahramanmaraş'),
            'akasya': ('Antakya', 'Hatay'),
            'armutlu': ('Antakya', 'Hatay'),
            'yurt': ('Seyhan', 'Adana'),
            'sümerler': ('Antakya', 'Hatay'),
            'kemal': ('İskenderun', 'Hatay'),  # Mustafa Kemal
            'selim': ('Nurdağı', 'Gaziantep'),  # Yavuz Selim
            'reis': ('Merkez', 'Adıyaman'),  # Turgut Reis
            'çay': ('İskenderun', 'Hatay'),
        }

        for _, row in tweet_df.iterrows():
            name = row['neighborhood_name']
            name_lower = name.lower()

            # Check if we have a mapping
            if name_lower in district_mappings:
                district, province = district_mappings[name_lower]
            else:
                # Default to Antakya/Hatay as most common
                district = 'Unknown'
                province = 'Unknown'

            name_clean, name_normalized = self.normalize_name(name)
            added.append({
                'neighborhood_name': name,
                'neighborhood_clean': name_clean,
                'neighborhood_normalized': name_normalized,
                'district': district,
                'province': province,
                'source': 'tweet_extraction',
                'osm_type': '',
                'lat': None,
                'lon': None,
                'mention_count': row['mention_count']
            })

        log(f"  Added {len(added)} neighborhoods from tweets")
        return added

    def add_manual_neighborhoods(self):
        """Add well-known neighborhoods that might be missing from OSM"""
        log("\nAdding manually verified neighborhoods...")

        # These are neighborhoods we know exist from the tweet data and news
        manual_neighborhoods = [
            # Hatay - Antakya
            ('Akevler', 'Antakya', 'Hatay'),
            ('Odabaşı', 'Antakya', 'Hatay'),
            ('Cumhuriyet', 'Antakya', 'Hatay'),
            ('Cebrail', 'Antakya', 'Hatay'),
            ('Hayrullah', 'Antakya', 'Hatay'),
            ('General Şükrü Kanatlı', 'Antakya', 'Hatay'),
            ('Ekinci', 'Antakya', 'Hatay'),
            ('Ürgenpaşa', 'Antakya', 'Hatay'),
            ('Akasya', 'Antakya', 'Hatay'),
            ('Armutlu', 'Antakya', 'Hatay'),
            ('Sümerler', 'Antakya', 'Hatay'),
            ('Sümerevler', 'Antakya', 'Hatay'),
            ('Güzelburç', 'Antakya', 'Hatay'),
            ('Emek', 'Antakya', 'Hatay'),
            ('Saraykent', 'Antakya', 'Hatay'),
            ('Kışlasaray', 'Antakya', 'Hatay'),
            ('Harbiye', 'Antakya', 'Hatay'),
            ('Küçükdalyan', 'Antakya', 'Hatay'),
            ('Haraparası', 'Antakya', 'Hatay'),
            ('Esenlik', 'Antakya', 'Hatay'),
            ('Numune', 'Antakya', 'Hatay'),
            # Hatay - İskenderun
            ('Mustafa Kemal', 'İskenderun', 'Hatay'),
            ('Çay', 'İskenderun', 'Hatay'),
            ('Barbaros', 'İskenderun', 'Hatay'),
            # Hatay - Defne
            ('Tütün', 'Defne', 'Hatay'),
            # Kahramanmaraş
            ('Şazibey', 'Onikişubat', 'Kahramanmaraş'),
            ('Bahçelievler', 'Dulkadiroğlu', 'Kahramanmaraş'),
            ('Oruç Reis', 'Onikişubat', 'Kahramanmaraş'),
            ('Dumlupınar', 'Onikişubat', 'Kahramanmaraş'),
            ('Yenişehir', 'Onikişubat', 'Kahramanmaraş'),
            # Adıyaman
            ('Alitaşı', 'Merkez', 'Adıyaman'),
            ('Turgut Reis', 'Merkez', 'Adıyaman'),
            ('Mimar Sinan', 'Merkez', 'Adıyaman'),
            ('Siteler', 'Merkez', 'Adıyaman'),
            ('Fatih', 'Merkez', 'Adıyaman'),
            # Gaziantep - Nurdağı
            ('Yavuz Selim', 'Nurdağı', 'Gaziantep'),
            ('Atatürk', 'Nurdağı', 'Gaziantep'),
            # Gaziantep - İslahiye
            ('Cumhuriyet', 'İslahiye', 'Gaziantep'),
            # Adana
            ('Yurt', 'Seyhan', 'Adana'),
            ('Ceyhan', 'Ceyhan', 'Adana'),
            # Malatya
            ('Battalgazi', 'Battalgazi', 'Malatya'),
            ('Yeşilyurt', 'Yeşilyurt', 'Malatya'),
        ]

        added = []
        for name, district, province in manual_neighborhoods:
            name_clean, name_normalized = self.normalize_name(name)
            added.append({
                'neighborhood_name': name,
                'neighborhood_clean': name_clean,
                'neighborhood_normalized': name_normalized,
                'district': district,
                'province': province,
                'source': 'manual',
                'osm_type': 'neighbourhood',
                'lat': None,
                'lon': None
            })

        log(f"  Added {len(added)} manual neighborhoods")
        return added

    def collect_all(self):
        """Collect neighborhoods from all sources"""
        log("=" * 80)
        log("EARTHQUAKE REGION NEIGHBORHOOD GAZETTEER COLLECTION")
        log("=" * 80)

        all_neighborhoods = []

        # Collect from OSM for each region (prioritize primary regions)
        sorted_regions = sorted(REGIONS.items(), key=lambda x: x[1]['priority'])

        for province, config in sorted_regions:
            districts = config['districts']
            neighborhoods = self.collect_from_osm(province, districts)
            all_neighborhoods.extend(neighborhoods)
            log(f"  Total from {province}: {len(neighborhoods)}")

        # Add neighborhoods from tweet analysis
        top_neigh_path = os.path.join(ANALYSIS_DIR, 'top_neighborhoods.csv')
        tweet_neighborhoods = self.add_known_neighborhoods_from_tweets(top_neigh_path)
        all_neighborhoods.extend(tweet_neighborhoods)

        # Add manual neighborhoods
        manual_neighborhoods = self.add_manual_neighborhoods()
        all_neighborhoods.extend(manual_neighborhoods)

        self.neighborhoods = all_neighborhoods
        return all_neighborhoods

    def deduplicate_and_clean(self):
        """Remove duplicates and clean the data"""
        log("\n" + "=" * 80)
        log("DEDUPLICATION AND CLEANING")
        log("=" * 80)

        df = pd.DataFrame(self.neighborhoods)
        log(f"Before deduplication: {len(df)} entries")

        # Sort by source priority (manual > tweet > osm)
        source_priority = {'manual': 0, 'tweet_extraction': 1, 'osm_nominatim': 2, 'osm_overpass': 3}
        df['source_priority'] = df['source'].map(source_priority).fillna(4)
        df = df.sort_values('source_priority')

        # Remove duplicates based on normalized name + district + province
        df = df.drop_duplicates(subset=['neighborhood_normalized', 'district', 'province'], keep='first')

        # Also remove duplicates just on normalized name (keep first which has best source)
        df_unique = df.drop_duplicates(subset=['neighborhood_normalized'], keep='first')

        log(f"After deduplication (strict): {len(df)} entries")
        log(f"After deduplication (by name only): {len(df_unique)} entries")

        # Remove entries with empty names
        df = df[df['neighborhood_clean'].str.len() > 1]

        # Remove single-character names (likely parsing errors)
        df = df[df['neighborhood_clean'].str.len() > 1]

        log(f"After cleaning: {len(df)} entries")

        self.neighborhoods = df.to_dict('records')
        return df

    def validate_against_tweets(self):
        """Compare gazetteer against extracted neighborhoods from tweets"""
        log("\n" + "=" * 80)
        log("VALIDATION AGAINST TWEET DATA")
        log("=" * 80)

        top_neigh_path = os.path.join(ANALYSIS_DIR, 'top_neighborhoods.csv')
        if not os.path.exists(top_neigh_path):
            log("No tweet neighborhood data found for validation")
            return

        tweet_df = pd.read_csv(top_neigh_path)
        gazetteer_df = pd.DataFrame(self.neighborhoods)

        # Get normalized names from gazetteer
        gazetteer_names = set(gazetteer_df['neighborhood_normalized'].str.lower())

        # Check coverage of top tweet neighborhoods
        matched = 0
        unmatched = []

        for _, row in tweet_df.head(100).iterrows():
            name = row['neighborhood_name'].lower()
            # Normalize for comparison
            name_normalized = name
            for old, new in [('ı', 'i'), ('ğ', 'g'), ('ü', 'u'), ('ş', 's'), ('ö', 'o'), ('ç', 'c')]:
                name_normalized = name_normalized.replace(old, new)

            if name in gazetteer_names or name_normalized in gazetteer_names:
                matched += 1
            else:
                unmatched.append((name, row['mention_count']))

        log(f"\nTop 100 tweet neighborhoods coverage:")
        log(f"  Matched: {matched}/100 ({matched}%)")
        log(f"  Unmatched: {100-matched}/100")

        if unmatched:
            log("\nUnmatched neighborhoods (top 20):")
            for name, count in unmatched[:20]:
                log(f"  - {name} ({count} mentions)")

        return matched, unmatched

    def generate_statistics(self):
        """Generate statistics report"""
        log("\n" + "=" * 80)
        log("STATISTICS REPORT")
        log("=" * 80)

        df = pd.DataFrame(self.neighborhoods)

        log(f"\nTotal neighborhoods: {len(df)}")

        # By province
        log("\nBreakdown by Province:")
        province_counts = df['province'].value_counts()
        for province, count in province_counts.items():
            log(f"  {province}: {count}")

        # By source
        log("\nBreakdown by Source:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            log(f"  {source}: {count}")

        # By district (top 20)
        log("\nTop 20 Districts by neighborhood count:")
        district_counts = df.groupby(['province', 'district']).size().sort_values(ascending=False).head(20)
        for (province, district), count in district_counts.items():
            log(f"  {district}, {province}: {count}")

        return df

    def save_gazetteer(self):
        """Save the final gazetteer"""
        log("\n" + "=" * 80)
        log("SAVING GAZETTEER")
        log("=" * 80)

        df = pd.DataFrame(self.neighborhoods)

        # Select and order columns
        columns = ['neighborhood_name', 'neighborhood_clean', 'neighborhood_normalized',
                   'district', 'province', 'source', 'osm_type', 'lat', 'lon']
        df = df[[c for c in columns if c in df.columns]]

        # Sort by province, district, name
        df = df.sort_values(['province', 'district', 'neighborhood_clean'])

        # Save main gazetteer
        output_path = os.path.join(OUTPUT_DIR, 'earthquake_region_neighborhoods.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        log(f"Saved gazetteer to: {output_path}")
        log(f"Total entries: {len(df)}")

        # Save log
        save_log()
        log(f"Saved log to: {log_file}")

        return df


def main():
    collector = NeighborhoodCollector()

    # Collect from all sources
    collector.collect_all()

    # Clean and deduplicate
    collector.deduplicate_and_clean()

    # Validate against tweet data
    collector.validate_against_tweets()

    # Generate statistics
    collector.generate_statistics()

    # Save final gazetteer
    df = collector.save_gazetteer()

    log("\n" + "=" * 80)
    log("COLLECTION COMPLETE")
    log("=" * 80)

    return df


if __name__ == '__main__':
    main()
