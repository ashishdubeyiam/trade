import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import os
import re # For more sophisticated keyword matching, like whole words

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Assume calling script configures logger. For standalone testing, basicConfig can be used.

# --- Configuration ---
DEFAULT_HEADLINES_CSV = 'news_headlines.csv'
EXTREME_KEYWORDS_LIST = [
    "emergency meeting", "market crisis", "flash crash", "central bank intervention",
    "currency peg break", "geopolitical shock", "trade war escalation",
    "unexpected rate hike", "unexpected rate cut", "financial contagion",
    "sovereign default", "bank run", "black swan event" # Added a few more
]
HEADLINE_LOOKBACK_HOURS = 24 # Default lookback period

def load_news_headlines(csv_filepath=DEFAULT_HEADLINES_CSV):
    """
    Loads news headlines from a CSV file.

    Args:
        csv_filepath (str): Path to the CSV file containing news headlines.

    Returns:
        pandas.DataFrame: DataFrame with headlines, or None if loading fails.
                          Ensures 'published_utc' is parsed as datetime objects (UTC).
    """
    logger.info(f"Attempting to load news headlines from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"News headlines file not found: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        logger.info(f"Successfully loaded {len(df)} headlines from {csv_filepath}.")

        required_cols = ['published_utc', 'headline']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing one or more required columns ({required_cols}) in {csv_filepath}.")
            return None

        # Parse 'published_utc' to datetime objects. Assume they are ISO format strings.
        # pd.to_datetime will infer format, errors='coerce' will make unparseable dates NaT.
        df['published_utc'] = pd.to_datetime(df['published_utc'], errors='coerce', utc=True)

        # Drop rows where 'published_utc' could not be parsed
        original_len = len(df)
        df.dropna(subset=['published_utc'], inplace=True)
        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows due to unparseable 'published_utc' dates.")

        if df.empty and original_len > 0:
            logger.warning(f"DataFrame became empty after attempting to parse 'published_utc' dates from {csv_filepath}.")
            return None # Or return empty df based on desired strictness

        return df
    except FileNotFoundError: # Should be caught by os.path.exists
        logger.error(f"News headlines file not found (FileNotFoundError): {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"News headlines file is empty: {csv_filepath}")
        return pd.DataFrame(columns=required_cols + ['source_url', 'feed_source']) # Return empty with expected schema
    except Exception as e:
        logger.error(f"Error loading news headlines from {csv_filepath}: {e}", exc_info=True)
        return None

def check_for_extreme_news(headlines_df, keywords=EXTREME_KEYWORDS_LIST,
                           lookback_hours=HEADLINE_LOOKBACK_HOURS):
    """
    Checks recent headlines for the presence of extreme keywords.

    Args:
        headlines_df (pd.DataFrame): DataFrame loaded by load_news_headlines.
        keywords (list): List of extreme keywords/phrases to search for (case-insensitive).
        lookback_hours (int): How many hours back from current UTC time to check headlines.

    Returns:
        tuple: (bool, list_of_dicts).
               First element is True if extreme news found, False otherwise.
               Second element is a list of dicts, each containing info about a triggering headline
               (e.g., {'headline': str, 'published_utc': str, 'feed_source': str, 'matched_keyword': str}).
    """
    if headlines_df is None or headlines_df.empty:
        logger.info("Headlines DataFrame is empty or None. No extreme news to check.")
        return False, []

    if not keywords:
        logger.warning("Keyword list is empty. No keywords to check for.")
        return False, []

    # Ensure 'published_utc' is datetime like and UTC localized
    if not pd.api.types.is_datetime64_any_dtype(headlines_df['published_utc']):
        logger.error("'published_utc' column is not in datetime format. Cannot perform time window filter.")
        return False, []
    if headlines_df['published_utc'].dt.tz is None:
        logger.warning("'published_utc' column is not timezone-aware. Assuming UTC, but this could be inaccurate.")
        # Attempt to localize, though this assumes it was originally UTC if naive
        # headlines_df['published_utc'] = headlines_df['published_utc'].dt.tz_localize('UTC')


    now_utc = datetime.now(timezone.utc)
    start_time_utc = now_utc - timedelta(hours=lookback_hours)

    logger.debug(f"Checking for extreme news published after {start_time_utc.isoformat()} UTC.")

    # Filter for recent headlines
    # Ensure comparison is between timezone-aware datetimes
    recent_headlines_df = headlines_df[headlines_df['published_utc'] >= start_time_utc].copy() # Use .copy()

    if recent_headlines_df.empty:
        logger.info(f"No headlines found within the last {lookback_hours} hours.")
        return False, []

    logger.info(f"Checking {len(recent_headlines_df)} headlines from the last {lookback_hours} hours for keywords.")

    triggering_headlines_info = []
    found_keywords_set = set() # To log unique keywords found

    # Prepare keywords: lowercase and compile regex for whole word matching (optional but better)
    # Using \b for word boundaries to avoid matching substrings within words.
    compiled_keywords = []
    for k in keywords:
        # Escaping special regex characters in keyword, then adding word boundaries
        # And making it case insensitive via flags=re.IGNORECASE
        try:
            compiled_keywords.append(re.compile(r'\b' + re.escape(k.lower()) + r'\b', flags=re.IGNORECASE))
        except re.error as e:
            logger.error(f"Invalid regex from keyword '{k}': {e}. Skipping this keyword.")


    for index, row in recent_headlines_df.iterrows():
        headline_text = str(row.get('headline', '')) # Ensure it's a string
        headline_lower = headline_text.lower() # For simple check if needed, but regex handles case

        for i, keyword_regex in enumerate(compiled_keywords):
            if keyword_regex.search(headline_text): # Use search for regex
                original_keyword = keywords[i] # Get the original casing for logging
                info = {
                    'headline': headline_text,
                    'published_utc': row['published_utc'].isoformat(),
                    'feed_source': row.get('feed_source', 'N/A'),
                    'matched_keyword': original_keyword
                }
                triggering_headlines_info.append(info)
                found_keywords_set.add(original_keyword)
                # Log first occurrence, then break from inner loop to not report multiple keywords for same headline (optional)
                # Or collect all keywords per headline (current behavior is one entry per keyword match)
                logger.warning(f"EXTREME NEWS DETECTED! Keyword: '{original_keyword}' found in headline: \"{headline_text}\" (Source: {info['feed_source']}, Published: {info['published_utc']})")
                # To find only one keyword per headline and then move to next headline:
                # break

    if triggering_headlines_info:
        logger.warning(f"Found {len(triggering_headlines_info)} instances of extreme keywords in recent news. Matched keywords: {list(found_keywords_set)}")
        return True, triggering_headlines_info
    else:
        logger.info(f"No extreme keywords found in {len(recent_headlines_df)} recent headlines.")
        return False, []

if __name__ == '__main__':
    # --- Basic Test for the utility module ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("--- Testing news_processor_utils.py ---")

    # Create a dummy news_headlines.csv for testing
    now = datetime.now(timezone.utc)
    dummy_news_data = {
        'fetched_utc': [(now - timedelta(minutes=i*60)).isoformat() for i in range(5)],
        'published_utc': [
            (now - timedelta(hours=1)).isoformat(),  # Recent, should be checked
            (now - timedelta(hours=5)).isoformat(),  # Recent, should be checked
            (now - timedelta(hours=30)).isoformat(), # Too old
            (now - timedelta(hours=2)).isoformat(),  # Recent, should be checked
            (now - timedelta(hours=48)).isoformat(), # Too old
        ],
        'headline': [
            "Major bank announces emergency meeting amidst market turmoil.", # Match: emergency meeting
            "Analysts discuss market crisis scenarios for coming week.",    # Match: market crisis
            "Old news about a minor currency fluctuation.",
            "Geopolitical shock rocks global markets, central bank intervention expected.", # Match: geopolitical shock, central bank intervention
            "Another old headline not relevant."
        ],
        'source_url': ['http://news.com/1', 'http://news.com/2', 'http://news.com/3', 'http://news.com/4', 'http://news.com/5'],
        'feed_source': ['Feed A', 'Feed B', 'Feed A', 'Feed C', 'Feed B']
    }
    dummy_headlines_df = pd.DataFrame(dummy_news_data)
    dummy_csv_path = 'dummy_test_headlines.csv'
    dummy_headlines_df.to_csv(dummy_csv_path, index=False)
    logger.info(f"Created dummy headlines file: {dummy_csv_path}")

    # Test load_news_headlines
    loaded_df = load_news_headlines(csv_filepath=dummy_csv_path)
    if loaded_df is not None:
        logger.info(f"Loaded dummy headlines. Shape: {loaded_df.shape}")
        logger.info("Checking data types of 'published_utc':")
        logger.info(str(loaded_df['published_utc'].dtype)) # Should be datetime64[ns, UTC]
        # print(loaded_df.head())

        # Test check_for_extreme_news
        logger.info(f"\nTesting check_for_extreme_news with lookback_hours={HEADLINE_LOOKBACK_HOURS}...")

        # Using default keywords from the module
        extreme_news_found, triggering_headlines = check_for_extreme_news(
            loaded_df,
            keywords=EXTREME_KEYWORDS_LIST, # Use the global list for testing consistency
            lookback_hours=HEADLINE_LOOKBACK_HOURS
        )

        logger.info(f"Extreme news found: {extreme_news_found}")
        if extreme_news_found:
            logger.info("Triggering headlines:")
            for item in triggering_headlines:
                logger.info(f"  - Keyword: '{item['matched_keyword']}', Headline: \"{item['headline']}\", Source: {item['feed_source']}, Published: {item['published_utc']}")

        # Test with a shorter lookback to exclude some headlines
        logger.info(f"\nTesting check_for_extreme_news with lookback_hours=3...")
        extreme_news_found_short, triggering_headlines_short = check_for_extreme_news(
            loaded_df, keywords=EXTREME_KEYWORDS_LIST, lookback_hours=3
        )
        logger.info(f"Extreme news found (3hr lookback): {extreme_news_found_short}")
        if extreme_news_found_short:
            logger.info("Triggering headlines (3hr lookback):")
            for item in triggering_headlines_short:
                 logger.info(f"  - Keyword: '{item['matched_keyword']}', Headline: \"{item['headline']}\", Source: {item['feed_source']}, Published: {item['published_utc']}")

    else:
        logger.error("Failed to load dummy headlines data for testing.")

    # Clean up dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
        logger.info(f"Removed dummy headlines file: {dummy_csv_path}")
