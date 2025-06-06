import feedparser
import pandas as pd
import logging
from datetime import datetime, timezone
import os
import pytz # For robust timezone handling, ensure it's installed if not standard

# --- Configuration ---
RSS_FEED_URLS = {
    "Investing.com Forex": "https://www.investing.com/rss/news_285.rss",
    "FXStreet News": "https://www.fxstreet.com/rss/news",
    "DailyFX News": "https://www.dailyfx.com/feeds/latest",
    # Example of a feed that might have different date formats or issues:
    # "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"
    # Note: Reuters feed above was problematic in testing, using more stable ones.
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex" # General finance
}
OUTPUT_CSV_FILE = 'news_headlines.csv'
LOG_FILE = 'news_headlines_fetcher.log'

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to console and file: {LOG_FILE}")
except Exception as e:
    logger.error(f"Failed to set up file logger for {LOG_FILE}: {e}", exc_info=True)
    logger.info("Logging to console only.")

def parse_entry_published_date(entry):
    """
    Attempts to parse the published date from a feedparser entry.
    Converts the date to UTC.
    Returns a timezone-aware datetime object in UTC or None if parsing fails.
    """
    parsed_time_struct = None
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        parsed_time_struct = entry.published_parsed
    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed: # Fallback
        parsed_time_struct = entry.updated_parsed

    if parsed_time_struct:
        try:
            dt = datetime(*parsed_time_struct[:6]) # Create naive datetime
            # feedparser time_structs are supposed to be UTC if no TZ info, but let's be explicit.
            # If the feed *does* provide timezone offset via the struct, datetime will be offset-naive.
            # We assume feedparser normalizes to UTC if possible, or it's given in UTC.
            # For robustness, if we know a feed isn't UTC and lacks TZ, manual offset would be needed here.
            # However, standard practice for feeds is UTC or TZ-aware.
            # Let's assume it's UTC if no tzinfo, or convert if tzinfo is present.

            # Create a timezone-aware datetime object, assuming UTC if naive.
            # The time.struct_time from feedparser is generally UTC.
            dt_aware = datetime(
                parsed_time_struct.tm_year,
                parsed_time_struct.tm_mon,
                parsed_time_struct.tm_mday,
                parsed_time_struct.tm_hour,
                parsed_time_struct.tm_min,
                parsed_time_struct.tm_sec,
                tzinfo=timezone.utc # Assume parsed_time_struct is UTC as per feedparser docs for feeds lacking explicit offset
            )
            return dt_aware
        except Exception as e:
            logger.warning(f"Could not parse date from time.struct_time: {parsed_time_struct}. Error: {e}", exc_info=True)
            return None
    logger.debug(f"No 'published_parsed' or 'updated_parsed' field for entry: {entry.get('title', 'N/A')}")
    return None


def fetch_and_store_headlines():
    logger.info("--- Starting News Headlines Fetcher ---")
    all_headlines = []
    fetched_script_utc = datetime.now(timezone.utc)

    for feed_name, feed_url in RSS_FEED_URLS.items():
        logger.info(f"Fetching headlines from: {feed_name} ({feed_url})")
        try:
            feed_data = feedparser.parse(feed_url)

            if feed_data.bozo: # Check for malformed feed
                bozo_exception = feed_data.get('bozo_exception', 'Unknown error')
                logger.warning(f"Feed {feed_name} may be malformed. Bozo exception: {bozo_exception}")
                # Continue to try parsing entries if any exist

            if not feed_data.entries:
                logger.warning(f"No entries found in feed: {feed_name}")
                continue

            logger.info(f"Found {len(feed_data.entries)} entries in {feed_name}.")

            for entry in feed_data.entries:
                headline = entry.get('title', 'N/A').strip()
                source_url = entry.get('link', 'N/A').strip()

                published_dt_utc = parse_entry_published_date(entry)

                if published_dt_utc is None:
                    logger.warning(f"Could not determine published date for headline: '{headline[:50]}...' from {feed_name}. Skipping.")
                    # Or assign fetched_script_utc as a fallback, but this might be misleading
                    # published_dt_utc = fetched_script_utc
                    continue


                all_headlines.append({
                    'fetched_utc': fetched_script_utc.isoformat(),
                    'published_utc': published_dt_utc.isoformat(), # Store as ISO 8601 string
                    'headline': headline,
                    'source_url': source_url,
                    'feed_source': feed_name
                })
        except Exception as e:
            logger.error(f"Error fetching or parsing feed {feed_name} ({feed_url}): {e}", exc_info=True)
            continue # Continue to the next feed

    if not all_headlines:
        logger.info("No headlines fetched from any feed.")
        return

    new_headlines_df = pd.DataFrame(all_headlines)
    logger.info(f"Total headlines fetched across all feeds: {len(new_headlines_df)}")

    # Handling Duplicates and Appending
    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            logger.info(f"Loading existing headlines from {OUTPUT_CSV_FILE} for deduplication.")
            existing_headlines_df = pd.read_csv(OUTPUT_CSV_FILE)
            combined_df = pd.concat([existing_headlines_df, new_headlines_df], ignore_index=True)
            logger.info(f"Combined headlines count (before deduplication): {len(combined_df)}")
        except Exception as e:
            logger.error(f"Error reading existing CSV {OUTPUT_CSV_FILE}: {e}. Will overwrite with new headlines.", exc_info=True)
            combined_df = new_headlines_df # Fallback to only new headlines
    else:
        logger.info("No existing headline CSV found. Creating new file.")
        combined_df = new_headlines_df

    # Deduplication: Prefer 'source_url' if available and unique, else 'headline' and 'published_utc'
    # Using source_url is generally more robust if available and not just a generic feed link
    if 'source_url' in combined_df.columns:
        # Normalize placeholder URLs before deduplication if any
        combined_df['source_url'] = combined_df['source_url'].replace(['N/A', '', None], pd.NA)
        # Deduplicate where source_url is valid
        df_with_valid_url = combined_df.dropna(subset=['source_url'])
        df_with_invalid_url = combined_df[combined_df['source_url'].isna()]

        df_with_valid_url.drop_duplicates(subset=['source_url'], keep='first', inplace=True)
        # For items with no valid source_url, deduplicate by headline and published time
        if not df_with_invalid_url.empty:
            df_with_invalid_url.drop_duplicates(subset=['headline', 'published_utc'], keep='first', inplace=True)
            final_df = pd.concat([df_with_valid_url, df_with_invalid_url], ignore_index=True)
        else:
            final_df = df_with_valid_url
    else: # Fallback if source_url column doesn't exist for some reason
        final_df = combined_df.drop_duplicates(subset=['headline', 'published_utc'], keep='first')

    # Sort data, typically by published date descending
    final_df.sort_values(by=['published_utc', 'fetched_utc'], ascending=[False, False], inplace=True)

    try:
        final_df.to_csv(OUTPUT_CSV_FILE, index=False)
        logger.info(f"Successfully saved {len(final_df)} unique headlines to {OUTPUT_CSV_FILE}")
        if 'existing_headlines_df' in locals():
             logger.info(f"Number of new headlines added: {len(final_df) - len(existing_headlines_df)}")
        else:
             logger.info(f"Total headlines in new file: {len(final_df)}")
    except Exception as e:
        logger.error(f"Error saving data to CSV {OUTPUT_CSV_FILE}: {e}", exc_info=True)

    logger.info("--- News Headlines Fetcher Finished ---")

if __name__ == '__main__':
    fetch_and_store_headlines()
