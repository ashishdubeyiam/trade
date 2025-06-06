import pandas as pd
import os
import logging
from datetime import datetime, timezone, timedelta # Added timedelta
from alpha_vantage_utils import (
    fetch_alpha_vantage_intraday_ohlcv,
    fetch_alpha_vantage_intraday_vwap,
    DEFAULT_AV_API_KEY # To use the same default logic for API key
)
import pytz # For timezone utilities, though utils handle primary conversion

# --- Configuration ---
MASTER_M5_DATA_FILE = 'eur_usd_m5_master_data.csv'
LOG_FILE = 'intraday_data_downloader.log'

FROM_SYMBOL = 'EUR'
TO_SYMBOL = 'USD'
INTERVAL = '5min'

# API Key from environment or default from utils
ALPHA_VANTAGE_API_KEY = os.environ.get('AV_API_KEY', DEFAULT_AV_API_KEY)

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


def update_intraday_master_dataset():
    logger.info(f"--- Starting Intraday M5 Data Update for {FROM_SYMBOL}/{TO_SYMBOL} ---")
    logger.info(f"Master M5 data file: {MASTER_M5_DATA_FILE}")
    logger.warning("Alpha Vantage API has limits (e.g., 5 calls/min, 500/day for free keys). "
                   "Fetching 'full' intraday data can be intensive if run too often or for many symbols.")

    if ALPHA_VANTAGE_API_KEY == 'demo':
        logger.warning("Using DEMO Alpha Vantage API key. Data may be limited or inconsistent.")
    elif not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_API_KEY_PLACEHOLDER': # Example placeholder check
        logger.error("Alpha Vantage API key is not configured or is a placeholder. Cannot fetch data.")
        return

    existing_master_df = None
    last_timestamp_utc = None
    output_size_ohlcv = 'compact' # Default for updates
    # output_size_vwap = 'compact' # VWAP doesn't use outputsize, it aligns with intraday series

    if os.path.exists(MASTER_M5_DATA_FILE):
        logger.info(f"Master file {MASTER_M5_DATA_FILE} exists. Reading existing data.")
        try:
            existing_master_df = pd.read_csv(MASTER_M5_DATA_FILE, parse_dates=['timestamp'])
            if not existing_master_df.empty and 'timestamp' in existing_master_df.columns:
                # Ensure timestamp is UTC after loading
                if existing_master_df['timestamp'].dt.tz is None:
                    logger.warning("Loaded master data timestamps are naive. Assuming UTC.")
                    existing_master_df['timestamp'] = existing_master_df['timestamp'].dt.tz_localize('UTC')
                else:
                    existing_master_df['timestamp'] = existing_master_df['timestamp'].dt.tz_convert('UTC')

                existing_master_df.sort_values('timestamp', inplace=True)
                last_timestamp_utc = existing_master_df['timestamp'].iloc[-1]
                logger.info(f"Last recorded UTC timestamp in master file: {last_timestamp_utc.isoformat()}")

                # Determine if 'full' might be needed for OHLCV
                # If last timestamp is older than ~1 day (compact gives ~100 of 5min bars = ~8 hours of active trading)
                # This is a rough check. 'compact' for intraday is often just a few days.
                # AlphaVantage 'full' for intraday is typically a few weeks/months.
                if last_timestamp_utc < (datetime.now(timezone.utc) - timedelta(days=2)): # If data is older than 2 days
                    logger.info("Last timestamp is older than 2 days. Will attempt to fetch 'full' OHLCV history to catch up.")
                    output_size_ohlcv = 'full'
            else:
                logger.warning("Master file is empty or 'timestamp' column missing. Will fetch 'full' history.")
                output_size_ohlcv = 'full'
                existing_master_df = pd.DataFrame() # Ensure it's an empty DF for concat
        except Exception as e:
            logger.error(f"Error reading master file {MASTER_M5_DATA_FILE}: {e}. Will attempt to fetch 'full' history.", exc_info=True)
            output_size_ohlcv = 'full'
            existing_master_df = pd.DataFrame()
    else:
        logger.info(f"Master file {MASTER_M5_DATA_FILE} does not exist. Will fetch 'full' history.")
        output_size_ohlcv = 'full'
        existing_master_df = pd.DataFrame()

    # Fetch new data
    logger.info(f"Fetching M5 OHLCV data for {FROM_SYMBOL}/{TO_SYMBOL} (outputsize={output_size_ohlcv})...")
    ohlcv_df = fetch_alpha_vantage_intraday_ohlcv(
        api_key=ALPHA_VANTAGE_API_KEY,
        from_symbol=FROM_SYMBOL,
        to_symbol=TO_SYMBOL,
        interval=INTERVAL,
        outputsize=output_size_ohlcv
    )

    if ohlcv_df is None or ohlcv_df.empty:
        logger.error("Failed to fetch M5 OHLCV data. Aborting update.")
        return

    logger.info(f"Fetching M5 VWAP data for {FROM_SYMBOL}/{TO_SYMBOL} (interval={INTERVAL})...")
    # VWAP data length depends on underlying series. Fetching with same interval.
    vwap_df = fetch_alpha_vantage_intraday_vwap(
        api_key=ALPHA_VANTAGE_API_KEY,
        from_symbol=FROM_SYMBOL,
        to_symbol=TO_SYMBOL,
        interval=INTERVAL
    )

    if vwap_df is None or vwap_df.empty:
        logger.warning("Failed to fetch M5 VWAP data. OHLCV data will be saved without VWAP if new.")
        # Create an empty VWAP DataFrame with correct columns to allow merge to proceed if needed
        merged_df = ohlcv_df.copy()
        merged_df['vwap'] = pd.NA # Or np.nan
    else:
        # Merge OHLCV and VWAP data
        logger.info("Merging OHLCV and VWAP data...")
        # Using outer merge to keep all timestamps; then we can see where data might be missing for one source.
        merged_df = pd.merge(ohlcv_df, vwap_df, on='timestamp', how='outer')

        # Log merge discrepancies (e.g., timestamps present in one but not the other)
        ohlcv_only_count = merged_df['open'].notna().sum() - merged_df['vwap'].notna().sum()
        vwap_only_count = merged_df['vwap'].notna().sum() - merged_df['open'].notna().sum()
        if ohlcv_only_count > 0 : logger.warning(f"{ohlcv_only_count} timestamps have OHLCV but missing VWAP data after merge.")
        if vwap_only_count > 0 : logger.warning(f"{vwap_only_count} timestamps have VWAP but missing OHLCV data after merge (should be rare).")

        # For our purpose, OHLCV is primary. So, we might prefer a left merge if VWAP points without OHLCV are not useful.
        # Let's stick to outer for now to see data, but typically a left merge on OHLCV would be fine.
        # If we did left merge: merged_df = pd.merge(ohlcv_df, vwap_df, on='timestamp', how='left')
        # This would ensure all OHLCV bars are kept, and VWAP is added if available.

    # Filter new_merged_data for records newer than last_timestamp (if applicable)
    if last_timestamp_utc:
        new_records_df = merged_df[merged_df['timestamp'] > last_timestamp_utc].copy()
        logger.info(f"Found {len(new_records_df)} new M5 records (OHLCV+VWAP) since {last_timestamp_utc.isoformat()}.")
    else:
        new_records_df = merged_df.copy()
        logger.info(f"No existing master data, all {len(new_records_df)} fetched M5 records are considered new.")

    if new_records_df.empty:
        logger.info("No new M5 records to add.")
    else:
        # Append new data
        if not existing_master_df.empty:
            combined_df = pd.concat([existing_master_df, new_records_df], ignore_index=True)
        else:
            combined_df = new_records_df

        logger.info(f"Total records before final deduplication: {len(combined_df)}")

        # Ensure 'timestamp' is primary key, sort, and deduplicate
        combined_df.sort_values('timestamp', inplace=True)
        combined_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        try:
            combined_df.to_csv(MASTER_M5_DATA_FILE, index=False)
            logger.info(f"Successfully updated M5 master dataset: {MASTER_M5_DATA_FILE}. Total records: {len(combined_df)}.")
            if not existing_master_df.empty:
                 logger.info(f"Number of new records effectively added: {len(combined_df) - len(existing_master_df)}")
        except Exception as e:
            logger.error(f"Error writing updated M5 master file {MASTER_M5_DATA_FILE}: {e}", exc_info=True)

    logger.info(f"--- Intraday M5 Data Update Finished ---")

if __name__ == '__main__':
    update_intraday_master_dataset()
