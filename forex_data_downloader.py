import pandas as pd
import os
import logging
from datetime import datetime
from alpha_vantage_utils import fetch_alpha_vantage_daily_data # Assuming alpha_vantage_utils.py is in the same directory or PYTHONPATH

# --- Configuration ---
MASTER_DATA_FILE = 'eur_usd_master_data.csv'
FROM_SYMBOL = "EUR"
TO_SYMBOL = "USD"
LOG_FILE = 'forex_data_downloader.log'

# Attempt to get API key from environment variable, otherwise default to 'demo'
# This key will be passed to fetch_alpha_vantage_daily_data if that function doesn't directly use os.environ
ALPHA_VANTAGE_API_KEY = os.environ.get('AV_API_KEY', 'demo')

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

def update_master_dataset():
    """
    Updates the master EUR/USD dataset by fetching new data from Alpha Vantage.
    If the master file doesn't exist, it creates it with full historical data.
    """
    logger.info(f"Starting update process for master dataset: {MASTER_DATA_FILE}")

    if ALPHA_VANTAGE_API_KEY == 'demo':
        logger.warning("Using DEMO Alpha Vantage API key. Data may be limited or inconsistent.")
    elif ALPHA_VANTAGE_API_KEY == 'YOUR_API_KEY_PLACEHOLDER' or not ALPHA_VANTAGE_API_KEY:
        logger.error("Alpha Vantage API key is not configured or is a placeholder. Cannot fetch data.")
        return

    master_df = None
    last_timestamp = None

    if os.path.exists(MASTER_DATA_FILE):
        logger.info(f"Master file {MASTER_DATA_FILE} exists. Reading existing data.")
        try:
            master_df = pd.read_csv(MASTER_DATA_FILE, parse_dates=['timestamp'])
            if not master_df.empty and 'timestamp' in master_df.columns:
                master_df = master_df.sort_values('timestamp').reset_index(drop=True)
                last_timestamp = master_df['timestamp'].iloc[-1]
                logger.info(f"Last recorded timestamp in master file: {last_timestamp.strftime('%Y-%m-%d')}")
            else:
                logger.warning("Master file is empty or does not have a 'timestamp' column. Will attempt to fetch full history.")
                master_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Ensure schema
        except Exception as e:
            logger.error(f"Error reading master file {MASTER_DATA_FILE}: {e}. Will attempt to fetch full history as a new file.", exc_info=True)
            master_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Ensure schema
    else:
        logger.info(f"Master file {MASTER_DATA_FILE} does not exist. Will fetch full history.")
        master_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Ensure schema for concatenation

    # Fetch new data from Alpha Vantage
    # Using 'full' outputsize to get all available data. We'll filter it later.
    logger.info("Fetching new data from Alpha Vantage (outputsize=full)...")
    new_data_df = fetch_alpha_vantage_daily_data(
        api_key=ALPHA_VANTAGE_API_KEY,
        from_symbol=FROM_SYMBOL,
        to_symbol=TO_SYMBOL,
        outputsize='full'
    )

    if new_data_df is None or new_data_df.empty:
        logger.error("Failed to fetch new data from Alpha Vantage. No updates will be made.")
        return

    # Ensure new_data_df has the 'timestamp' column and it's datetime
    if 'timestamp' not in new_data_df.columns or not pd.api.types.is_datetime64_any_dtype(new_data_df['timestamp']):
        logger.error("Fetched data is missing 'timestamp' column or it's not datetime. Cannot proceed.")
        return

    new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp']) # Ensure datetime type

    # Filter new_data_df for records newer than last_timestamp
    if last_timestamp:
        original_new_data_count = len(new_data_df)
        new_data_df = new_data_df[new_data_df['timestamp'] > last_timestamp]
        logger.info(f"Filtered new data: {len(new_data_df)} records are newer than {last_timestamp.strftime('%Y-%m-%d')}. (Original fetched: {original_new_data_count})")

    if not new_data_df.empty:
        # Append new data to the master DataFrame
        combined_df = pd.concat([master_df, new_data_df], ignore_index=True)

        # Ensure 'timestamp' is primary key for deduplication
        if 'timestamp' not in combined_df.columns:
             logger.error("Timestamp column is missing after concatenation. This should not happen.")
             return

        # Remove duplicates, keeping the first occurrence (preferring existing data if timestamps overlap exactly)
        # and sort by timestamp again
        combined_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        combined_df.sort_values('timestamp', inplace=True)
        combined_df.reset_index(drop=True, inplace=True) # Reset index after sort

        try:
            combined_df.to_csv(MASTER_DATA_FILE, index=False)
            logger.info(f"Successfully updated master dataset: {MASTER_DATA_FILE}. Total records: {len(combined_df)}. New records added: {len(new_data_df) if last_timestamp else len(combined_df)}.")
        except Exception as e:
            logger.error(f"Error writing updated master file {MASTER_DATA_FILE}: {e}", exc_info=True)
    elif master_df is not None and not master_df.empty and os.path.exists(MASTER_DATA_FILE): # If master_df was loaded and no new data to add
        logger.info("No new unique records to add to the master dataset.")
        # Optionally re-save master_df if any in-memory cleaning happened (like initial sort or schema fix)
        # For now, only save if there were actual new records or if it's the first time.
        if not last_timestamp: # If it was an empty file fill scenario
            try:
                master_df.to_csv(MASTER_DATA_FILE, index=False) # Save the schema if it was an empty file
                logger.info(f"Master file was empty or non-existent, initialized with schema (or fetched data if any): {MASTER_DATA_FILE}.")
            except Exception as e:
                logger.error(f"Error writing initial master file {MASTER_DATA_FILE}: {e}", exc_info=True)

    else: # No new data fetched, and master file didn't exist or was empty
        logger.info("No new data fetched and no existing master data to update.")
        if not os.path.exists(MASTER_DATA_FILE) and master_df is not None: # If master_df is just the empty schema
             try:
                master_df.to_csv(MASTER_DATA_FILE, index=False)
                logger.info(f"Initialized empty master dataset with schema: {MASTER_DATA_FILE}.")
             except Exception as e:
                logger.error(f"Error writing empty master file {MASTER_DATA_FILE}: {e}", exc_info=True)


if __name__ == '__main__':
    logger.info("Forex Data Downloader for Master Dataset - Started")
    update_master_dataset()
    logger.info("Forex Data Downloader for Master Dataset - Finished")
