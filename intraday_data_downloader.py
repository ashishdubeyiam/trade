import pandas as pd
import numpy as np # For NaN and potentially other calculations
import os
import logging
from datetime import datetime, timezone, date # Ensure date is imported for groupby
# Removed timedelta as it's not directly used in this version's fetch logic. OANDA count fetches most recent.

# Import from oanda_utils.py
try:
    from oanda_utils import get_oanda_api_client, fetch_oanda_candles
    OANDA_UTILS_AVAILABLE = True
except ImportError:
    # Fallback or mock if oanda_utils is not found, for structural checking
    OANDA_UTILS_AVAILABLE = False
    def get_oanda_api_client(): return None
    def fetch_oanda_candles(api_client, account_id, instrument, count, granularity, price_components): return None
    logging.getLogger(__name__).critical("Failed to import oanda_utils.py. OANDA functionality will not work.")


# --- Configuration ---
MASTER_M5_DATA_FILE = 'eur_usd_m5_master_data.csv' # Will now store OANDA data + calculated daily VWAP
LOG_FILE = 'intraday_data_downloader_oanda.log' # New log file name

INSTRUMENT = "EUR_USD"
GRANULARITY = "M5"
PRICE_COMPONENTS = "M" # Midpoint prices for OHLCV

INITIAL_FETCH_COUNT = 4500  # For initial data pull if master file is empty
INCREMENTAL_FETCH_COUNT = 4500 # For updates (fetch recent, then filter overlap)

# OANDA credentials will be sourced by oanda_utils from environment variables

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')

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

if not OANDA_UTILS_AVAILABLE:
    logger.warning("oanda_utils.py not available. Script will not function as intended.")


def calculate_daily_vwap(df):
    """
    Calculates Daily VWAP on an M5 DataFrame.
    Input DataFrame must have 'timestamp' (UTC datetime index),
    'high', 'low', 'close', and 'volume' columns.
    VWAP resets at the start of each UTC day.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame for VWAP calculation is empty or None.")
        return df

    if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        logger.error("DataFrame missing one or more required columns (high, low, close, volume) for VWAP calculation.")
        return df # Or raise error

    # Ensure timestamp is index and UTC for proper daily grouping
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be a DatetimeIndex for VWAP calculation.")
        return df
    if df.index.tz != timezone.utc:
        logger.warning(f"DataFrame index timezone is {df.index.tz}, expected UTC. Converting to UTC for VWAP calc.")
        try:
            df = df.tz_convert(timezone.utc)
        except TypeError: # If it was naive
            df = df.tz_localize(timezone.utc)


    logger.info(f"Calculating Daily VWAP for {len(df)} rows...")

    df_copy = df.copy() # Work on a copy
    df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
    df_copy['tp_x_volume'] = df_copy['typical_price'] * df_copy['volume']

    # Group by date (UTC day) and calculate cumulative sums for VWAP
    # Using .transform with lambda for groupby().cumsum() is efficient for assigning back
    # It handles alignment automatically.
    daily_groups = df_copy.groupby(df_copy.index.date)

    cumulative_tp_x_volume = daily_groups['tp_x_volume'].cumsum()
    cumulative_volume = daily_groups['volume'].cumsum()

    # Calculate VWAP, handle division by zero (where cumulative_volume is 0)
    df_copy['vwap'] = np.where(cumulative_volume != 0, cumulative_tp_x_volume / cumulative_volume, np.nan)

    # If VWAP is NaN due to zero cumulative volume, can fill with typical price of that bar or leave as NaN
    # For first bar of day where volume might be 0 or low, VWAP might be same as typical price if vol=0 then first trade.
    # If first bar volume is non-zero, VWAP = typical_price.
    # Let's fill NaN VWAPs (from zero cumulative volume) with the typical price of that bar,
    # as this is often how initial VWAP is considered.
    df_copy['vwap'].fillna(df_copy['typical_price'], inplace=True)

    logger.info("Daily VWAP calculation complete.")
    return df_copy[['vwap']] # Return only the VWAP column as a DataFrame to merge


def update_intraday_master_dataset():
    logger.info(f"--- Starting OANDA Intraday M5 Data Update for {INSTRUMENT} ---")
    logger.info(f"Master M5 data file: {MASTER_M5_DATA_FILE}")
    logger.warning("OANDA API has rate limits. Ensure this script is run appropriately.")

    if not OANDA_UTILS_AVAILABLE:
        logger.critical("OANDA utilities are not available. Cannot proceed.")
        return

    api_client = get_oanda_api_client() # Uses env vars via oanda_utils
    if not api_client:
        logger.critical("Failed to initialize OANDA API client (check credentials in env). Exiting.")
        return

    existing_df = pd.DataFrame()
    last_known_timestamp_utc = None
    fetch_count = INITIAL_FETCH_COUNT

    if os.path.exists(MASTER_M5_DATA_FILE):
        logger.info(f"Master file {MASTER_M5_DATA_FILE} exists. Reading existing data.")
        try:
            existing_df = pd.read_csv(MASTER_M5_DATA_FILE, index_col='timestamp', parse_dates=['timestamp'])
            if not existing_df.index.empty: # Check if index (timestamps) is empty
                # Ensure existing timestamps are UTC
                if existing_df.index.tz is None:
                    logger.warning("Loaded master data timestamps are naive. Localizing to UTC.")
                    existing_df = existing_df.tz_localize(timezone.utc)
                elif existing_df.index.tz != timezone.utc:
                    logger.warning(f"Loaded master data timestamps are {existing_df.index.tz}. Converting to UTC.")
                    existing_df = existing_df.tz_convert(timezone.utc)

                existing_df.sort_index(inplace=True)
                last_known_timestamp_utc = existing_df.index[-1]
                logger.info(f"Last recorded UTC timestamp in master file: {last_known_timestamp_utc.isoformat()}")
                fetch_count = INCREMENTAL_FETCH_COUNT # Use incremental count for updates
            else:
                logger.warning("Master file found but contains no valid timestamps or is empty after parsing. Will fetch initial data.")
        except Exception as e:
            logger.error(f"Error reading or parsing master file {MASTER_M5_DATA_FILE}: {e}. Will attempt to fetch initial data.", exc_info=True)
            existing_df = pd.DataFrame() # Ensure it's an empty DF
    else:
        logger.info(f"Master file {MASTER_M5_DATA_FILE} does not exist. Will perform initial data fetch.")

    logger.info(f"Fetching {fetch_count} recent {GRANULARITY} candles for {INSTRUMENT} from OANDA...")
    new_ohlcv_df = fetch_oanda_candles(
        api_client=api_client,
        instrument=INSTRUMENT,
        count=fetch_count,
        granularity=GRANULARITY,
        price_components=PRICE_COMPONENTS
    )

    if new_ohlcv_df is None or new_ohlcv_df.empty:
        logger.error("Failed to fetch new M5 OHLCV data from OANDA or no data returned. Aborting update.")
        if existing_df.empty: # If this was an initial fetch and it failed
             logger.info("No existing data and no new data. Output file will not be created/updated.")
        return

    # Set timestamp as index for VWAP calculation and merging
    new_ohlcv_df.set_index('timestamp', inplace=True)
    new_ohlcv_df.sort_index(inplace=True) # Ensure it's sorted before VWAP calc

    # Calculate Daily VWAP on the new data
    vwap_series_df = calculate_daily_vwap(new_ohlcv_df) # Returns DataFrame with 'vwap' column and same index

    # Merge VWAP into the new OHLCV data
    if vwap_series_df is not None and not vwap_series_df.empty:
        new_data_with_vwap_df = new_ohlcv_df.merge(vwap_series_df, left_index=True, right_index=True, how='left')
    else:
        logger.warning("VWAP calculation failed or returned empty. Adding NaN 'vwap' column.")
        new_data_with_vwap_df = new_ohlcv_df.copy()
        new_data_with_vwap_df['vwap'] = np.nan

    logger.info(f"New data fetched and VWAP calculated. Shape: {new_data_with_vwap_df.shape}")

    # Filter new data to be after the last known timestamp (if updating)
    if last_known_timestamp_utc:
        new_data_to_append = new_data_with_vwap_df[new_data_with_vwap_df.index > last_known_timestamp_utc].copy()
        logger.info(f"{len(new_data_to_append)} new records found after last known timestamp {last_known_timestamp_utc.isoformat()}.")
    else:
        new_data_to_append = new_data_with_vwap_df.copy()
        logger.info(f"This is an initial data load, all {len(new_data_to_append)} fetched records will be used.")

    if new_data_to_append.empty and last_known_timestamp_utc: # Check if it was an update and no new bars
        logger.info("No new unique M5 records to add to the master dataset.")
        # Save existing_df back if it was loaded and potentially re-indexed/tz-fixed (optional)
        # For now, only save if there's new data or it's an initial creation.
        if existing_df.empty and not os.path.exists(MASTER_M5_DATA_FILE): # Create empty file with schema if it's first run and no data
            pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap']).to_csv(MASTER_M5_DATA_FILE, index_label='timestamp')
            logger.info(f"Initialized empty master file with schema: {MASTER_M5_DATA_FILE}")
        return

    # Combine with existing data
    if not existing_df.empty:
        # Align columns before concat, ensure 'vwap' exists in existing_df or add it
        if 'vwap' not in existing_df.columns: existing_df['vwap'] = np.nan
        # Ensure column order is the same to avoid performance warning on concat, though not strictly necessary for correctness
        cols_order = new_data_to_append.columns.tolist()
        combined_df = pd.concat([existing_df[cols_order], new_data_to_append], ignore_index=False) # Keep index for now
    else:
        combined_df = new_data_to_append

    logger.info(f"Total records before final deduplication: {len(combined_df)}")

    # Deduplicate based on index (timestamp) and sort
    # Since index is unique timestamp, drop_duplicates on index is not needed if concat doesn't create them.
    # Sorting index is important.
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')] # Keep first (older if overlap)
    combined_df.sort_index(inplace=True)

    try:
        # Save with timestamp as index in the CSV
        combined_df.to_csv(MASTER_M5_DATA_FILE, index=True, index_label='timestamp')
        logger.info(f"Successfully updated M5 master dataset: {MASTER_M5_DATA_FILE}. Total records: {len(combined_df)}.")
        if not existing_df.empty:
            logger.info(f"Number of records effectively added or changed: {len(combined_df) - len(existing_df[existing_df.index.isin(combined_df.index)])}") # Complex to get exact added due to potential overlaps and deduplication
    except Exception as e:
        logger.error(f"Error writing updated M5 master file {MASTER_M5_DATA_FILE}: {e}", exc_info=True)

    logger.info(f"--- Intraday M5 Data Update Finished (OANDA) ---")


if __name__ == '__main__':
    # Setup basic logging for standalone run
    if not logger.handlers or len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler) : # Check if only a NullHandler might be present
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                            handlers=[logging.StreamHandler()])
        try:
            fh = logging.FileHandler(LOG_FILE)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
            logging.getLogger().addHandler(fh) # Add to root logger
        except Exception as e:
            logger.error(f"Could not add file handler for {LOG_FILE} in standalone mode: {e}")

    if not OANDA_UTILS_AVAILABLE:
        logger.critical("Exiting: oanda_utils.py is required but not found or failed to import.")
    else:
        update_intraday_master_dataset()
