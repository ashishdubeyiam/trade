import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timezone

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Assume calling script configures logger. For standalone testing, basicConfig can be used.

# --- Configuration ---
DEFAULT_INPUT_M5_CSV = 'eur_usd_m5_master_data.csv' # Expects features like ATR already
DEFAULT_OUTPUT_LABELED_M5_CSV = 'eur_usd_m5_labeled_data.csv'
LOG_FILE = 'intraday_data_labeler.log'

TP_ATR_MULTIPLIER = 1.5
SL_ATR_MULTIPLIER = 1.0
MAX_HOLDING_BARS = 12 # e.g., 12 * 5 min = 1 hour lookahead
TARGET_COLUMN_NAME = 'up_move_success' # Binary: 1 for success (TP hit before SL/timeout), 0 for failure/timeout
ATR_COL_NAME = 'atr_14' # Column name for ATR from feature engineer

def load_featured_m5_data(csv_filepath=DEFAULT_INPUT_M5_CSV):
    """
    Loads M5 data which should already have features (especially ATR).
    Sets 'timestamp' as a UTC datetime index.
    Validates required columns and converts them to numeric.
    """
    logger.info(f"Attempting to load M5 data with features from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"M5 data file not found: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            logger.warning(f"M5 data file is empty: {csv_filepath}")
            return pd.DataFrame()

        logger.info(f"Successfully loaded {len(df)} records from {csv_filepath}.")

        if 'timestamp' not in df.columns:
            logger.error("M5 data CSV must contain a 'timestamp' column.")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        required_cols = ['high', 'low', 'close', ATR_COL_NAME]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error(f"M5 data is missing required columns for labeling: {missing_cols}")
            return None

        for col in required_cols + ['open', 'volume', 'vwap']: # Convert common cols if they exist
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows if essential columns became NaN after numeric conversion
        # df.dropna(subset=required_cols, inplace=True) # Be careful if ATR can legitimately be NaN for first few rows
        # logger.info(f"Shape after numeric conversion: {df.shape}")

        return df
    except FileNotFoundError:
        logger.error(f"M5 data file not found (FileNotFoundError): {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"M5 data file is empty (EmptyDataError): {csv_filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading M5 data from {csv_filepath}: {e}", exc_info=True)
        return None

def generate_labels(df, tp_atr_mult=TP_ATR_MULTIPLIER, sl_atr_mult=SL_ATR_MULTIPLIER,
                    max_holding_bars=MAX_HOLDING_BARS, target_col_name=TARGET_COLUMN_NAME,
                    atr_col=ATR_COL_NAME):
    """
    Generates target labels for predicting an upward price move.
    Label '1' (success) if TP is hit before SL within max_holding_bars.
    Label '0' (failure) if SL is hit first or if neither TP nor SL is hit (timeout).
    Labels np.nan if ATR is invalid or not enough future bars.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Cannot generate labels.")
        return df

    if not all(col in df.columns for col in ['high', 'low', 'close', atr_col]):
        logger.error(f"DataFrame is missing one or more required columns: 'high', 'low', 'close', '{atr_col}'.")
        return df # Or raise error

    n_rows = len(df)
    labels = pd.Series(np.nan, index=df.index) # Initialize labels series with NaNs

    logger.info(f"Generating labels for {n_rows} rows with max_holding_bars={max_holding_bars}, TP_mult={tp_atr_mult}, SL_mult={sl_atr_mult}.")

    # Using .values for faster access in the loop for price/ATR data
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    atr_values = df[atr_col].values

    for i in range(n_rows - max_holding_bars): # Iterate up to where there are enough future bars
        current_atr = atr_values[i]

        if pd.isna(current_atr) or current_atr <= 1e-7: # Check for NaN or effectively zero ATR
            # labels.iloc[i] = np.nan # Already NaN, or use -1 if preferred for "cannot_label"
            logger.debug(f"Skipping label for index {df.index[i]} due to invalid ATR: {current_atr}")
            continue

        entry_price = close_prices[i]
        tp_level = entry_price + (tp_atr_mult * current_atr)
        sl_level = entry_price - (sl_atr_mult * current_atr)

        outcome_determined = False
        current_label = 0 # Default to failure/timeout

        for j in range(1, max_holding_bars + 1):
            future_bar_idx = i + j
            future_high = high_prices[future_bar_idx]
            future_low = low_prices[future_bar_idx]

            # Check SL first: if a bar hits both SL and TP, SL is usually considered first
            if future_low <= sl_level:
                current_label = 0 # Failure
                outcome_determined = True
                logger.debug(f"SL hit for entry at {df.index[i]} (Entry: {entry_price:.5f}, SL: {sl_level:.5f}) on bar {df.index[future_bar_idx]} (Low: {future_low:.5f})")
                break

            if future_high >= tp_level:
                current_label = 1 # Success
                outcome_determined = True
                logger.debug(f"TP hit for entry at {df.index[i]} (Entry: {entry_price:.5f}, TP: {tp_level:.5f}) on bar {df.index[future_bar_idx]} (High: {future_high:.5f})")
                break

        # If loop completes without TP or SL hit, it's a timeout (label remains 0)
        if not outcome_determined:
            logger.debug(f"Trade timed out for entry at {df.index[i]} (Entry: {entry_price:.5f}, TP: {tp_level:.5f}, SL: {sl_level:.5f})")
            # current_label is already 0

        labels.iloc[i] = current_label

    df[target_col_name] = labels

    # Log summary of labels
    if target_col_name in df:
        label_counts = df[target_col_name].value_counts(dropna=False)
        logger.info(f"Label generation complete. Label distribution:\n{label_counts.to_string()}")

    return df

def main():
    logger.info("--- Starting Intraday M5 Data Labeling Process ---")

    # Load data with features
    featured_df = load_featured_m5_data(csv_filepath=DEFAULT_INPUT_M5_CSV)

    if featured_df is None or featured_df.empty:
        logger.error(f"Failed to load featured M5 data from {DEFAULT_INPUT_M5_CSV} or it's empty. Exiting.")
        return

    # Generate labels
    labeled_df = generate_labels(featured_df) # Uses default params from config section

    if labeled_df is None or labeled_df.empty: # Should not happen if featured_df was valid
        logger.error("Label generation resulted in an empty or None DataFrame. Exiting.")
        return

    # Handle rows that couldn't be labeled (target is NaN)
    # Option: drop them, or keep them (current generate_labels assigns NaN for unlabelable rows)
    rows_before_dropna = len(labeled_df)
    labeled_df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)
    rows_after_dropna = len(labeled_df)
    logger.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows where target label was NaN (due to insufficient future data or invalid ATR).")

    if labeled_df.empty:
        logger.warning("No rows remaining after dropping NaN labels. Output file will not be saved or will be empty.")

    # Save labeled data
    try:
        labeled_df.to_csv(DEFAULT_OUTPUT_LABELED_M5_CSV, index=True) # Save with timestamp index
        logger.info(f"Successfully saved labeled M5 data ({len(labeled_df)} rows) to {DEFAULT_OUTPUT_LABELED_M5_CSV}")
    except Exception as e:
        logger.error(f"Error saving labeled M5 data to {DEFAULT_OUTPUT_LABELED_M5_CSV}: {e}", exc_info=True)

    logger.info("--- Intraday M5 Data Labeling Process Finished ---")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                        handlers=[logging.StreamHandler()])
    # Add file handler for standalone run log
    try:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
        logging.getLogger().addHandler(fh) # Add to root logger
    except Exception as e:
        logger.error(f"Could not add file handler in standalone mode: {e}")

    # Create a dummy input CSV for testing (eur_usd_m5_master_data.csv)
    # This should have features, especially ATR_COL_NAME
    num_rows_test = MAX_HOLDING_BARS + 50 # Enough for some labels
    base_time = datetime.now(timezone.utc) - timedelta(minutes=5*num_rows_test)
    test_data = {
        'timestamp': pd.date_range(start=base_time, periods=num_rows_test, freq='5min', tz='UTC'),
        'open': np.random.rand(num_rows_test) * 10 + 1.0500,
    }
    test_df = pd.DataFrame(test_data)
    test_df['high'] = test_df['open'] + np.random.rand(num_rows_test) * 0.0010
    test_df['low'] = test_df['open'] - np.random.rand(num_rows_test) * 0.0010
    test_df['close'] = (test_df['open'] + test_df['high'] + test_df['low']) / 3
    test_df[ATR_COL_NAME] = np.random.rand(num_rows_test) * 0.0005 + 0.0001 # Dummy ATR values
    # Add other columns expected by load_featured_m5_data if any strict checks
    test_df['volume'] = 0
    test_df['vwap'] = test_df['close']


    dummy_input_path = DEFAULT_INPUT_M5_CSV
    test_df.to_csv(dummy_input_path, index=False) # Save with timestamp as column first
    logger.info(f"Created dummy input M5 data file for testing: {dummy_input_path} with {len(test_df)} rows.")

    main() # Run the main labeling process

    # Clean up dummy file
    if os.path.exists(dummy_input_path):
        os.remove(dummy_input_path)
        logger.info(f"Removed dummy input M5 data file: {dummy_input_path}")
    if os.path.exists(DEFAULT_OUTPUT_LABELED_M5_CSV): # Also remove output if created by test
        # os.remove(DEFAULT_OUTPUT_LABELED_M5_CSV)
        logger.info(f"Test output saved to: {DEFAULT_OUTPUT_LABELED_M5_CSV} (not removed for inspection).")
