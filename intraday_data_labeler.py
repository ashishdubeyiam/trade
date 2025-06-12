import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timezone

import logging
import os
from datetime import datetime, timezone

# Import from intraday_feature_engineer
try:
    from intraday_feature_engineer import add_technical_indicators, ATR_PERIOD as IFE_ATR_PERIOD
    IFE_AVAILABLE = True
except ImportError:
    IFE_AVAILABLE = False
    # Mock function if import fails, to allow structural checks of this script
    def add_technical_indicators(df):
        logging.getLogger(__name__).error("intraday_feature_engineer.add_technical_indicators is NOT AVAILABLE. Returning raw DF.")
        return df
    IFE_ATR_PERIOD = 14 # Fallback
    logging.getLogger(__name__).critical("Failed to import from intraday_feature_engineer.py. Feature engineering step will be skipped.")


# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Assume calling script configures logger. For standalone testing, basicConfig can be used.

# --- Configuration ---
# Input CSV now expects raw OHLCV + VWAP data, features will be added by this script.
DEFAULT_INPUT_OHLCV_VWAP_M5_CSV = 'eur_usd_m5_master_data.csv'
DEFAULT_OUTPUT_LABELED_M5_CSV = 'eur_usd_m5_labeled_data.csv'
LOG_FILE = 'intraday_data_labeler.log'

TP_ATR_MULTIPLIER = 1.5
SL_ATR_MULTIPLIER = 1.0
MAX_HOLDING_BARS = 12
TARGET_COLUMN_NAME = 'up_move_success'
# ATR column name is now derived from the imported feature engineering settings
ATR_COL_NAME = f'atr_{IFE_ATR_PERIOD}'

def load_raw_m5_data_for_labeling(csv_filepath=DEFAULT_INPUT_OHLCV_VWAP_M5_CSV):
    """
    Loads M5 OHLCV + VWAP data (prior to feature engineering for labeling).
    Sets 'timestamp' as a UTC datetime index.
    Validates required base columns and converts them to numeric.
    """
    logger.info(f"Attempting to load raw M5 data from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"Raw M5 data file not found: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            logger.warning(f"Raw M5 data file is empty: {csv_filepath}")
            return pd.DataFrame()

        logger.info(f"Successfully loaded {len(df)} raw records from {csv_filepath}.")

        if 'timestamp' not in df.columns:
            logger.error("Raw M5 data CSV must contain a 'timestamp' column.")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Base columns needed before feature engineering for labeling (ATR needs HLC)
        # VWAP is not strictly needed for ATR calc for labeling, but good to have it loaded.
        required_base_cols = ['high', 'low', 'close', 'open', 'volume']
        if not all(col in df.columns for col in required_base_cols):
            missing_cols = [col for col in required_base_cols if col not in df.columns]
            logger.error(f"Raw M5 data is missing required base columns for feature engineering: {missing_cols}")
            return None

        for col in required_base_cols + ['vwap']: # Convert common cols if they exist
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except FileNotFoundError: # Should be caught by os.path.exists
        logger.error(f"Raw M5 data file not found (FileNotFoundError): {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Raw M5 data file is empty (EmptyDataError): {csv_filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading raw M5 data from {csv_filepath}: {e}", exc_info=True)
        return None

def generate_labels(df, tp_atr_mult=TP_ATR_MULTIPLIER, sl_atr_mult=SL_ATR_MULTIPLIER,
                    max_holding_bars=MAX_HOLDING_BARS, target_col_name=TARGET_COLUMN_NAME,
                    atr_col=ATR_COL_NAME): # atr_col now uses the dynamically set name
    """
    Generates target labels for predicting an upward price move.
    Label '1' (success) if TP is hit before SL within max_holding_bars.
    Label '0' (failure) if SL is hit first or if neither TP nor SL is hit (timeout).
    Labels np.nan if ATR is invalid or not enough future bars.
    """
    """Generates target labels for predicting an upward price move."""
    if df is None or df.empty:
        logger.warning("Input DataFrame for label generation is empty or None.")
        return df # Return as is, or an empty DF with target col

    # Critical check: ATR column must exist for labeling logic
    if atr_col not in df.columns:
        logger.error(f"Required ATR column '{atr_col}' not found in DataFrame. Cannot generate labels.")
        # Add an empty target column and return, or raise error
        df[target_col_name] = np.nan
        return df

    n_rows = len(df)
    labels = pd.Series(np.nan, index=df.index)

    logger.info(f"Generating labels for {n_rows} rows using ATR col '{atr_col}', max_holding_bars={max_holding_bars}, TP_mult={tp_atr_mult}, SL_mult={sl_atr_mult}.")

    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    atr_values = df[atr_col].values

    for i in range(n_rows - max_holding_bars):
        current_atr = atr_values[i]

        if pd.isna(current_atr) or current_atr <= 1e-7:
            logger.debug(f"Skipping label for index {df.index[i]} due to invalid ATR: {current_atr}")
            continue

        entry_price = close_prices[i]
        tp_level = entry_price + (tp_atr_mult * current_atr)
        sl_level = entry_price - (sl_atr_mult * current_atr)

        outcome_determined = False
        current_label = 0

        for j in range(1, max_holding_bars + 1):
            future_bar_idx = i + j
            future_high = high_prices[future_bar_idx]
            future_low = low_prices[future_bar_idx]

            if future_low <= sl_level:
                current_label = 0
                outcome_determined = True
                logger.debug(f"SL hit for entry at {df.index[i]} (Entry: {entry_price:.5f}, SL: {sl_level:.5f}) on bar {df.index[future_bar_idx]} (Low: {future_low:.5f})")
                break

            if future_high >= tp_level:
                current_label = 1
                outcome_determined = True
                logger.debug(f"TP hit for entry at {df.index[i]} (Entry: {entry_price:.5f}, TP: {tp_level:.5f}) on bar {df.index[future_bar_idx]} (High: {future_high:.5f})")
                break

        if not outcome_determined:
            logger.debug(f"Trade timed out for entry at {df.index[i]} (Entry: {entry_price:.5f}, TP: {tp_level:.5f}, SL: {sl_level:.5f})")

        labels.iloc[i] = current_label

    df[target_col_name] = labels

    if target_col_name in df:
        label_counts = df[target_col_name].value_counts(dropna=False)
        logger.info(f"Label generation complete. Label distribution:\n{label_counts.to_string()}")

    return df

def main():
    logger.info("--- Starting Intraday M5 Data Labeling Process (with integrated Feature Engineering) ---")

    if not IFE_AVAILABLE:
        logger.critical("intraday_feature_engineer module not available. Cannot proceed with feature engineering and labeling. Exiting.")
        return

    # 1. Load raw M5 data (OHLCV + VWAP)
    raw_m5_df = load_raw_m5_data_for_labeling(csv_filepath=DEFAULT_INPUT_OHLCV_VWAP_M5_CSV)

    if raw_m5_df is None or raw_m5_df.empty:
        logger.error(f"Failed to load raw M5 data from {DEFAULT_INPUT_OHLCV_VWAP_M5_CSV} or it's empty. Exiting.")
        return
    logger.info(f"Raw M5 data loaded. Shape: {raw_m5_df.shape}")

    # 2. Add technical indicators
    logger.info("Performing feature engineering by adding technical indicators...")
    df_with_features = add_technical_indicators(raw_m5_df) # This is the imported function

    if df_with_features is None or df_with_features.empty:
        logger.error("Feature engineering failed or resulted in an empty DataFrame. Exiting.")
        return
    logger.info(f"Technical indicators added. Shape after feature engineering: {df_with_features.shape}")
    logger.debug(f"Columns after feature eng: {df_with_features.columns.tolist()}")


    # 3. Generate labels using the DataFrame that now includes features (esp. ATR)
    # The ATR_COL_NAME constant should now correctly refer to the column created by add_technical_indicators
    logger.info(f"Generating labels using ATR column: '{ATR_COL_NAME}'")
    labeled_df = generate_labels(df_with_features)

    if labeled_df is None or labeled_df.empty:
        logger.error("Label generation resulted in an empty or None DataFrame. Exiting.")
        return

    rows_before_dropna = len(labeled_df)
    labeled_df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True) # Drop rows where label is NaN
    rows_after_dropna = len(labeled_df)
    if rows_before_dropna > rows_after_dropna:
        logger.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows where target label remained NaN (due to insufficient future data or invalid initial ATR for labeling).")

    if labeled_df.empty:
        logger.warning("No rows remaining after dropping NaN labels. Output file will not be saved or will be empty.")

    try:
        labeled_df.to_csv(DEFAULT_OUTPUT_LABELED_M5_CSV, index=True)
        logger.info(f"Successfully saved labeled M5 data ({len(labeled_df)} rows) to {DEFAULT_OUTPUT_LABELED_M5_CSV}")
    except Exception as e:
        logger.error(f"Error saving labeled M5 data to {DEFAULT_OUTPUT_LABELED_M5_CSV}: {e}", exc_info=True)

    logger.info("--- Intraday M5 Data Labeling Process Finished ---")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                        handlers=[logging.StreamHandler()])
    try:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
        logging.getLogger().addHandler(fh)
    except Exception as e:
        logger.error(f"Could not add file handler for {LOG_FILE} in standalone mode: {e}")

    # Create a dummy input CSV for testing (eur_usd_m5_master_data.csv)
    # This should now be raw data, as features are added internally.
    num_rows_test = MAX_HOLDING_BARS + 100 # Enough for features and some labels
    base_time = datetime.now(timezone.utc) - timedelta(minutes=5*num_rows_test)
    test_data_payload = { # Renamed to avoid conflict with loaded_df later
        'timestamp': pd.date_range(start=base_time, periods=num_rows_test, freq='5min', tz='UTC'),
        'open': np.random.rand(num_rows_test) * 0.01 + 1.0700, # Forex prices
        'volume': np.random.randint(10, 100, num_rows_test).astype(float) # Volume as float
    }
    _test_df = pd.DataFrame(test_data_payload) # Use temp name
    _test_df['high'] = _test_df['open'] + np.random.rand(num_rows_test) * 0.0010
    _test_df['low'] = _test_df['open'] - np.random.rand(num_rows_test) * 0.0010
    _test_df['close'] = (_test_df['open'] + _test_df['high'] + _test_df['low']) / 3
    _test_df['vwap'] = _test_df['close'] + (np.random.rand(num_rows_test) * 0.0002 - 0.0001)

    # Ensure all base columns for feature engineering are present
    # ATR_COL_NAME is NOT expected here, it will be generated.

    dummy_input_path = DEFAULT_INPUT_OHLCV_VWAP_M5_CSV # Use the new constant
    _test_df.to_csv(dummy_input_path, index=False)
    logger.info(f"Created dummy input M5 data file for testing: {dummy_input_path} with {len(_test_df)} rows.")

    main()

    if os.path.exists(dummy_input_path):
        os.remove(dummy_input_path)
        logger.info(f"Removed dummy input M5 data file: {dummy_input_path}")
    # Output file is not removed for inspection
    logger.info(f"Test output (if any) saved to: {DEFAULT_OUTPUT_LABELED_M5_CSV}")
