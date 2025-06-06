import pandas as pd
import numpy as np
import logging
import os

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Assume calling script configures logger. For standalone testing, basicConfig can be used.

# --- Configuration ---
DEFAULT_INPUT_M5_CSV = 'eur_usd_m5_master_data.csv' # Used in __main__ for testing
EMA_SHORT_PERIOD = 5
EMA_LONG_PERIOD = 12
RSI_PERIOD = 9
ATR_PERIOD = 14

def load_m5_data(csv_filepath=DEFAULT_INPUT_M5_CSV):
    """
    Loads M5 OHLCV + VWAP data from a CSV file.
    Sets 'timestamp' as a UTC datetime index.
    Converts relevant columns to numeric.
    """
    logger.info(f"Attempting to load M5 data from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"M5 data file not found: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            logger.warning(f"M5 data file is empty: {csv_filepath}")
            return pd.DataFrame() # Return empty DataFrame

        logger.info(f"Successfully loaded {len(df)} records from {csv_filepath}.")

        # Timestamp processing
        if 'timestamp' not in df.columns:
            logger.error("M5 data CSV must contain a 'timestamp' column.")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp parsing failed
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True) # Ensure chronological order

        # Convert OHLCV + VWAP columns to numeric
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.warning(f"Column '{col}' not found in M5 data. It will be missing from features.")

        # It's good practice to check for NaNs that might have been introduced by 'coerce'
        # For example, if a column had non-numeric text.
        # df.dropna(subset=cols_to_numeric, inplace=True) # Optional: drop rows with any NaNs in crucial data
        # logger.info(f"Shape after numeric conversion and NaN drop (if any): {df.shape}")

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

def calculate_emas(df, short_period=EMA_SHORT_PERIOD, long_period=EMA_LONG_PERIOD):
    """Calculates short and long period EMAs on the 'close' price."""
    if 'close' not in df.columns:
        logger.error("DataFrame must contain 'close' column to calculate EMAs.")
        return df
    logger.debug(f"Calculating EMAs with short_period={short_period}, long_period={long_period}")
    df[f'ema_{short_period}'] = df['close'].ewm(span=short_period, adjust=False).mean()
    df[f'ema_{long_period}'] = df['close'].ewm(span=long_period, adjust=False).mean()
    return df

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculates RSI on the 'close' price using Exponential Moving Average for gains/losses."""
    if 'close' not in df.columns:
        logger.error("DataFrame must contain 'close' column to calculate RSI.")
        return df
    logger.debug(f"Calculating RSI with period={period}")

    delta = df['close'].diff(1)

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0) # Loss is positive value

    # Use EWM for average gain/loss calculation
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero, result in NaN for RSI then

    rsi = 100 - (100 / (1 + rs))
    df[f'rsi_{period}'] = rsi
    return df

def calculate_atr(df, period=ATR_PERIOD):
    """Calculates Average True Range (ATR)."""
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        logger.error("DataFrame must contain 'high', 'low', and 'close' columns to calculate ATR.")
        return df
    logger.debug(f"Calculating ATR with period={period}")

    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))

    # True Range (TR)
    # Max of (High - Low), abs(High - Previous Close), abs(Low - Previous Close)
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)

    # ATR is typically a smoothed moving average of TR. EWM is common.
    df[f'atr_{period}'] = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return df

def calculate_ema_trend_score(df, short_ema_col=f'ema_{EMA_SHORT_PERIOD}', long_ema_col=f'ema_{EMA_LONG_PERIOD}'):
    """Calculates a trend score based on the percentage difference between short and long EMAs."""
    if not all(col in df.columns for col in [short_ema_col, long_ema_col]):
        logger.error(f"DataFrame must contain '{short_ema_col}' and '{long_ema_col}' to calculate EMA trend score.")
        return df
    logger.debug(f"Calculating EMA trend score using columns: {short_ema_col}, {long_ema_col}")

    # Percentage difference: (short_ema - long_ema) / long_ema * 100
    # Handle potential division by zero if long_ema is 0 (though unlikely for prices)
    trend_score = (df[short_ema_col] - df[long_ema_col]) / df[long_ema_col].replace(0, np.nan) * 100
    df['ema_trend_score'] = trend_score
    return df

def add_technical_indicators(df):
    """
    Adds a suite of technical indicators to the M5 DataFrame.
    Input DataFrame should have 'timestamp' as index and OHLCV + VWAP columns.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Cannot add technical indicators.")
        return df

    logger.info(f"Adding technical indicators to DataFrame with shape {df.shape}")

    # Ensure required columns are present before starting
    required_ohlc = ['open', 'high', 'low', 'close', 'volume'] # VWAP is also expected by this point
    if not all(col in df.columns for col in required_ohlc):
        logger.error(f"Input DataFrame for add_technical_indicators is missing one or more required columns: {required_ohlc}")
        # Return df as is, or an empty df, or raise error. For now, return as is.
        return df

    df_with_indicators = df.copy() # Work on a copy

    # Calculate indicators
    df_with_indicators = calculate_emas(df_with_indicators)
    df_with_indicators = calculate_rsi(df_with_indicators)
    df_with_indicators = calculate_atr(df_with_indicators)
    df_with_indicators = calculate_ema_trend_score(df_with_indicators)

    # VWAP is assumed to be already present in the input 'df'
    if 'vwap' not in df_with_indicators.columns:
        logger.warning("'vwap' column not found in DataFrame. It will not be included in the final output from this function.")

    logger.info(f"Finished adding technical indicators. DataFrame shape: {df_with_indicators.shape}")
    logger.debug(f"Columns after adding indicators: {df_with_indicators.columns.tolist()}")

    return df_with_indicators


if __name__ == '__main__':
    # --- Basic Test for the utility module ---
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("--- Testing intraday_feature_engineer.py ---")

    # Create a dummy M5 data CSV for testing
    num_rows = 100
    data = {
        'timestamp': pd.date_range(end=datetime.now(timezone.utc), periods=num_rows, freq='5min'),
        'open': np.random.rand(num_rows) * 10 + 1.0500, # Random prices around 1.05-1.15
        'high': lambda df: df['open'] + np.random.rand(num_rows) * 0.0010,
        'low': lambda df: df['open'] - np.random.rand(num_rows) * 0.0010,
        'close': lambda df: (df['open'] + df['high']() + df['low']()) / 3 , # More realistic close
        'volume': np.random.randint(100, 1000, num_rows),
        'vwap': lambda df: df['close']() + (np.random.rand(num_rows) * 0.0002 - 0.0001) # VWAP close to close
    }

    # Resolve lambda functions to create the DataFrame
    dummy_m5_df = pd.DataFrame({'timestamp': data['timestamp']})
    dummy_m5_df['open'] = data['open']
    dummy_m5_df['high'] = data['high'](dummy_m5_df)
    dummy_m5_df['low'] = data['low'](dummy_m5_df)
    dummy_m5_df['close'] = data['close'](dummy_m5_df) # Pass the DataFrame to the lambda
    dummy_m5_df['volume'] = data['volume']
    dummy_m5_df['vwap'] = data['vwap'](dummy_m5_df) # Pass the DataFrame to the lambda

    dummy_csv_path = 'dummy_test_m5_data.csv'
    dummy_m5_df.to_csv(dummy_csv_path, index=False) # Save without index, as load_m5_data expects timestamp as column
    logger.info(f"Created dummy M5 data file: {dummy_csv_path} with {len(dummy_m5_df)} rows.")

    # Test load_m5_data
    loaded_df = load_m5_data(csv_filepath=dummy_csv_path)

    if loaded_df is not None and not loaded_df.empty:
        logger.info(f"Loaded dummy M5 data. Shape: {loaded_df.shape}, Index type: {type(loaded_df.index)}")
        # print("\nLoaded DataFrame head:\n", loaded_df.head())
        # print("\nLoaded DataFrame info:\n")
        # loaded_df.info()

        # Test add_technical_indicators
        logger.info("\nTesting add_technical_indicators...")
        df_with_inds = add_technical_indicators(loaded_df)

        logger.info(f"DataFrame with indicators shape: {df_with_inds.shape}")
        logger.info("\nDataFrame with indicators - Head:\n" + df_with_inds.head(EMA_LONG_PERIOD + 2).to_string()) # Show enough rows for EMAs to start
        logger.info("\nDataFrame with indicators - Tail:\n" + df_with_inds.tail().to_string())
        # logger.info("\nInfo after adding indicators:\n")
        # df_with_inds.info()

        # Check for NaNs at the beginning, and non-NaNs at the end for a typical indicator
        rsi_col_name = f'rsi_{RSI_PERIOD}'
        if rsi_col_name in df_with_inds.columns:
            logger.info(f"'{rsi_col_name}' head (expect NaNs):\n{df_with_inds[rsi_col_name].head(RSI_PERIOD + 5).to_string()}")
            logger.info(f"'{rsi_col_name}' tail (expect values):\n{df_with_inds[rsi_col_name].tail().to_string()}")
            if df_with_inds[rsi_col_name].iloc[RSI_PERIOD:].isnull().any(): # Check for NaNs after initial period
                 logger.warning(f"Unexpected NaNs found in '{rsi_col_name}' after initial period.")
        else:
            logger.error(f"'{rsi_col_name}' column not found after adding indicators.")

    else:
        logger.error("Failed to load or process dummy M5 data for testing.")

    # Clean up dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
        logger.info(f"Removed dummy M5 data file: {dummy_csv_path}")
