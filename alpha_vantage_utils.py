import pandas as pd
import requests
import logging
import os

logger = logging.getLogger(__name__) # Logger for this module

# Default API key if not overridden by specific script using this util
DEFAULT_AV_API_KEY = os.environ.get('AV_API_KEY', 'demo')

def fetch_alpha_vantage_daily_data(api_key, from_symbol, to_symbol, outputsize='compact'):
    """
    Fetches daily FX data from Alpha Vantage and returns a pandas DataFrame.
    Outputsize 'compact' returns last 100 data points. 'full' returns up to 20 years.
    Data is sorted chronologically (oldest first).
    Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'] (volume may be 0 if not provided by AV for FX)
    """
    FUNCTION = "FX_DAILY"
    AV_URL = "https://www.alphavantage.co/query"

    # Use the provided api_key, or fall back to the module's default (environment or 'demo')
    effective_api_key = api_key if api_key else DEFAULT_AV_API_KEY

    params = {
        "function": FUNCTION,
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": outputsize,
        "apikey": effective_api_key,
        "datatype": "json"
    }
    logger.debug(f"Fetching Alpha Vantage data with params: {params.get('function')}, {params.get('from_symbol')}/{params.get('to_symbol')}, outputsize={params.get('outputsize')}")
    try:
        response = requests.get(AV_URL, params=params, timeout=20) # Increased timeout for potentially large 'full' data
        response.raise_for_status()
        data = response.json()

        if "Time Series FX (Daily)" not in data:
            err_msg = f"'Time Series FX (Daily)' not in Alpha Vantage response for {from_symbol}/{to_symbol}. Full Response: {data}"
            if "Error Message" in data:
                err_msg += f" API Error: {data['Error Message']}"
            elif "Information" in data:
                 err_msg += f" API Info: {data['Information']}"
            logger.error(err_msg)
            return None

        time_series = data["Time Series FX (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')

        # Sanitize column names (e.g., "1. open" to "open", "5. volume" to "volume")
        # Note: Alpha Vantage might not provide 'volume' for FX_DAILY. If so, this column will be missing.
        rename_map = {}
        for col in df.columns:
            if ". " in col:
                rename_map[col] = col.split(". ")[1]
        df.rename(columns=rename_map, inplace=True)

        df.index = pd.to_datetime(df.index)

        # Ensure all expected OHLC columns exist, convert to float
        for col_name in ['open', 'high', 'low', 'close']:
            if col_name not in df.columns:
                logger.error(f"Column '{col_name}' not found in Alpha Vantage response for {from_symbol}/{to_symbol} after parsing. Columns found: {df.columns.tolist()}")
                return None # Essential column missing
            df[col_name] = df[col_name].astype(float)

        # Volume is optional, if not present, we can add it as 0 or skip
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
        else:
            df['volume'] = 0.0 # Add volume column with zeros if not provided

        df = df.sort_index(ascending=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        # Reorder columns to a standard format
        standard_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[standard_columns]

        logger.info(f"Successfully fetched and parsed {len(df)} data points from Alpha Vantage for {from_symbol}/{to_symbol}.")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Alpha Vantage data for {from_symbol}/{to_symbol}: {e}", exc_info=True)
        return None
    except ValueError as e: # Includes JSONDecodeError
        logger.error(f"Error decoding JSON response from Alpha Vantage for {from_symbol}/{to_symbol}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching Alpha Vantage data for {from_symbol}/{to_symbol}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Basic test for the utility function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("Testing alpha_vantage_utils.py...")

    # Use an explicit API key for testing or rely on environment AV_API_KEY / 'demo'
    test_api_key = os.environ.get('AV_API_KEY', 'demo') # Or replace 'demo' with a specific test key
    if test_api_key == 'demo':
        logger.warning("Using 'demo' API key for testing. Data might be limited or inconsistent.")

    eur_usd_data = fetch_alpha_vantage_daily_data(api_key=test_api_key, from_symbol="EUR", to_symbol="USD", outputsize='compact')
    if eur_usd_data is not None:
        logger.info(f"EUR/USD data fetched. Shape: {eur_usd_data.shape}")
        logger.info("Last 5 rows:\n" + eur_usd_data.tail().to_string())
    else:
        logger.error("Failed to fetch EUR/USD data for testing.")

    # Example for a different pair if needed
    # gbp_jpy_data = fetch_alpha_vantage_daily_data(api_key=test_api_key, from_symbol="GBP", to_symbol="JPY", outputsize='compact')
    # if gbp_jpy_data is not None:
    #     logger.info(f"GBP/JPY data fetched. Shape: {gbp_jpy_data.shape}")
    #     logger.info("Last 5 rows:\n" + gbp_jpy_data.tail().to_string())
    # else:
    #     logger.error("Failed to fetch GBP/JPY data for testing.")

def fetch_alpha_vantage_intraday_ohlcv(api_key, from_symbol, to_symbol, interval='5min', outputsize='compact'):
    """
    Fetches intraday OHLCV FX data from Alpha Vantage.
    Interval examples: '1min', '5min', '15min', '30min', '60min'.
    Outputsize 'compact' returns last 100 data points. 'full' returns more, but limited for intraday.
    Data is sorted chronologically (oldest first).
    Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    FUNCTION = "FX_INTRADAY" # Note: AlphaVantage uses FX_INTRADAY for forex
    AV_URL = "https://www.alphavantage.co/query"

    effective_api_key = api_key if api_key else DEFAULT_AV_API_KEY

    params = {
        "function": FUNCTION,
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": effective_api_key,
        "datatype": "json"
    }
    logger.debug(f"Fetching Alpha Vantage Intraday OHLCV: {from_symbol}/{to_symbol}, interval={interval}, outputsize={outputsize}")
    try:
        response = requests.get(AV_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        time_series_key = f"Time Series FX ({interval})"
        if time_series_key not in data:
            err_msg = f"'{time_series_key}' not in Alpha Vantage response for {from_symbol}/{to_symbol}@{interval}. Full Response: {data}"
            if "Error Message" in data: err_msg += f" API Error: {data['Error Message']}"
            elif "Information" in data: err_msg += f" API Info: {data['Information']}"
            logger.error(err_msg)
            return None

        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')

        rename_map = {}
        for col in df.columns: # e.g., "1. open" to "open"
            if ". " in col: rename_map[col] = col.split(". ")[1]
        df.rename(columns=rename_map, inplace=True)

        df.index = pd.to_datetime(df.index) # Index is timestamp

        for col_name in ['open', 'high', 'low', 'close', 'volume']: # Volume is usually available for FX intraday from AV
            if col_name not in df.columns:
                logger.error(f"Column '{col_name}' not found in Intraday OHLCV for {from_symbol}/{to_symbol}@{interval}. Cols: {df.columns.tolist()}")
                return None
            df[col_name] = df[col_name].astype(float)

        df = df.sort_index(ascending=True) # Sort by timestamp
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        # Convert timestamps to UTC (Alpha Vantage intraday timestamps are typically US/Eastern)
        # Assuming timestamps from AV for FX_INTRADAY are ET. This needs verification, but is common.
        try:
            et_tz = pytz.timezone('America/New_York')
            df['timestamp'] = df['timestamp'].dt.tz_localize(et_tz).dt.tz_convert('UTC')
        except Exception as tz_e:
            logger.error(f"Error during timezone conversion for intraday data {from_symbol}/{to_symbol}@{interval}: {tz_e}. Timestamps might be naive or incorrect.", exc_info=True)
            # Depending on strictness, could return None or proceed with naive timestamps (undesirable)
            # For now, proceed but log error. Ideally, ensure source timezone is known.
            # If AV intraday is already UTC, tz_localize(None).dt.tz_convert('UTC') or tz_localize('UTC') is fine.
            # Given AV's US base, ET is a strong assumption for non-TZ-aware timestamps.

        standard_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[standard_columns]

        logger.info(f"Successfully fetched {len(df)} Intraday OHLCV points for {from_symbol}/{to_symbol}@{interval}.")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Intraday OHLCV for {from_symbol}/{to_symbol}@{interval}: {e}", exc_info=True)
        return None
    except ValueError as e:
        logger.error(f"Error decoding JSON for Intraday OHLCV {from_symbol}/{to_symbol}@{interval}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Intraday OHLCV for {from_symbol}/{to_symbol}@{interval}: {e}", exc_info=True)
        return None

def fetch_alpha_vantage_intraday_vwap(api_key, from_symbol, to_symbol, interval='5min'):
    """
    Fetches intraday VWAP data from Alpha Vantage.
    Interval examples: '1min', '5min', '15min', '30min', '60min'.
    The number of data points returned by VWAP depends on the underlying intraday data's availability.
    Data is sorted chronologically (oldest first).
    Columns: ['timestamp', 'vwap']
    """
    FUNCTION = "VWAP"
    AV_URL = "https://www.alphavantage.co/query"

    effective_api_key = api_key if api_key else DEFAULT_AV_API_KEY

    # Construct symbol for VWAP: For FX, it's just FROMCURRENCY/TOCURRENCY e.g. EURUSD
    # However, AV's VWAP endpoint documentation is more oriented towards stocks.
    # For FX, the `symbol` parameter for technical indicators is typically the stock symbol.
    # Let's test if it accepts FROM/TO like other FX endpoints or needs concatenation.
    # The examples show `symbol=MSFT`. For FX, it might implicitly use the from_symbol/to_symbol context
    # if we query an FX pair. The API might not support VWAP for FX directly via this endpoint.
    # Let's assume for now it needs a stock-like symbol. This part is speculative for FX.
    # **Correction**: Technical indicators on AV *do* work for FX pairs by using `symbol=EURUSD` (or from_symbol to_symbol).
    # The API docs for technicals show `symbol=EURUSD` for `function=EMA&symbol=EURUSD&interval=weekly...`
    # So, we should use `symbol=FROM_SYMBOL+TO_SYMBOL`.

    # The API documentation for VWAP specifically says "This API returns the volume weighted average price (VWAP) for intraday time series"
    # and takes `symbol` and `interval`.
    # It does NOT take from_symbol/to_symbol.
    # So, we must pass symbol as "EURUSD" for example.

    symbol_param = from_symbol + to_symbol # e.g. "EURUSD"

    params = {
        "function": FUNCTION,
        "symbol": symbol_param,
        "interval": interval,
        "apikey": effective_api_key,
        "datatype": "json"
    }
    logger.debug(f"Fetching Alpha Vantage Intraday VWAP: {symbol_param}, interval={interval}")
    try:
        response = requests.get(AV_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        vwap_data_key = f"Technical Analysis: VWAP"
        if vwap_data_key not in data:
            err_msg = f"'{vwap_data_key}' not in Alpha Vantage response for {symbol_param}@{interval}. Full Response: {data}"
            if "Error Message" in data: err_msg += f" API Error: {data['Error Message']}"
            elif "Information" in data: err_msg += f" API Info: {data['Information']}"
            logger.error(err_msg)
            return None

        time_series = data[vwap_data_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')

        df.index = pd.to_datetime(df.index) # Index is timestamp
        df.rename(columns={'VWAP': 'vwap'}, inplace=True) # Column is just 'VWAP'

        if 'vwap' not in df.columns:
            logger.error(f"Column 'vwap' not found in VWAP data for {symbol_param}@{interval}. Cols: {df.columns.tolist()}")
            return None
        df['vwap'] = df['vwap'].astype(float)

        df = df.sort_index(ascending=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        # Timezone conversion for VWAP timestamps
        # Assuming VWAP timestamps are also ET, consistent with intraday OHLCV
        try:
            et_tz = pytz.timezone('America/New_York')
            df['timestamp'] = df['timestamp'].dt.tz_localize(et_tz).dt.tz_convert('UTC')
        except Exception as tz_e:
            logger.error(f"Error during timezone conversion for VWAP data {symbol_param}@{interval}: {tz_e}.", exc_info=True)
            # Proceed with caution, timestamps might be naive.

        standard_columns = ['timestamp', 'vwap']
        df = df[standard_columns]

        logger.info(f"Successfully fetched {len(df)} Intraday VWAP points for {symbol_param}@{interval}.")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Intraday VWAP for {symbol_param}@{interval}: {e}", exc_info=True)
        return None
    except ValueError as e:
        logger.error(f"Error decoding JSON for Intraday VWAP {symbol_param}@{interval}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Intraday VWAP for {symbol_param}@{interval}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    # Basic test for the utility function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("Testing alpha_vantage_utils.py...")

    test_api_key = os.environ.get('AV_API_KEY', 'demo')
    if test_api_key == 'demo':
        logger.warning("Using 'demo' API key for testing. Data might be limited or inconsistent.")

    # Test daily fetcher (existing)
    # eur_usd_daily_data = fetch_alpha_vantage_daily_data(api_key=test_api_key, from_symbol="EUR", to_symbol="USD", outputsize='compact')
    # if eur_usd_daily_data is not None:
    #     logger.info(f"EUR/USD Daily data fetched. Shape: {eur_usd_daily_data.shape}. Last 5:\n{eur_usd_daily_data.tail().to_string()}")
    # else:
    #     logger.error("Failed to fetch EUR/USD Daily data for testing.")

    # Test intraday OHLCV fetcher
    logger.info("\nTesting Intraday OHLCV fetcher...")
    eur_usd_m5_ohlcv = fetch_alpha_vantage_intraday_ohlcv(api_key=test_api_key, from_symbol="EUR", to_symbol="USD", interval='5min', outputsize='compact')
    if eur_usd_m5_ohlcv is not None:
        logger.info(f"EUR/USD M5 OHLCV data fetched. Shape: {eur_usd_m5_ohlcv.shape}. Last 5:\n{eur_usd_m5_ohlcv.tail().to_string()}")
        if not eur_usd_m5_ohlcv.empty and eur_usd_m5_ohlcv['timestamp'].dt.tz is not None:
             logger.info(f"Intraday OHLCV timestamps are timezone-aware: {eur_usd_m5_ohlcv['timestamp'].dt.tz}")
        elif not eur_usd_m5_ohlcv.empty:
             logger.warning(f"Intraday OHLCV timestamps are naive: {eur_usd_m5_ohlcv['timestamp'].dt.tz}")

    else:
        logger.error("Failed to fetch EUR/USD M5 OHLCV data for testing.")

    # Test intraday VWAP fetcher
    logger.info("\nTesting Intraday VWAP fetcher...")
    eur_usd_m5_vwap = fetch_alpha_vantage_intraday_vwap(api_key=test_api_key, from_symbol="EUR", to_symbol="USD", interval='5min')
    if eur_usd_m5_vwap is not None:
        logger.info(f"EUR/USD M5 VWAP data fetched. Shape: {eur_usd_m5_vwap.shape}. Last 5:\n{eur_usd_m5_vwap.tail().to_string()}")
        if not eur_usd_m5_vwap.empty and eur_usd_m5_vwap['timestamp'].dt.tz is not None:
             logger.info(f"Intraday VWAP timestamps are timezone-aware: {eur_usd_m5_vwap['timestamp'].dt.tz}")
        elif not eur_usd_m5_vwap.empty:
             logger.warning(f"Intraday VWAP timestamps are naive: {eur_usd_m5_vwap['timestamp'].dt.tz}")
    else:
        logger.error("Failed to fetch EUR/USD M5 VWAP data for testing.")
