import os
import logging
import pandas as pd
from datetime import datetime, timezone
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.exceptions

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Assume calling script configures logger. For standalone testing, basicConfig can be used.

# --- Environment Variable Loading & Configuration ---
OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID')
OANDA_API_TOKEN = os.environ.get('OANDA_API_TOKEN')
OANDA_API_ENVIRONMENT = os.environ.get('OANDA_API_ENVIRONMENT', 'practice').lower() # Default to practice

API_DOMAINS = {
    "practice": "api-fxpractice.oanda.com",
    "live": "api-fxtrade.oanda.com"
}

# Log status of environment variables at module load time
if not OANDA_ACCOUNT_ID:
    logger.critical("OANDA_ACCOUNT_ID environment variable not set. OANDA functions will fail.")
if not OANDA_API_TOKEN:
    logger.critical("OANDA_API_TOKEN environment variable not set. OANDA functions will fail.")
if OANDA_API_ENVIRONMENT not in API_DOMAINS:
    logger.critical(f"OANDA_API_ENVIRONMENT '{OANDA_API_ENVIRONMENT}' is invalid. Use 'practice' or 'live'. Defaulting to 'practice'.")
    OANDA_API_ENVIRONMENT = 'practice' # Fallback to practice if invalid value

logger.info(f"OANDA Utils configured for Account ID (masked): "...{OANDA_ACCOUNT_ID[-4:] if OANDA_ACCOUNT_ID else 'N/A'}", Environment: {OANDA_API_ENVIRONMENT}")


def get_oanda_api_client(api_token=OANDA_API_TOKEN, environment=OANDA_API_ENVIRONMENT):
    """
    Initializes and returns an oandapyV20.API client instance.

    Args:
        api_token (str): The OANDA API access token.
        environment (str): "practice" or "live".

    Returns:
        oandapyV20.API: Initialized API client, or None if configuration is missing/invalid.
    """
    if not api_token:
        logger.error("Cannot initialize OANDA API client: API token is missing.")
        return None

    if environment not in API_DOMAINS:
        logger.error(f"Cannot initialize OANDA API client: Invalid environment '{environment}'. Use 'practice' or 'live'.")
        return None

    api_domain = API_DOMAINS[environment]

    try:
        api_client = oandapyV20.API(access_token=api_token, environment=environment) # oandapyV20 handles domain via environment param
        logger.info(f"OANDA API client initialized for environment: {environment} (domain: {api_domain})")
        return api_client
    except Exception as e:
        logger.error(f"Failed to initialize OANDA API client: {e}", exc_info=True)
        return None

def fetch_oanda_candles(api_client, account_id=OANDA_ACCOUNT_ID, instrument="EUR_USD",
                        count=100, granularity="M5", price_components="M"):
    """
    Fetches historical candle data from OANDA.

    Args:
        api_client (oandapyV20.API): Initialized OANDA API client.
        account_id (str): OANDA Account ID. (Note: Not directly used by InstrumentsCandles, but good to have for context or other calls)
        instrument (str): Target instrument (e.g., "EUR_USD").
        count (int): Number of candles to fetch.
        granularity (str): Candle timeframe (e.g., "M5", "H1", "D").
        price_components (str): Price components to fetch ("M" for Mid, "B" for Bid, "A" for Ask, or "MBA").
                                The function will prioritize Mid, then Bid, then Ask for OHLC if multiple are requested.

    Returns:
        pandas.DataFrame: DataFrame with OHLCV data, sorted by timestamp, or None if an error occurs.
                          Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if not api_client:
        logger.error("OANDA API client is not initialized. Cannot fetch candles.")
        return None
    if not account_id: # Though not used directly in InstrumentsCandles, it's a sanity check for config
        logger.error("OANDA Account ID is not configured. Cannot fetch candles.")
        return None

    params = {
        "count": count,
        "granularity": granularity,
        "price": price_components # e.g. "M" for Midpoint, "BA" for BidAsk
    }

    logger.info(f"Fetching {count} {granularity} candles for {instrument} (Price: {price_components})")

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)

    try:
        api_client.request(r)
    except oandapyV20.exceptions.V20Error as e:
        logger.error(f"OANDA API V20Error fetching candles for {instrument}: {e}", exc_info=True)
        if hasattr(e, 'msg') and 'body' in e.msg: logger.error(f"OANDA Error Body: {e.msg['body']}")
        return None
    except requests.exceptions.RequestException as e: # If oandapyV20 re-raises requests errors
        logger.error(f"Network error fetching OANDA candles for {instrument}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during OANDA API request for {instrument}: {e}", exc_info=True)
        return None

    response = r.response
    if 'candles' not in response or not response['candles']:
        logger.warning(f"No candle data returned from OANDA for {instrument} with params: {params}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    candle_data = []

    # Determine which price point to use for OHLC (mid, bid, or ask)
    # Prioritize in order: Mid ('m'), Bid ('b'), Ask ('a') based on what's available in typical candle data
    price_key_priority = []
    if 'M' in price_components.upper(): price_key_priority.append('mid')
    if 'B' in price_components.upper(): price_key_priority.append('bid')
    if 'A' in price_components.upper(): price_key_priority.append('ask')

    if not price_key_priority:
        logger.error(f"Invalid price_components '{price_components}'. Must include 'M', 'B', or 'A'.")
        return None

    for oanda_candle in response['candles']:
        if not oanda_candle.get('complete', True): # Skip incomplete candles if present
            logger.debug(f"Skipping incomplete candle for {instrument} at {oanda_candle.get('time')}")
            continue

        # Timestamp
        try:
            # OANDA timestamps are RFC3339 format, which is UTC.
            # Example: "2023-12-01T10:00:00.000000000Z"
            ts = pd.to_datetime(oanda_candle['time'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
        except ValueError: # Fallback for formats without fractional seconds if AV changes
             ts = pd.to_datetime(oanda_candle['time'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        except Exception as e_ts:
            logger.warning(f"Could not parse timestamp '{oanda_candle['time']}'. Error: {e_ts}. Skipping candle.")
            continue

        # OHLC - find the first available price key based on priority
        ohlc_data = None
        chosen_price_key = None
        for p_key in price_key_priority:
            if p_key in oanda_candle:
                ohlc_data = oanda_candle[p_key]
                chosen_price_key = p_key
                break

        if not ohlc_data:
            logger.warning(f"No suitable price data (mid/bid/ask based on '{price_components}') found for candle at {ts}. Skipping.")
            continue

        try:
            candle_dict = {
                'timestamp': ts,
                'open': float(ohlc_data['o']),
                'high': float(ohlc_data['h']),
                'low': float(ohlc_data['l']),
                'close': float(ohlc_data['c']),
                'volume': int(oanda_candle['volume'])
            }
            candle_data.append(candle_dict)
        except KeyError as ke:
            logger.warning(f"KeyError parsing candle data for {ts} with price key '{chosen_price_key}': {ke}. Candle: {oanda_candle}. Skipping.")
            continue
        except ValueError as ve:
            logger.warning(f"ValueError parsing OHLC for {ts} with price key '{chosen_price_key}': {ve}. OHLC Data: {ohlc_data}. Skipping.")
            continue

    if not candle_data:
        logger.warning(f"No valid candles processed for {instrument} after parsing.")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(candle_data)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Successfully parsed {len(df)} candles for {instrument} using '{chosen_price_key}' prices.")
    return df


if __name__ == '__main__':
    # Setup basic logging for standalone testing
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                            handlers=[logging.StreamHandler()])

    logger.info("--- Testing oanda_utils.py ---")

    # Attempt to load .env file for local testing if python-dotenv is available
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            logger.info("Loaded .env file for OANDA credentials.")
            # Reload environment variables after dotenv load
            OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID', OANDA_ACCOUNT_ID) # Keep existing if dotenv didn't set
            OANDA_API_TOKEN = os.environ.get('OANDA_API_TOKEN', OANDA_API_TOKEN)
            OANDA_API_ENVIRONMENT = os.environ.get('OANDA_API_ENVIRONMENT', OANDA_API_ENVIRONMENT).lower()
            logger.info(f"OANDA Test Config: Account ID (masked): "...{OANDA_ACCOUNT_ID[-4:] if OANDA_ACCOUNT_ID else 'N/A'}", Env: {OANDA_API_ENVIRONMENT}")
        else:
            logger.info(".env file not found or empty, relying on globally set environment variables.")
    except ImportError:
        logger.info("python-dotenv not installed, relying on globally set environment variables for testing.")
    except Exception as e_dotenv:
        logger.error(f"Error loading .env file: {e_dotenv}")


    if not OANDA_ACCOUNT_ID or not OANDA_API_TOKEN:
        logger.critical("OANDA_ACCOUNT_ID or OANDA_API_TOKEN is not set. Cannot run test.")
    else:
        client = get_oanda_api_client() # Uses module-level loaded token and environment
        if client:
            logger.info("\n--- Test Fetching M5 EUR_USD Candles (Midpoint) ---")
            eur_usd_m5 = fetch_oanda_candles(client, instrument="EUR_USD", count=10, granularity="M5", price_components="M")
            if eur_usd_m5 is not None:
                logger.info(f"Fetched {len(eur_usd_m5)} EUR_USD M5 candles.")
                logger.info("Head:\n" + eur_usd_m5.head().to_string())
                logger.info("Tail:\n" + eur_usd_m5.tail().to_string())
                if not eur_usd_m5.empty: logger.info(f"Timestamp Dtype: {eur_usd_m5['timestamp'].dtype}") # Should be datetime64[ns, UTC]
            else:
                logger.error("Failed to fetch EUR_USD M5 candles.")

            logger.info("\n--- Test Fetching M1 GBP_JPY Candles (Bid prices) ---")
            gbp_jpy_m1 = fetch_oanda_candles(client, instrument="GBP_JPY", count=5, granularity="M1", price_components="B")
            if gbp_jpy_m1 is not None:
                logger.info(f"Fetched {len(gbp_jpy_m1)} GBP_JPY M1 candles.")
                logger.info("Head:\n" + gbp_jpy_m1.head().to_string())
            else:
                logger.error("Failed to fetch GBP_JPY M1 candles.")
        else:
            logger.error("Failed to initialize OANDA API client for testing.")
    logger.info("--- OANDA Utils Test Finished ---")
