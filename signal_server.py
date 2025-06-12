from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib # For loading the ML model
from datetime import datetime, timezone, timedelta, date as dt_date
import logging
import os
import pytz
from waitress import serve

# Import utility modules
from intraday_feature_engineer import load_m5_data, add_technical_indicators
from event_processor_utils import load_calendar_events, get_relevant_events_for_day, is_event_blackout_active
from news_processor_utils import load_news_headlines, check_for_extreme_news, EXTREME_KEYWORDS_LIST

# Import feature list from trainer to ensure consistency
FEATURE_COLUMNS_TO_USE = [] # Default to empty, will be populated from trainer if import works
try:
    from intraday_model_trainer import FEATURE_COLUMNS_TO_USE as TRAINER_FEATURE_COLUMNS
    if TRAINER_FEATURE_COLUMNS: # If it's not empty or None
        FEATURE_COLUMNS_TO_USE = TRAINER_FEATURE_COLUMNS
except ImportError:
    # This logger is defined before the main logger setup, so it's temporary for startup error.
    startup_logger = logging.getLogger(__name__ + "_startup_check")
    if not startup_logger.handlers: # Avoid adding handler multiple times if script is reloaded
        startup_stream_handler = logging.StreamHandler()
        startup_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        startup_logger.addHandler(startup_stream_handler)
        startup_logger.setLevel(logging.WARNING)
    startup_logger.error("Could not import FEATURE_COLUMNS_TO_USE from intraday_model_trainer.py. Critical feature list missing.")


# --- Configuration ---
M5_DATA_CSV = 'eur_usd_m5_master_data.csv'
CALENDAR_CSV = 'economic_calendar_events.csv'
NEWS_CSV = 'news_headlines.csv'
LIVE_INTRADAY_MODEL_FILE = 'live_intraday_m5_model.joblib'

NETHERLANDS_TIMEZONE_STR = 'Europe/Amsterdam'
TRADING_PAUSE_START_HOUR_CET = 14
TRADING_RESUME_HOUR_CET = 1
NEWS_KEYWORD_LOOKBACK_HOURS = 1
M5_DATA_LOAD_WINDOW_BARS = 200

SERVER_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
SERVER_PORT = int(os.environ.get('FLASK_PORT', 5000))
LOG_FILE = 'signal_server.log'

PROB_THRESHOLD_BUY = 0.55
PROB_THRESHOLD_SELL = 0.45

SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.0

# Technical indicator parameters (used by intraday_feature_engineer)
# These are defaults in intraday_feature_engineer, listed here for clarity and for required_indicators check
# Make sure these match the actual parameters used if they were changed in intraday_feature_engineer
EMA_SHORT_PERIOD = 5
EMA_LONG_PERIOD = 12
RSI_PERIOD = 9
ATR_PERIOD = 14

app = Flask(__name__)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e_fh:
        logger.error(f"Failed to set up file logger for {LOG_FILE}: {e_fh}", exc_info=True)
    logger.setLevel(logging.INFO)

# --- Global Variables for Loaded ML Model ---
ML_MODEL = None
LOADED_MODEL_FILENAME = None
LOADED_MODEL_TIMESTAMP = None

def load_ml_model_on_startup():
    global ML_MODEL, LOADED_MODEL_FILENAME, LOADED_MODEL_TIMESTAMP
    logger.info(f"Attempting to load ML model from: {LIVE_INTRADAY_MODEL_FILE}")
    try:
        if os.path.exists(LIVE_INTRADAY_MODEL_FILE):
            ML_MODEL = joblib.load(LIVE_INTRADAY_MODEL_FILE)
            LOADED_MODEL_FILENAME = LIVE_INTRADAY_MODEL_FILE
            mtime = os.path.getmtime(LIVE_INTRADAY_MODEL_FILE)
            LOADED_MODEL_TIMESTAMP = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
            logger.info(f"ML Model '{LOADED_MODEL_FILENAME}' (Timestamp: {LOADED_MODEL_TIMESTAMP}) loaded successfully.")
            if not FEATURE_COLUMNS_TO_USE: # This list comes from intraday_model_trainer.py
                 logger.warning("FEATURE_COLUMNS_TO_USE is empty (likely failed import from trainer). Model predictions might fail or be incorrect.")
        else:
            logger.critical(f"CRITICAL: ML Model file {LIVE_INTRADAY_MODEL_FILE} not found.")
            ML_MODEL = None
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading ML model {LIVE_INTRADAY_MODEL_FILE}: {e}", exc_info=True)
        ML_MODEL = None

# Load the model at server startup
load_ml_model_on_startup()


# --- Main Application Logic ---
@app.route('/getsignal', methods=['GET'])
def get_signal_endpoint():
    now_utc = datetime.now(timezone.utc)
    client_ip = request.remote_addr
    logger.info(f"Received /getsignal request from IP: {client_ip} at {now_utc.isoformat()}")

    # Initialize response variables
    final_signal = "HOLD"
    blackout_reason = "NONE"
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    extreme_news_alert_active = False
    triggering_headlines_info = []
    market_situation_data = {}
    prediction_probability_up = None

    def error_response(message, status_code=500, reason="ERROR"):
        logger.error(message)
        return jsonify({
            "signal": "HOLD", "blackout_reason": reason, "error_message": message,
            "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "signal_timestamp_utc": now_utc.isoformat(),
            "model_name": LOADED_MODEL_FILENAME if LOADED_MODEL_FILENAME else "Intraday M5 ML (Not Loaded)",
            "model_version_timestamp": LOADED_MODEL_TIMESTAMP if LOADED_MODEL_TIMESTAMP else "N/A",
            "extreme_news_alert_active": False, "triggering_headlines": [],
            "market_situation": {}, "prediction_probability_up": None
        }), status_code

    if ML_MODEL is None:
        return error_response("ML Model is not loaded. Cannot generate signal.", reason="MODEL_NOT_LOADED")
    if not FEATURE_COLUMNS_TO_USE:
        return error_response("FEATURE_COLUMNS_TO_USE is not defined (import issue from trainer). Cannot make predictions.", reason="MODEL_FEATURES_CONFIG_EMPTY")

    # --- M5 Data Processing and ML Prediction ---
    technical_signal = "HOLD" # Default technical signal
    try:
        m5_full_df = load_m5_data(M5_DATA_CSV)
        if m5_full_df is None or m5_full_df.empty:
            return error_response(f"Could not load M5 data from {M5_DATA_CSV} or file is empty.", reason="DATA_LOAD_ERROR_M5")

        # Ensure enough data for feature calculation window + a few for stability
        min_bars_for_features = max(EMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD) + 20 # Heuristic for stable features
        if len(m5_full_df) < min_bars_for_features :
             logger.warning(f"M5 data ({len(m5_full_df)} bars) too short for reliable indicators. Need at least {min_bars_for_features }.")
             return error_response(f"M5 data window too short for reliable indicators ({len(m5_full_df)} vs {min_bars_for_features} needed).",
                                   status_code=422, reason="DATA_WINDOW_TOO_SHORT_M5")

        # Use a window of data for feature engineering, ensuring it's enough for all lookbacks
        # M5_DATA_LOAD_WINDOW_BARS should be >= min_bars_for_features
        data_load_window = max(M5_DATA_LOAD_WINDOW_BARS, min_bars_for_features)
        m5_window_df = m5_full_df.iloc[-data_load_window:].copy() if len(m5_full_df) >= data_load_window else m5_full_df.copy()


        m5_data_with_indicators = add_technical_indicators(m5_window_df)
        if m5_data_with_indicators is None or m5_data_with_indicators.empty:
            return error_response("Failed to add technical indicators.", reason="FEATURE_ENG_ERROR_M5")

        latest_bar = m5_data_with_indicators.iloc[-1]
        nan_model_features = [feat for feat in FEATURE_COLUMNS_TO_USE if pd.isnull(latest_bar.get(feat))]
        if nan_model_features:
            return error_response(f"Latest M5 bar ML features are NaN for: {', '.join(nan_model_features)}.",
                                  status_code=422, reason="MODEL_FEATURES_NAN_M5")

        latest_atr_col_name = f'atr_{ATR_PERIOD}'
        latest_atr = latest_bar.get(latest_atr_col_name, np.nan)
        if pd.isnull(latest_atr): # ATR is needed for SL/TP, even if not a model feature
            return error_response(f"Latest M5 bar ATR ('{latest_atr_col_name}') is NaN.",
                                  status_code=422, reason="ATR_NAN_M5")

        latest_close = latest_bar['close']
        entry_price = latest_close # Base entry price for signal

        market_situation_data = {'m5_timestamp_utc': latest_bar.name.isoformat(), 'm5_close': round(latest_close, 5)}
        for col in FEATURE_COLUMNS_TO_USE + [latest_atr_col_name]: # Include ATR for market situation
            if col in latest_bar and pd.notnull(latest_bar[col]):
                # More specific rounding for different types of values
                if 'rsi' in col or 'score' in col : market_situation_data[col] = round(latest_bar[col], 2)
                else: market_situation_data[col] = round(latest_bar[col], 5)
        logger.info(f"Latest M5 market situation features prepared: {market_situation_data}")

        # ML Prediction
        feature_vector_df = pd.DataFrame([latest_bar[FEATURE_COLUMNS_TO_USE]], columns=FEATURE_COLUMNS_TO_USE)
        prob_array = ML_MODEL.predict_proba(feature_vector_df)
        prediction_probability_up = round(prob_array[0][1], 4)
        market_situation_data['prediction_probability_up'] = prediction_probability_up # Add to market_situation
        logger.info(f"ML Model P(up_move_success) = {prediction_probability_up}")

        if prediction_probability_up > PROB_THRESHOLD_BUY: technical_signal = "BUY"
        elif prediction_probability_up < PROB_THRESHOLD_SELL: technical_signal = "SELL"
        logger.info(f"ML-based technical signal: {technical_signal}")

    except Exception as e:
        logger.error(f"Error during M5 data processing or ML prediction: {e}", exc_info=True)
        # Fall through, technical_signal remains "HOLD", reason will be set if it was already an error
        # Or explicitly set error state:
        return error_response(f"Error during M5 data processing or ML prediction: {e}", reason="M5_PROCESSING_ML_ERROR")


    # --- Filter Hierarchy ---
    final_signal = technical_signal
    if technical_signal == "HOLD": # If ML itself suggests HOLD
        blackout_reason = "ML_THRESHOLD_HOLD"

    # 1. Economic Calendar Blackout
    try:
        calendar_df = load_calendar_events(CALENDAR_CSV)
        if calendar_df is not None and not calendar_df.empty:
            relevant_events = get_relevant_events_for_day(calendar_df, now_utc.date(),
                                                          target_currencies=['USD', 'EUR'], target_impacts=['High'])
            if is_event_blackout_active(relevant_events):
                final_signal = "HOLD"; blackout_reason = "NEWS_DAY_BLACKOUT"
        elif calendar_df is None: logger.warning(f"Could not load {CALENDAR_CSV} for calendar check. Calendar filter skipped.")
        else: logger.info(f"{CALENDAR_CSV} is empty. No calendar events to check.")
    except Exception as e: logger.error(f"Error in calendar check: {e}", exc_info=True)

    # 2. Netherlands Session Pause (only if not already blacked out by higher priority)
    if final_signal == "HOLD" and blackout_reason != "NEWS_DAY_BLACKOUT": # If still HOLD from ML, or if ML was BUY/SELL but no news blackout
        pass # Don't overwrite ML_THRESHOLD_HOLD unless session pause is active
    if blackout_reason == "NONE" or blackout_reason == "ML_THRESHOLD_HOLD": # Only apply if not already news blacked out
        try:
            cet_tz = pytz.timezone(NETHERLANDS_TIMEZONE_STR)
            now_cet = now_utc.astimezone(cet_tz)
            if now_cet.hour >= TRADING_PAUSE_START_HOUR_CET or now_cet.hour < TRADING_RESUME_HOUR_CET:
                final_signal = "HOLD"; blackout_reason = "SESSION_PAUSE_CET"
        except Exception as e: logger.error(f"Error in session pause check: {e}", exc_info=True)

    # 3. Extreme News Check (informational)
    try:
        headlines_df = load_news_headlines(NEWS_CSV)
        if headlines_df is not None and not headlines_df.empty:
            extreme_news_alert_active, triggering_headlines_info = check_for_extreme_news(
                headlines_df, keywords=EXTREME_KEYWORDS_LIST, lookback_hours=NEWS_KEYWORD_LOOKBACK_HOURS)
            if extreme_news_alert_active: logger.warning(f"EXTREME NEWS ALERT active. Headlines: {triggering_headlines_info}")
        elif headlines_df is None: logger.warning(f"Could not load {NEWS_CSV} for news check.")
        else: logger.info(f"{NEWS_CSV} is empty. No news to check.")
    except Exception as e: logger.error(f"Error in news check: {e}", exc_info=True)

    logger.info(f"Signal after filters: {final_signal}, Reason: {blackout_reason}")

    # --- Calculate SL/TP ---
    if final_signal in ["BUY", "SELL"]:
        if pd.notnull(latest_atr) and latest_atr > 1e-6: # Ensure ATR is valid and not effectively zero
            if final_signal == "BUY":
                stop_loss = entry_price - (SL_ATR_MULTIPLIER * latest_atr)
                take_profit = entry_price + (TP_ATR_MULTIPLIER * latest_atr)
            else: # SELL
                stop_loss = entry_price + (SL_ATR_MULTIPLIER * latest_atr)
                take_profit = entry_price - (TP_ATR_MULTIPLIER * latest_atr)
            logger.info(f"Calculated SL: {stop_loss:.5f}, TP: {take_profit:.5f} for Entry: {entry_price:.5f}, ATR: {latest_atr:.5f}")
        else: # Should have been caught by ATR_NAN_M5 earlier, but as a final safety
            logger.warning(f"Signal was {final_signal} but ATR is invalid ({latest_atr}). Forcing HOLD.")
            final_signal = "HOLD"
            current_br = blackout_reason
            if current_br == "NONE" or current_br == "ML_THRESHOLD_HOLD":
                blackout_reason = "INVALID_ATR_FOR_SLTP"
            else: blackout_reason = current_br + ";INVALID_ATR" # Append reason
            stop_loss = 0.0; take_profit = 0.0

    response_data = {
        "signal": final_signal, "blackout_reason": blackout_reason,
        "entry_price": round(entry_price, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "signal_timestamp_utc": now_utc.isoformat(),
        "model_name": LOADED_MODEL_FILENAME or "Intraday M5 ML (Not Loaded)",
        "model_version_timestamp": LOADED_MODEL_TIMESTAMP or "N/A",
        "extreme_news_alert_active": extreme_news_alert_active,
        "triggering_headlines": triggering_headlines_info,
        "market_situation": market_situation_data,
        "prediction_probability_up": prediction_probability_up
    }
    logger.info(f"Final response: {response_data}")
    return jsonify(response_data)

if __name__ == '__main__':
    if not FEATURE_COLUMNS_TO_USE:
        logger.critical("SERVER STARTUP FAILED: FEATURE_COLUMNS_TO_USE is empty. Import from intraday_model_trainer.py might have failed or list was empty there. Model predictions will fail.")
        exit(1) # Exit if critical configuration is missing

    logger.info(f"Starting M5 Signal Server with Waitress on http://{SERVER_HOST}:{SERVER_PORT}")
    if ML_MODEL is None: logger.warning("ML Model was not loaded at startup. Server will produce errors for ML-based signals until a model is available at {LIVE_INTRADAY_MODEL_FILE}.")

    serve(app, host=SERVER_HOST, port=SERVER_PORT, threads=10)
