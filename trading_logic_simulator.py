import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb # Though model is joblib, good to have for type hinting if needed

# --- Configuration from previous script (for feature engineering consistency) ---
MODEL_PATH = 'lgbm_eurusd_direction_model.joblib'
DATA_FILE = 'eur_usd_daily_1yr.csv' # Used for ATR and feature engineering context
LAG_DAYS = 3
SMA_SHORT_PERIOD = 5
SMA_LONG_PERIOD = 10
RSI_PERIOD = 14
ATR_PERIOD = 14

# --- Simulation Configuration ---
PROB_THRESHOLD_BUY = 0.65
PROB_THRESHOLD_SELL = 0.35
FIXED_LOT_SIZE = 0.02
ATR_MULTIPLIER_SL = 1.0
ATR_MULTIPLIER_TP = 1.0

# --- Global state for simulation (simplified) ---
current_position = None  # None, 'LONG', 'SHORT'
entry_price = 0.0
stop_loss_price = 0.0
take_profit_price = 0.0
sim_trades = [] # To store details of trades for summary

# --- Helper Functions (some might be copied/adapted from eurusd_prediction_model.py) ---

def load_data(file_path):
    """Loads data from a CSV file, ensuring datetime parsing."""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif df.columns[0] == 'timestamp' or (df.columns[0].startswith('Unnamed') and 'date' in df.columns[1].lower()): # common CSV export issues
             # Try to infer timestamp from first column if not explicitly named or if it's an index with date info next
            try:
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
                df = df.sort_values('timestamp').reset_index(drop=True)
                print("Inferred 'timestamp' from the first column.")
            except Exception as e_infer:
                print(f"Warning: Could not infer 'timestamp' from first column: {e_infer}. Data should be sorted chronologically.")
        else:
            print("Warning: 'timestamp' column not found and could not be inferred. Data must be pre-sorted chronologically.")

        # Ensure essential columns are present
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Data must contain {required_cols}. Found: {df.columns.tolist()}")
            return None
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Avoid division by zero for loss; if loss is 0, RSI is 100 (or 0 if gain is also 0)
    rs = gain / loss.replace(0, 0.000001) # Replace 0 loss with a tiny number
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)"""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def engineer_features(df_segment, full_data_for_atr_calc=None):
    """
    Engineers features for a given data segment.
    Requires enough prior data in df_segment for lags/SMAs/RSI,
    or full_data_for_atr_calc for ATR on the latest point.
    """
    if df_segment is None or 'close' not in df_segment.columns:
        print("Error: DataFrame is None or 'close' column is missing for feature engineering.")
        return None, None # Return None for features and ATR

    df_feat = df_segment.copy()

    # 1. Lagged closing prices
    for i in range(1, LAG_DAYS + 1):
        df_feat[f'Close_t-{i}'] = df_feat['close'].shift(i)

    # 2. Simple Moving Averages (SMA)
    df_feat[f'SMA_{SMA_SHORT_PERIOD}'] = df_feat['close'].rolling(window=SMA_SHORT_PERIOD).mean()
    df_feat[f'SMA_{SMA_LONG_PERIOD}'] = df_feat['close'].rolling(window=SMA_LONG_PERIOD).mean()

    # 3. Relative Strength Index (RSI)
    df_feat[f'RSI_{RSI_PERIOD}'] = calculate_rsi(df_feat['close'], period=RSI_PERIOD)

    # 4. ATR (calculated on the full_data_for_atr_calc to get the latest ATR)
    # The ATR for the *current* day (latest_row) uses high, low, close from full_data_for_atr_calc
    current_atr = None
    if full_data_for_atr_calc is not None:
        atr_series = calculate_atr(full_data_for_atr_calc['high'],
                                   full_data_for_atr_calc['low'],
                                   full_data_for_atr_calc['close'],
                                   period=ATR_PERIOD)
        if not atr_series.empty:
            current_atr = atr_series.iloc[-1] # Get the latest ATR value corresponding to the latest row in full_data_for_atr_calc

    # We need features for the *last row* of df_feat to make a prediction for the *next* day.
    # So, we select the last row after all calculations.
    # NaNs will be present in early rows, but the last row should be valid if df_segment is long enough.

    # Get the latest features
    latest_features = df_feat.iloc[-1:] # Keep as DataFrame

    # Check if latest_features has NaN values (can happen if df_segment is too short)
    # These are the features that will be fed into the model
    feature_cols_for_model = [f'Close_t-{i}' for i in range(1, LAG_DAYS + 1)] + \
                             [f'SMA_{SMA_SHORT_PERIOD}', f'SMA_{SMA_LONG_PERIOD}', f'RSI_{RSI_PERIOD}']

    if latest_features[feature_cols_for_model].isnull().any().any():
        # print(f"Warning: NaN values found in latest features for timestamp {df_feat['timestamp'].iloc[-1]}. Not generating prediction.")
        # print(latest_features[feature_cols_for_model])
        return None, None # Not enough data to generate valid features for the latest point

    return latest_features[feature_cols_for_model], current_atr


# --- Main Simulation Logic ---
def run_simulation(model, historical_data):
    global current_position, entry_price, stop_loss_price, take_profit_price, sim_trades

    print("\n--- Starting Trading Simulation ---")

    # For simulation, we iterate through a portion of the data as if it's new daily data.
    # We need enough history for feature engineering.
    # Let's define the start point for simulation well after initial NaN period for features.
    # Max period needed: max(LAG_DAYS, SMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD) + buffer
    min_history_needed = max(LAG_DAYS + 1, SMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD) + 5 # +1 for diff, + buffer

    if len(historical_data) < min_history_needed:
        print(f"Error: Not enough historical data to run simulation. Need at least {min_history_needed} rows.")
        return

    # Simulate iterating from 'min_history_needed' onwards.
    # The row at `i` is the "current day". We use data up to `i` to predict for `i+1`.
    # Trades based on prediction for `i+1` are evaluated/executed using open/high/low/close of day `i+1`.

    simulation_start_index = min_history_needed

    print(f"Simulation will start from index {simulation_start_index} of the historical data.")

    for i in range(simulation_start_index, len(historical_data)):
        current_day_data = historical_data.iloc[i] # This is the day that has just "closed"

        # For SL/TP check, we need the open, high, low of the "current processing day" (which is `i` in this loop)
        # if a trade was opened based on prediction from `i-1`'s close.
        day_open = current_day_data['open']
        day_high = current_day_data['high']
        day_low = current_day_data['low']
        day_close = current_day_data['close'] # Used for counter-signal exits or new entries
        day_timestamp = current_day_data['timestamp']

        print(f"\nProcessing Day: {day_timestamp.strftime('%Y-%m-%d')} (Close: {day_close:.4f})")

        # --- 1. Check Exits for Active Trades (based on previous day's H/L) ---
        # This logic assumes SL/TP are checked against the H/L of the day *after* the signal.
        # So, if a trade was opened based on data up to day `i-1` (predicting for day `i`),
        # we check SL/TP using day `i`'s H/L.

        position_closed_this_iteration = False # Flag to prevent immediate re-entry after counter-signal exit

        if current_position == 'LONG':
            if day_low <= stop_loss_price:
                print(f"EXIT LONG (SL Hit): Close at {stop_loss_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'LONG', 'entry': entry_price, 'exit': stop_loss_price, 'reason': 'SL'})
                current_position = None
                position_closed_this_iteration = True
            elif day_high >= take_profit_price:
                print(f"EXIT LONG (TP Hit): Close at {take_profit_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'LONG', 'entry': entry_price, 'exit': take_profit_price, 'reason': 'TP'})
                current_position = None
                position_closed_this_iteration = True
        elif current_position == 'SHORT':
            if day_high >= stop_loss_price:
                print(f"EXIT SHORT (SL Hit): Close at {stop_loss_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': stop_loss_price, 'reason': 'SL'})
                current_position = None
                position_closed_this_iteration = True
            elif day_low <= take_profit_price:
                print(f"EXIT SHORT (TP Hit): Close at {take_profit_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': take_profit_price, 'reason': 'TP'})
                current_position = None
                position_closed_this_iteration = True

        # --- 2. Feature Engineering & Prediction for NEXT Day ---
        # We use data up to and including `current_day_data` (row `i`)
        # The window of data for feature engineering ends at row `i`.
        data_for_features = historical_data.iloc[:i+1] # Includes current day

        features, current_atr = engineer_features(data_for_features, full_data_for_atr_calc=data_for_features)

        if features is None or current_atr is None or np.isnan(current_atr):
            print(f"Could not engineer features or ATR for {day_timestamp.strftime('%Y-%m-%d')}. Holding.")
            # If we are in a position, counter-signal exit check might still be possible if model makes a prediction
            # but for now, if features fail, we skip prediction.
            # Consider if a position should be closed if ATR becomes NaN (e.g. data issue)
            if current_position and (current_atr is None or np.isnan(current_atr)):
                 print(f"Warning: ATR is NaN. Cannot update SL/TP dynamically if needed. Current position: {current_position}")
            continue # Skip to next day

        # Predict probability for the NEXT day (i+1)
        # Model was trained on features to predict Target = (Close[t+1] > Close[t])
        pred_proba = model.predict_proba(features) # Output: [[prob_class_0, prob_class_1]]
        prob_up = pred_proba[0][1] # Probability of price going up (class 1)

        print(f"Prediction for { (day_timestamp + pd.Timedelta(days=1)).strftime('%Y-%m-%d') }: Prob(Up) = {prob_up:.4f}. Current ATR: {current_atr:.4f}")

        # --- 3. Counter-Signal Exit Logic (if still in position and not closed by SL/TP) ---
        if not position_closed_this_iteration: # only if not already closed by SL/TP
            if current_position == 'LONG' and prob_up < PROB_THRESHOLD_SELL: # Strong SELL signal
                print(f"EXIT LONG (Counter Signal): Close at {day_close:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Prob(Up): {prob_up:.4f}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'LONG', 'entry': entry_price, 'exit': day_close, 'reason': 'Counter'})
                current_position = None
                position_closed_this_iteration = True
            elif current_position == 'SHORT' and prob_up > PROB_THRESHOLD_BUY: # Strong BUY signal
                print(f"EXIT SHORT (Counter Signal): Close at {day_close:.4f} on {day_timestamp.strftime('%Y-%m-%d')}. Prob(Up): {prob_up:.4f}. Entry: {entry_price:.4f}")
                sim_trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': day_close, 'reason': 'Counter'})
                current_position = None
                position_closed_this_iteration = True

        # --- 4. Entry Logic (if no current position or position just closed by counter signal) ---
        if current_position is None: # Check if we can enter a new trade
            if prob_up > PROB_THRESHOLD_BUY:
                current_position = 'LONG'
                entry_price = day_close # Entry at the close of the current signal day
                stop_loss_price = entry_price - (ATR_MULTIPLIER_SL * current_atr)
                take_profit_price = entry_price + (ATR_MULTIPLIER_TP * current_atr)
                print(f"NEW TRADE: BUY EUR/USD at {entry_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}, SL: {stop_loss_price:.4f}, TP: {take_profit_price:.4f}, Lots: {FIXED_LOT_SIZE}")
            elif prob_up < PROB_THRESHOLD_SELL:
                current_position = 'SHORT'
                entry_price = day_close # Entry at the close of the current signal day
                stop_loss_price = entry_price + (ATR_MULTIPLIER_SL * current_atr)
                take_profit_price = entry_price - (ATR_MULTIPLIER_TP * current_atr)
                print(f"NEW TRADE: SELL EUR/USD at {entry_price:.4f} on {day_timestamp.strftime('%Y-%m-%d')}, SL: {stop_loss_price:.4f}, TP: {take_profit_price:.4f}, Lots: {FIXED_LOT_SIZE}")
            else:
                print("Signal is HOLD or not strong enough. No new trade.")
        elif current_position is not None:
             print(f"Holding {current_position} position. Entry: {entry_price:.4f}, SL: {stop_loss_price:.4f}, TP: {take_profit_price:.4f}")


    # --- End of Simulation Loop ---
    # If a position is still open at the end, we could choose to close it or report it as open
    if current_position is not None:
        last_close_price = historical_data['close'].iloc[-1]
        print(f"\nEnd of simulation. Closing open {current_position} position at market price: {last_close_price:.4f}")
        if current_position == 'LONG':
            sim_trades.append({'type': 'LONG', 'entry': entry_price, 'exit': last_close_price, 'reason': 'EoSim'})
        elif current_position == 'SHORT':
            sim_trades.append({'type': 'SHORT', 'entry': entry_price, 'exit': last_close_price, 'reason': 'EoSim'})
        current_position = None

    # --- Simulation Summary ---
    print("\n--- Simulation Summary ---")
    if not sim_trades:
        print("No trades were executed.")
        return

    num_trades = len(sim_trades)
    print(f"Total trades executed: {num_trades}")

    # Basic P/L calculation (ignores lot size for simplicity here, focuses on points)
    # For EUR/USD, 1 pip = 0.0001. Profit in pips.
    total_pips = 0
    for trade in sim_trades:
        pips = 0
        if trade['type'] == 'LONG':
            pips = (trade['exit'] - trade['entry']) * 10000 # (Exit - Entry) for LONG
        elif trade['type'] == 'SHORT':
            pips = (trade['entry'] - trade['exit']) * 10000 # (Entry - Exit) for SHORT
        total_pips += pips
        print(f"Trade: {trade['type']} | Entry: {trade['entry']:.4f} | Exit: {trade['exit']:.4f} | Pips: {pips:.1f} | Reason: {trade['reason']}")

    print(f"\nTotal Pips (approx, ignoring spread/commission/lot size): {total_pips:.2f} pips")
    # To make it more "real" with fixed lots:
    # Value per pip for EUR/USD with 0.02 lots on a standard account ($100,000 contract size)
    # (0.0001 / 1) * (2000) = $0.20 per pip (assuming USD account currency and EUR/USD is ~1.0)
    # This can vary based on quote currency and account currency.
    # For simplicity, let's just use pips.
    # If we wanted to approximate value:
    # Profit_USD = total_pips * PIP_VALUE_PER_LOT * LOT_SIZE (e.g. $10 per pip for 1 lot, so $0.2 for 0.02 lots)
    # PIP_VALUE_FOR_0_02_LOTS = 0.20 # Approx $0.20 per pip for 0.02 lots of EURUSD
    # print(f"Approximate P/L for {FIXED_LOT_SIZE} lots: ${total_pips * PIP_VALUE_FOR_0_02_LOTS:.2f}")


if __name__ == '__main__':
    # Load Model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_PATH} not found. Please train and save the model first.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Load Data
    historical_data = load_data(DATA_FILE)
    if historical_data is None:
        print(f"Error: Could not load data from {DATA_FILE}. Exiting.")
        exit()

    # Ensure dummy eur_usd_daily_1yr.csv from previous subtask has 'open', 'high', 'low', 'close'
    # If not, the load_data will fail or ATR/feature engineering might.
    # The dummy CSV created in the previous step has these.

    run_simulation(model, historical_data)
