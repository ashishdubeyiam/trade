import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import shutil # For moving files

# Import functions from intraday_model_trainer.py
# Ensure intraday_model_trainer.py is in the same directory or PYTHONPATH
try:
    from intraday_model_trainer import (
        load_labeled_data,
        prepare_data_for_training,
        split_data_chronological,
        train_classification_model, # Assuming 'logistic_regression' is default or configurable
        evaluate_classification_model,
        # save_trained_model, # We'll use joblib.dump directly for candidate, then shutil.move
        TARGET_COLUMN as DEFAULT_TARGET_COLUMN, # Import default target name
        FEATURE_COLUMNS_TO_USE as DEFAULT_FEATURE_COLUMNS, # Import default features
        RANDOM_STATE as DEFAULT_RANDOM_STATE
    )
    logger_trainer_exists = True
except ImportError as e:
    print(f"ERROR: Could not import from intraday_model_trainer.py: {e}. Ensure it's in PYTHONPATH.")
    # Fallback or exit if critical functions are missing
    logger_trainer_exists = False
    # Define placeholders if you want the script to be partially runnable for structure checks
    def load_labeled_data(csv_filepath): return None
    def prepare_data_for_training(df, tc, fc): return None, None
    def split_data_chronological(X,y,ts): return None,None,None,None
    def train_classification_model(X,y,rt,rs): return None
    def evaluate_classification_model(m,X,y): return None
    DEFAULT_TARGET_COLUMN = 'target'
    DEFAULT_FEATURE_COLUMNS = []
    DEFAULT_RANDOM_STATE = 42


# --- Configuration Constants ---
LABELED_M5_DATA_CSV = 'eur_usd_m5_labeled_data.csv'
LIVE_MODEL_FILE = 'live_intraday_m5_model.joblib'
CANDIDATE_MODEL_FILE = 'candidate_intraday_m5_model.joblib'
ARCHIVE_DIR = 'archived_intraday_models/'
LOG_FILE = 'intraday_model_retrainer.log'

VALIDATION_SET_SIZE_PERCENT = 0.20
MIN_TRAIN_RECORDS_THRESHOLD = 1000
PRIMARY_METRIC_FOR_PROMOTION = 'f1_score'
TARGET_COLUMN = DEFAULT_TARGET_COLUMN
FEATURE_COLUMNS_TO_USE = DEFAULT_FEATURE_COLUMNS
RANDOM_STATE = DEFAULT_RANDOM_STATE

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

if not logger_trainer_exists:
     logger.critical("Failed to import functions from intraday_model_trainer.py. Retrainer cannot operate.")


def run_retraining_pipeline():
    logger.info("--- Starting Intraday M5 Model Retraining and Validation Pipeline ---")
    if not logger_trainer_exists:
        logger.error("Essential training functions not imported. Aborting.")
        return

    # 1. Load Full Labeled Data
    logger.info(f"Loading labeled M5 data from: {LABELED_M5_DATA_CSV}")
    full_labeled_df = load_labeled_data(csv_filepath=LABELED_M5_DATA_CSV)

    if full_labeled_df is None or full_labeled_df.empty:
        logger.error(f"Failed to load labeled M5 data or file is empty from {LABELED_M5_DATA_CSV}. Exiting.")
        return
    logger.info(f"Loaded {len(full_labeled_df)} records from master labeled data.")

    # 2. Prepare Data (Feature Engineering already done, this step handles NaN and selection)
    X_full, y_full = prepare_data_for_training(
        full_labeled_df,
        target_column_name=TARGET_COLUMN,
        feature_cols=FEATURE_COLUMNS_TO_USE
    )
    if X_full is None or y_full is None or X_full.empty or y_full.empty:
        logger.error("Data preparation failed (X or y is empty/None). Exiting.")
        return
    logger.info(f"Data prepared. X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

    # Check total usable records against combined threshold for split
    if len(X_full) < MIN_TRAIN_RECORDS_THRESHOLD / (1-VALIDATION_SET_SIZE_PERCENT) : # Ensure enough data for train+val
        logger.error(f"Not enough usable records ({len(X_full)}) after preparation to meet minimum training and validation sizes. "
                     f"Need approx {MIN_TRAIN_RECORDS_THRESHOLD / (1-VALIDATION_SET_SIZE_PERCENT):.0f}. Exiting.")
        return

    # 3. Split Data into Training and Validation (Chronological)
    X_train, X_val, y_train, y_val = split_data_chronological(
        X_full, y_full, test_size_ratio=VALIDATION_SET_SIZE_PERCENT
    )
    if X_train is None or X_val is None or X_train.empty or X_val.empty : # y_train/y_val emptiness checked by X
        logger.error("Data splitting failed or resulted in empty train/validation sets. Exiting.")
        return

    logger.info(f"Data split: X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    if len(X_train) < MIN_TRAIN_RECORDS_THRESHOLD:
        logger.error(f"Training set ({len(X_train)} records) is smaller than MIN_TRAIN_RECORDS_THRESHOLD ({MIN_TRAIN_RECORDS_THRESHOLD}). Exiting.")
        return

    # 4. Train Candidate Model
    logger.info("Training candidate M5 model (Logistic Regression)...")
    candidate_model = train_classification_model(
        X_train, y_train,
        model_type='logistic_regression', # Explicitly stating, though it's default in imported func
        random_state=RANDOM_STATE
    )
    if candidate_model is None:
        logger.error("Candidate model training failed. Exiting.")
        return
    try:
        joblib.dump(candidate_model, CANDIDATE_MODEL_FILE)
        logger.info(f"Candidate model trained and saved to {CANDIDATE_MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error saving candidate model to {CANDIDATE_MODEL_FILE}: {e}", exc_info=True)
        return

    # 5. Evaluate Candidate Model
    logger.info("Evaluating candidate model on validation set...")
    candidate_metrics = evaluate_classification_model(candidate_model, X_val, y_val)
    if candidate_metrics is None:
        logger.error("Failed to evaluate candidate model. Exiting without promotion.")
        # Consider deleting CANDIDATE_MODEL_FILE here if evaluation fails catastrophically
        return
    logger.info(f"Candidate Model Metrics on Validation Set: {candidate_metrics}")

    # 6. Evaluate Live Model (if exists)
    live_model_metric_value = -1.0  # Default to a very bad score

    if os.path.exists(LIVE_MODEL_FILE):
        logger.info(f"Live model {LIVE_MODEL_FILE} exists. Loading and evaluating on validation set...")
        try:
            live_model = joblib.load(LIVE_MODEL_FILE)
            live_metrics = evaluate_classification_model(live_model, X_val, y_val)
            if live_metrics and PRIMARY_METRIC_FOR_PROMOTION in live_metrics:
                live_model_metric_value = live_metrics[PRIMARY_METRIC_FOR_PROMOTION]
                logger.info(f"Live Model Metrics on Validation Set: {live_metrics}")
            else:
                logger.warning(f"Could not retrieve '{PRIMARY_METRIC_FOR_PROMOTION}' for live model or evaluation failed.")
        except Exception as e:
            logger.error(f"Error loading or evaluating live model {LIVE_MODEL_FILE}: {e}", exc_info=True)
            # Live model is compromised or evaluation failed; proceed as if its performance is poor.
    else:
        logger.info(f"No live model found at {LIVE_MODEL_FILE}. Candidate will be promoted if its metrics are valid.")

    # 7. Promotion Decision Logic
    candidate_metric_value = candidate_metrics.get(PRIMARY_METRIC_FOR_PROMOTION, -1.0)
    logger.info(f"Promotion Check: Candidate Model {PRIMARY_METRIC_FOR_PROMOTION} = {candidate_metric_value:.4f} vs. "
                f"Live Model {PRIMARY_METRIC_FOR_PROMOTION} = {live_model_metric_value:.4f}")

    if candidate_metric_value > live_model_metric_value:
        logger.info(f"Candidate model performing better. Promoting candidate model to live.")
        os.makedirs(ARCHIVE_DIR, exist_ok=True) # Ensure archive directory exists

        if os.path.exists(LIVE_MODEL_FILE):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(ARCHIVE_DIR, f"live_intraday_m5_model_{timestamp_str}.joblib")
            try:
                shutil.move(LIVE_MODEL_FILE, archive_path)
                logger.info(f"Previous live model archived to {archive_path}")
            except Exception as e:
                logger.error(f"CRITICAL: Error archiving current live model from {LIVE_MODEL_FILE} to {archive_path}: {e}", exc_info=True)
                # This is a critical failure point. If archiving fails, we might not want to overwrite the live model.
                # For now, we'll log and the promotion will still attempt. Consider a more robust transaction here.

        try:
            shutil.move(CANDIDATE_MODEL_FILE, LIVE_MODEL_FILE)
            logger.info(f"Candidate model {CANDIDATE_MODEL_FILE} successfully promoted to live: {LIVE_MODEL_FILE}")
        except Exception as e:
            logger.critical(f"CRITICAL: Error promoting candidate model from {CANDIDATE_MODEL_FILE} to {LIVE_MODEL_FILE}: {e}", exc_info=True)
            # State is now potentially inconsistent: old live model might be archived, candidate failed to move.
    else:
        logger.info("Candidate model did not outperform live model. Live model retained.")
        try:
            # Option: Delete candidate or archive it
            failed_candidate_archive_path = os.path.join(ARCHIVE_DIR, f"failed_candidate_m5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
            shutil.move(CANDIDATE_MODEL_FILE, failed_candidate_archive_path)
            logger.info(f"Non-promoted candidate model {CANDIDATE_MODEL_FILE} archived to {failed_candidate_archive_path}")
        except Exception as e:
            logger.warning(f"Could not archive or delete non-promoted candidate model {CANDIDATE_MODEL_FILE}: {e}", exc_info=True)

    logger.info("--- Intraday M5 Model Retraining and Validation Pipeline Finished ---")


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

    if not logger_trainer_exists:
        logger.critical("Exiting due to missing intraday_model_trainer.py functions.")
    else:
        # Create dummy labeled data for testing if main file doesn't exist
        if not os.path.exists(LABELED_M5_DATA_CSV):
            logger.info(f"Creating dummy labeled M5 data file: {LABELED_M5_DATA_CSV} for testing.")
            # Ensure enough data for split and min training threshold
            num_rows_test = int((MIN_TRAIN_RECORDS_THRESHOLD / (1 - VALIDATION_SET_SIZE_PERCENT)) + 50)
            base_time = datetime.now(timezone.utc) - timedelta(minutes=5*num_rows_test)
            test_data_payload = {
                'timestamp': pd.date_range(start=base_time, periods=num_rows_test, freq='5min', tz='UTC'),
            }
            _df = pd.DataFrame(test_data_payload)
            for col in FEATURE_COLUMNS_TO_USE: # Use the imported default list
                _df[col] = np.random.rand(num_rows_test) * 10
            _df[TARGET_COLUMN] = np.random.randint(0, 2, num_rows_test)
            _df.loc[_df.index[:20], FEATURE_COLUMNS_TO_USE[0]] = np.nan # Simulate some NaNs
            _df.to_csv(LABELED_M5_DATA_CSV, index=False)

        run_retraining_pipeline()
        logger.info(f"Retraining test run finished. Live model: {LIVE_MODEL_FILE}, Candidate: {CANDIDATE_MODEL_FILE} (may be archived/deleted).")
        logger.info(f"Input labeled data was {LABELED_M5_DATA_CSV}.")
