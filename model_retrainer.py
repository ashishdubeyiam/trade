import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split # Not strictly needed if using chronological split
import os
import logging
from datetime import datetime
import shutil # For moving files

# Import refactored functions from eurusd_prediction_model.py
# Ensure eurusd_prediction_model.py is in the same directory or PYTHONPATH
try:
    from eurusd_prediction_model import (
        load_data as load_raw_data, # Alias to avoid confusion if we have other load_data
        get_features_and_target,
        train_lgbm_model,
        # Constants like LAG_DAYS, SMA_SHORT_PERIOD etc. are implicitly used by get_features_and_target
        # If these need to be configurable for the retrainer, they should be passed or managed differently.
        # For now, retrainer uses the same feature eng. params as defined in eurusd_prediction_model.py
    )
    # Import feature engineering parameters if they are indeed global in the other script and needed here
    # from eurusd_prediction_model import LAG_DAYS, SMA_SHORT_PERIOD, SMA_LONG_PERIOD, RSI_PERIOD # etc.
except ImportError:
    print("ERROR: Ensure eurusd_prediction_model.py is in the same directory or PYTHONPATH and is correctly refactored.")
    exit()


# --- Configuration ---
MASTER_DATA_FILE = 'eur_usd_master_data.csv'
LIVE_MODEL_FILE = 'live_model.joblib' # This will be the new name for the primary model
CANDIDATE_MODEL_FILE = 'candidate_model.joblib'
ARCHIVE_DIR = 'archived_models/'
LOG_FILE = 'model_retrainer.log'

VALIDATION_SET_SIZE = 30  # Number of recent days/records for validation
MIN_TRAIN_SET_SIZE = 200  # Minimum records needed for training after validation split
PRIMARY_METRIC = 'f1_score' # or 'accuracy' etc.

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


def evaluate_model(model, X_val, y_val):
    """Evaluates a model on the validation set and returns a dictionary of metrics."""
    if model is None or X_val is None or y_val is None or X_val.empty or y_val.empty:
        logger.error("Cannot evaluate model: Missing model, X_val, or y_val, or they are empty.")
        return None

    try:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] # Prob of class 1

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            # 'roc_auc': roc_auc_score(y_val, y_pred_proba) # Optional, needs y_pred_proba
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist() # Convert to list for logging/JSON
        }
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return None

def retrain_and_validate():
    logger.info("--- Starting Model Retraining and Validation Process ---")

    # 1. Load Data
    logger.info(f"Loading master data from: {MASTER_DATA_FILE}")
    raw_data = load_raw_data(MASTER_DATA_FILE) # Uses the imported function

    if raw_data is None:
        logger.error(f"Failed to load master data from {MASTER_DATA_FILE}. Exiting.")
        return

    if len(raw_data) < (VALIDATION_SET_SIZE + MIN_TRAIN_SET_SIZE):
        logger.error(f"Not enough data in {MASTER_DATA_FILE} for training and validation. "
                     f"Need {VALIDATION_SET_SIZE + MIN_TRAIN_SET_SIZE}, got {len(raw_data)}. Exiting.")
        return

    logger.info(f"Master data loaded. Total records: {len(raw_data)}")

    # 2. Prepare features and target (using the imported function)
    # This function internally handles feature engineering and target creation
    X_full, y_full = get_features_and_target(raw_data)

    if X_full is None or y_full is None:
        logger.error("Failed to engineer features or target from the raw data. Exiting.")
        return

    logger.info(f"Features and target processed. X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")

    # 3. Split Data (Chronological)
    if len(X_full) <= VALIDATION_SET_SIZE: # Ensure enough data for validation split itself
        logger.error(f"Processed data length ({len(X_full)}) is not greater than VALIDATION_SET_SIZE ({VALIDATION_SET_SIZE}). Cannot split.")
        return

    val_split_idx = len(X_full) - VALIDATION_SET_SIZE
    X_train = X_full.iloc[:val_split_idx]
    y_train = y_full.iloc[:val_split_idx]
    X_val = X_full.iloc[val_split_idx:]
    y_val = y_full.iloc[val_split_idx:]

    logger.info(f"Data split: Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    if len(X_train) < MIN_TRAIN_SET_SIZE:
        logger.error(f"Training set size ({len(X_train)}) is less than MIN_TRAIN_SET_SIZE ({MIN_TRAIN_SET_SIZE}). Exiting.")
        return
    if X_val.empty or y_val.empty:
        logger.error("Validation set is empty after split. Exiting.")
        return

    # 4. Train Candidate Model
    logger.info("Training candidate model...")
    candidate_model = train_lgbm_model(X_train, y_train) # Uses imported function

    if candidate_model is None:
        logger.error("Candidate model training failed. Exiting.")
        return

    try:
        joblib.dump(candidate_model, CANDIDATE_MODEL_FILE)
        logger.info(f"Candidate model trained and saved to {CANDIDATE_MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error saving candidate model to {CANDIDATE_MODEL_FILE}: {e}", exc_info=True)
        return # Cannot proceed without candidate model

    # 5. Evaluate Candidate Model
    logger.info("Evaluating candidate model on validation set...")
    candidate_metrics = evaluate_model(candidate_model, X_val, y_val)
    if candidate_metrics is None:
        logger.error("Failed to evaluate candidate model. Exiting.")
        # Optionally delete candidate model file here if it's invalid
        # try: os.remove(CANDIDATE_MODEL_FILE) except OSError: pass
        return
    logger.info(f"Candidate Model Metrics: {candidate_metrics}")

    # 6. Evaluate Live Model (if exists)
    live_model_metrics = None
    live_model_f1 = -1.0 # Default to worse than any valid F1

    if os.path.exists(LIVE_MODEL_FILE):
        logger.info(f"Live model {LIVE_MODEL_FILE} exists. Evaluating on validation set...")
        try:
            live_model = joblib.load(LIVE_MODEL_FILE)
            live_model_metrics = evaluate_model(live_model, X_val, y_val)
            if live_model_metrics:
                logger.info(f"Live Model Metrics: {live_model_metrics}")
                live_model_f1 = live_model_metrics.get(PRIMARY_METRIC, -1.0)
            else:
                logger.warning("Failed to evaluate live model, or evaluation returned no metrics.")
        except Exception as e:
            logger.error(f"Error loading or evaluating live model {LIVE_MODEL_FILE}: {e}", exc_info=True)
            # Proceed, considering live model performance as poor.
    else:
        logger.info(f"No live model found at {LIVE_MODEL_FILE}. Candidate model will be promoted if valid.")

    # 7. Promotion Decision
    candidate_model_f1 = candidate_metrics.get(PRIMARY_METRIC, -1.0)
    logger.info(f"Comparison: Candidate Model {PRIMARY_METRIC} = {candidate_model_f1:.4f}, Live Model {PRIMARY_METRIC} = {live_model_f1:.4f}")

    if candidate_model_f1 > live_model_f1:
        logger.info(f"Candidate model's {PRIMARY_METRIC} ({candidate_model_f1:.4f}) is better than live model's ({live_model_f1:.4f}). Promoting candidate.")

        # Create archive directory if it doesn't exist
        os.makedirs(ARCHIVE_DIR, exist_ok=True)

        # Archive current live model (if it exists)
        if os.path.exists(LIVE_MODEL_FILE):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(ARCHIVE_DIR, f"live_model_{timestamp_str}.joblib")
            try:
                shutil.move(LIVE_MODEL_FILE, archive_path)
                logger.info(f"Previous live model archived to {archive_path}")
            except Exception as e:
                logger.error(f"Error archiving previous live model from {LIVE_MODEL_FILE} to {archive_path}: {e}", exc_info=True)
                # Decide if this is critical enough to stop promotion. For now, proceed.

        # Promote candidate to live
        try:
            shutil.move(CANDIDATE_MODEL_FILE, LIVE_MODEL_FILE)
            logger.info(f"Candidate model {CANDIDATE_MODEL_FILE} promoted to live: {LIVE_MODEL_FILE}")
        except Exception as e:
            logger.error(f"Error promoting candidate model from {CANDIDATE_MODEL_FILE} to {LIVE_MODEL_FILE}: {e}", exc_info=True)
            # This is critical. The candidate was better but couldn't be moved.
            # State might be inconsistent here.
    else:
        logger.info(f"Candidate model's {PRIMARY_METRIC} ({candidate_model_f1:.4f}) is not better than live model's ({live_model_f1:.4f}). Live model retained.")
        # Optionally, delete or archive the non-promoted candidate model
        try:
            # Example: Move to a "failed_candidates" archive or just delete
            failed_candidate_path = os.path.join(ARCHIVE_DIR, f"failed_candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
            shutil.move(CANDIDATE_MODEL_FILE, failed_candidate_path)
            logger.info(f"Non-promoted candidate model moved to {failed_candidate_path}")
        except Exception as e:
            logger.warning(f"Could not move non-promoted candidate model {CANDIDATE_MODEL_FILE}: {e}", exc_info=True)

    logger.info("--- Model Retraining and Validation Process Finished ---")

if __name__ == '__main__':
    retrain_and_validate()
