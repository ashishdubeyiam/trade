import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import joblib
import numpy as np # Will be needed for RSI calculation

# --- Configuration ---
DATA_FILE = 'eur_usd_daily_1yr.csv'
MODEL_OUTPUT_FILE = 'lgbm_eurusd_direction_model.joblib'
LAG_DAYS = 3
SMA_SHORT_PERIOD = 5
SMA_LONG_PERIOD = 10
RSI_PERIOD = 14
TEST_SIZE = 0.2

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        # Ensure 'timestamp' is parsed as datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            print("Warning: 'timestamp' column not found. Data should be sorted chronologically.")
        # Assuming 'close' column contains the closing prices
        if 'close' not in df.columns:
            print("Error: 'close' column not found in the data. This is required for feature engineering.")
            return None
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
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_features(df):
    """Engineers features for the prediction model."""
    if df is None or 'close' not in df.columns:
        print("Error: Dataframe is None or 'close' column is missing for feature engineering.")
        return None

    df_feat = df.copy()

    # 1. Lagged closing prices
    for i in range(1, LAG_DAYS + 1):
        df_feat[f'Close_t-{i}'] = df_feat['close'].shift(i)

    # 2. Simple Moving Averages (SMA)
    df_feat[f'SMA_{SMA_SHORT_PERIOD}'] = df_feat['close'].rolling(window=SMA_SHORT_PERIOD).mean()
    df_feat[f'SMA_{SMA_LONG_PERIOD}'] = df_feat['close'].rolling(window=SMA_LONG_PERIOD).mean()

    # 3. Relative Strength Index (RSI)
    df_feat[f'RSI_{RSI_PERIOD}'] = calculate_rsi(df_feat['close'], period=RSI_PERIOD)

    # 4. Handle NaN values (by dropping rows with any NaN in crucial columns)
    # This is a simple approach. More sophisticated methods like imputation could be used.
    # We need to define which columns are crucial for prediction.
    # For now, let's consider all newly created features and 'close'

    # Identify feature columns (excluding 'timestamp' and original OHLC if not used directly)
    # For this model, we'll use the engineered features.
    # 'open', 'high', 'low' from original data might also be useful but are not in requirements for features.

    # Note: The print statements are kept for now if this script is run directly.
    # For library use, these might be converted to logging or removed.
    # print(f"\nShape before dropping NaNs for features: {df_feat.shape}")

    # Determine feature columns for NaN drop. We need to be careful not to drop rows if only 'Target' (created later) is NaN.
    # The crucial NaNs for feature engineering are in the feature columns themselves.
    feature_columns_for_nan_check = [f'Close_t-{i}' for i in range(1, LAG_DAYS + 1)] + \
                                    [f'SMA_{SMA_SHORT_PERIOD}', f'SMA_{SMA_LONG_PERIOD}', f'RSI_{RSI_PERIOD}']
    df_feat.dropna(subset=feature_columns_for_nan_check, inplace=True)
    # print(f"Shape after dropping NaNs from feature calculation: {df_feat.shape}")

    if df_feat.empty:
        # print("Error: DataFrame is empty after dropping NaN values from feature engineering.")
        return None # Return None if feature engineering results in empty dataframe

    return df_feat # Return dataframe with features, not yet reset_index, target not created

def create_target_variable(df_with_features):
    """Creates the binary target variable 'Target' on a dataframe that already has features."""
    if df_with_features is None or 'close' not in df_with_features.columns:
        # print("Error: Dataframe is None or 'close' column is missing for target variable creation.")
        return None

    df_target = df_with_features.copy()
    # Target: 1 if next day's close is higher than current day's close, 0 otherwise
    df_target['Target'] = (df_target['close'].shift(-1) > df_target['close']).astype(int)

    # Drop the last row because it will have a NaN for 'Target' (no next day to compare)
    df_target.dropna(subset=['Target'], inplace=True)
    # print(f"Shape after dropping last row for Target NaN: {df_target.shape}")

    if df_target.empty:
        # print("Error: DataFrame is empty after creating target variable and dropping NaN.")
        return None

    return df_target.reset_index(drop=True)

def get_features_and_target(raw_df):
    """
    Orchestrates feature engineering and target variable creation.
    Returns X (features) and y (target) as pandas DataFrames/Series.
    """
    if raw_df is None:
        print("Error: Raw DataFrame is None in get_features_and_target.")
        return None, None

    df_with_features = engineer_features(raw_df)
    if df_with_features is None:
        print("Error: Feature engineering failed.")
        return None, None

    final_data_with_target = create_target_variable(df_with_features)
    if final_data_with_target is None:
        print("Error: Target variable creation failed.")
        return None, None

    feature_columns = [col for col in final_data_with_target.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'Target']]
                       # Add any other original columns that are not features

    X = final_data_with_target[feature_columns]
    y = final_data_with_target['Target']

    if X.empty or y.empty:
        print("Error: Feature set X or target y is empty after processing.")
        return None, None

    return X, y

def train_lgbm_model(X_train, y_train, model_params=None):
    """
    Trains a LightGBM classifier model.
    model_params: dict, optional parameters for LGBMClassifier.
    """
    if X_train is None or y_train is None or X_train.empty or y_train.empty:
        print("Error: Training data (X_train or y_train) is None or empty.")
        return None

    default_params = {'random_state': 42}
    if model_params:
        default_params.update(model_params)

    model = lgb.LGBMClassifier(**default_params)
    try:
        model.fit(X_train, y_train)
        print("Model training completed successfully.")
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

if __name__ == '__main__':
    print("--- Running EUR/USD Prediction Model Training (Direct Execution) ---")
    data = load_data(DATA_FILE)

    if data is not None:
        print(f"\nOriginal Data loaded. Shape: {data.shape}")

        X, y = get_features_and_target(data)

        if X is not None and y is not None:
            print(f"\nFeatures (X) and Target (y) created. X shape: {X.shape}, y shape: {y.shape}")
            print("\nSample of features (X):")
            print(X.head())
            print("\nTarget (y) distribution:")
            print(y.value_counts(normalize=True))

            # --- Data Splitting (Chronological) ---
            if len(X) * TEST_SIZE < 1: # Ensure test set is at least 1 sample
                 print(f"Warning: Dataset too small for TEST_SIZE={TEST_SIZE}. Adjusting TEST_SIZE or using all for training.")
                 # In this case, could default to using a very small test set or all for training if appropriate
                 # For now, let's proceed, train_test_split might handle it or fail if X_test is empty

            split_idx = int(len(X) * (1 - TEST_SIZE))
            if split_idx == 0 and len(X) > 1 : # Avoid empty training set if data exists
                print(f"Warning: Calculated split_idx is 0 with {len(X)} samples. Setting training set to at least 1 sample.")
                split_idx = 1 # Ensure at least one sample for training if possible, though likely still too small

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            print(f"\nTraining set size: {X_train.shape[0]}")
            print(f"Test set size: {X_test.shape[0]}")

            if X_train.empty or y_train.empty:
                print("Error: Training set is empty after split. Cannot train model.")
            elif X_test.empty or y_test.empty:
                 print("Warning: Test set is empty after split. Model will be trained but not evaluated in this script's main block.")
                 # Train the model
                 model = train_lgbm_model(X_train, y_train)
                 if model:
                    try:
                        joblib.dump(model, MODEL_OUTPUT_FILE)
                        print(f"\nTrained model saved to {MODEL_OUTPUT_FILE} (Test set was empty).")
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                # Train the model
                model = train_lgbm_model(X_train, y_train)

                if model:
                    # --- Model Evaluation ---
                    print("\nEvaluating model on the test set...")
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)

                    print(f"\nAccuracy: {accuracy:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1-score: {f1:.4f}")
                    print("\nConfusion Matrix:")
                    print(cm)

                    # --- Save Model ---
                    try:
                        joblib.dump(model, MODEL_OUTPUT_FILE)
                        print(f"\nTrained model saved to {MODEL_OUTPUT_FILE}")
                    except Exception as e:
                        print(f"Error saving model: {e}")
        else:
            print("Exiting due to issues in feature/target creation.")
    else:
        print("Exiting due to data loading issues.")
