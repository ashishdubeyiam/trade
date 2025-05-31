# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import numpy as np
import os

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    print("Make sure config.py is in the src/ directory or PYTHONPATH includes src")
    # Fallback for simple execution, assuming relative paths if config not found
    class config: # Basic fallback
        RAW_DATA_FILE = '../data/fraud_dataset.csv' # Adjust if necessary
        PROCESSED_DATA_DIR = '../data/processed'
        TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
        TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
        CATEGORICAL_FEATURES = ['type']
        FEATURES_TO_DROP_EXPERIMENT2 = ['nameOrig', 'nameDest']
        RANDOM_SEED = 42
        # Ensure target variable name is correct as per dataset
        TARGET_COLUMN = 'isFraud' # This is based on the proposal


def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at {file_path}")
        print("Please ensure the dataset is available at the specified path in config.py (RAW_DATA_FILE).")
        print("You might need to download it and place it in the 'data/' directory.")
        return None
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def inspect_data(df):
    """Prints basic data inspection information."""
    if df is None:
        return
    print("\n--- Data Inspection ---")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nInfo:")
    df.info()
    print("\nDescribe:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    # As per proposal, 'isFraud' is the target. Check its distribution.
    if config.TARGET_COLUMN in df.columns:
        print(f"\nTarget variable '{config.TARGET_COLUMN}' distribution:\n", df[config.TARGET_COLUMN].value_counts(normalize=True))
    else:
        print(f"WARNING: Target column '{config.TARGET_COLUMN}' not found in dataframe.")

    # Optional: Inspect timestamp column if it exists
    if hasattr(config, 'TIMESTAMP_COLUMN') and config.TIMESTAMP_COLUMN in df.columns:
        print(f"\n--- Timestamp Column ('{config.TIMESTAMP_COLUMN}') Inspection ---")
        print(f"Data type: {df[config.TIMESTAMP_COLUMN].dtype}")
        if len(df[config.TIMESTAMP_COLUMN].unique()) > 5:
            print(f"First 5 unique values: {df[config.TIMESTAMP_COLUMN].unique()[:5]}")
        else:
            print(f"Unique values: {df[config.TIMESTAMP_COLUMN].unique()}")


def handle_class_imbalance(df, target_column):
    """Handles class imbalance using undersampling as per the proposal."""
    if df is None or target_column not in df.columns:
        print("DataFrame is None or target column not found for imbalance handling.")
        return df

    print(f"\n--- Handling Class Imbalance (Undersampling for target '{target_column}') ---")
    fraudulent = df[df[target_column] == 1]
    non_fraudulent = df[df[target_column] == 0]

    print(f"Original counts: Fraudulent={len(fraudulent)}, Non-fraudulent={len(non_fraudulent)}")

    if len(fraudulent) == 0:
        print("No fraudulent samples found. Undersampling cannot be performed.")
        return df
    if len(non_fraudulent) < len(fraudulent):
        print("Non-fraudulent samples are less than or equal to fraudulent samples. No undersampling needed or data is unusual.")
        return df

    non_fraudulent_undersampled = resample(non_fraudulent,
                                         replace=False,
                                         n_samples=len(fraudulent), # Match minority class size
                                         random_state=config.RANDOM_SEED)

    undersampled_df = pd.concat([non_fraudulent_undersampled, fraudulent])
    print(f"After undersampling: Fraudulent={len(undersampled_df[undersampled_df[target_column] == 1])}, Non-fraudulent={len(undersampled_df[undersampled_df[target_column] == 0])}")
    print(f"Total samples after undersampling: {len(undersampled_df)}")
    return undersampled_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True) # Shuffle


def preprocess_features(df, experiment_type='all_features'):
    """
    Preprocesses features:
    - Drops specified columns for experiment 2.
    - Performs one-hot encoding for categorical features.
    - Scales numerical features.
    """
    if df is None:
        print("DataFrame is None, cannot preprocess features.")
        return None, None

    print(f"\n--- Preprocessing Features (Experiment: {experiment_type}) ---")

    X = df.drop(config.TARGET_COLUMN, axis=1, errors='ignore')
    y = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None

    if y is None:
        print(f"Target column '{config.TARGET_COLUMN}' not found. Cannot proceed with feature processing.")
        return None, None

    # --- Timestamp Feature Engineering ---
    if hasattr(config, 'TIMESTAMP_COLUMN') and config.TIMESTAMP_COLUMN in X.columns:
        print(f"Processing timestamp column: {config.TIMESTAMP_COLUMN}")
        X[config.TIMESTAMP_COLUMN] = pd.to_datetime(X[config.TIMESTAMP_COLUMN], errors='coerce')

        X['Hour_of_Day'] = X[config.TIMESTAMP_COLUMN].dt.hour
        X['Day_of_Week'] = X[config.TIMESTAMP_COLUMN].dt.dayofweek
        X['Month_of_Year'] = X[config.TIMESTAMP_COLUMN].dt.month

        X = X.drop(columns=[config.TIMESTAMP_COLUMN])
        print(f"Created time-based features: Hour_of_Day, Day_of_Week, Month_of_Year. Dropped {config.TIMESTAMP_COLUMN}.")
    else:
        print(f"Timestamp column '{getattr(config, 'TIMESTAMP_COLUMN', 'N/A')}' not found or not configured. Skipping time-based feature engineering.")

    # --- Drop Specified Columns ---
    if hasattr(config, 'COLUMNS_TO_DROP') and config.COLUMNS_TO_DROP:
        print(f"Dropping specified columns: {config.COLUMNS_TO_DROP}")
        X = X.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')
        print(f"Features after dropping specified columns: {X.columns.tolist()}")

    # --- Experiment-Specific Feature Dropping ---
    if experiment_type == 'selected_features':
        if hasattr(config, 'FEATURES_TO_DROP_EXPERIMENT2') and config.FEATURES_TO_DROP_EXPERIMENT2:
            print(f"Dropping features for selected_features experiment: {config.FEATURES_TO_DROP_EXPERIMENT2}")
            X = X.drop(columns=config.FEATURES_TO_DROP_EXPERIMENT2, errors='ignore')
            print(f"Features after dropping for experiment: {X.columns.tolist()}")
        else:
            print("No features configured to be dropped for 'selected_features' experiment or config attribute missing.")

    # --- Identify Feature Types After Transformations ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Ensure categorical features from config are actually present in X after transformations
    categorical_features = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]

    print(f"Numerical features identified after transformations: {numerical_features}")
    print(f"Categorical features identified after transformations (and present in X): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) not specified, though ideally all are handled
    )

    # Fit and transform the features
    # Important: Fit only on training data in a real pipeline. Here, for simplicity in this script,
    # we might fit on the whole X if used directly. In main.py, split first.
    print("Applying preprocessing (OneHotEncoding and Scaling)...")
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    # This part can be tricky with ColumnTransformer. We'll try to reconstruct.
    feature_names_out = []
    try:
        ohe_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)
        feature_names_out.extend(ohe_feature_names)
    except Exception as e:
        print(f"Could not get OHE feature names: {e}. Using generic names.")
        feature_names_out.extend([f"cat_{i}" for i in range(len(categorical_features) * len(X[categorical_features[0]].unique()) if categorical_features else 0)]) # Approximation

    # Add numerical feature names (which are scaled)
    scaled_numerical_features = [col for col in numerical_features if col not in categorical_features] # Ensure no overlap if a feature was miscategorized
    feature_names_out.extend(scaled_numerical_features)

    # Add remainder columns if any
    if preprocessor.remainder == 'passthrough' and hasattr(preprocessor, 'feature_names_in_'):
        num_processed_cols = len(ohe_feature_names if 'ohe_feature_names' in locals() else []) + len(scaled_numerical_features)
        # This logic for remainder names might need refinement based on actual ColumnTransformer behavior
        remainder_cols = [X.columns[i] for i in range(len(X.columns)) if X.columns[i] not in categorical_features and X.columns[i] not in scaled_numerical_features]
        feature_names_out.extend(remainder_cols)


    X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out if len(feature_names_out) == X_processed.shape[1] else None)

    if X_processed_df.columns is None and X_processed.shape[1] > 0 :
         print(f"Warning: Column names could not be perfectly reconstructed. Processed X has {X_processed.shape[1]} columns.")
    else:
        print(f"Processed feature names: {X_processed_df.columns.tolist()}")
        print("Preprocessing complete.")

    return X_processed_df, y


def split_data(X, y, test_size=0.2):
    """Splits data into training and testing sets."""
    if X is None or y is None:
        print("X or y is None, cannot split data.")
        return None, None, None, None

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y if y is not None and len(np.unique(y)) > 1 else None
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, train_file, test_file):
    """Saves the processed training and testing dataframes along with labels."""
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Some data components are None, cannot save.")
        return

    print("\n--- Saving Processed Data ---")
    try:
        # Ensure directories exist
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        os.makedirs(os.path.dirname(test_file), exist_ok=True)

        # Combine features and target for saving
        train_df_to_save = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df_to_save = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        train_df_to_save.to_csv(train_file, index=False)
        test_df_to_save.to_csv(test_file, index=False)
        print(f"Processed training data saved to: {train_file}")
        print(f"Processed testing data saved to: {test_file}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

# Example of how this script might be run (for testing purposes)
if __name__ == '__main__':
    print("Running data_preprocessing.py as a script (for testing purposes)...")

    # 1. Load data
    raw_df = load__data(config.RAW_DATA_FILE) # Make sure this file exists in data/

    if raw_df is not None:
        # 2. Inspect data
        inspect_data(raw_df)

        # 3. Handle class imbalance
        # Ensure TARGET_COLUMN is correctly set in config and present in the data
        if config.TARGET_COLUMN in raw_df.columns:
            balanced_df = handle_class_imbalance(raw_df, config.TARGET_COLUMN)
        else:
            print(f"Skipping imbalance handling as target column '{config.TARGET_COLUMN}' is not found.")
            balanced_df = raw_df # Or handle error as appropriate

        # 4. Preprocess features (Example: 'all_features' experiment)
        # Note: In a full pipeline, fit ColumnTransformer on train data only.
        # Here, for a standalone script test, we might process the balanced_df.
        X_processed, y_processed = preprocess_features(balanced_df, experiment_type='all_features')

        if X_processed is not None and y_processed is not None:
            # 5. Split data
            X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

            # 6. Save processed data (optional, for inspection or use by other scripts)
            if X_train is not None: # Check if split was successful
                 save_processed_data(X_train, X_test, y_train, y_test,
                                    config.TRAIN_DATA_FILE, config.TEST_DATA_FILE)
        else:
            print("Feature processing failed, cannot proceed to split or save.")
    else:
        print("Data loading failed. Preprocessing steps skipped.")

    print("\nData preprocessing script finished.")
