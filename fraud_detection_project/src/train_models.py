# src/train_models.py

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf # Added for ANN

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    print("Make sure config.py is in the src/ directory or PYTHONPATH includes src")
    # Fallback for simple execution
    class config: # Basic fallback
        TRAIN_DATA_FILE = '../data/processed/train.csv' # Adjust if necessary
        MODEL_DIR = '../models'
        LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
        RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
        DECISION_TREE_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_model.joblib')
        ANN_MODEL_PATH = os.path.join(MODEL_DIR, 'ann_model.h5') # Added for ANN
        RF_N_ESTIMATORS = 100
        RF_MAX_DEPTH = 10
        RF_RANDOM_STATE = 42
        RANDOM_SEED = 42
        TARGET_COLUMN = 'isFraud' # Ensure this matches your target column name

        # Create fallback directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)


def load_processed_data(train_file_path):
    """Loads processed training data."""
    print(f"Loading processed training data from: {train_file_path}")
    if not os.path.exists(train_file_path):
        print(f"ERROR: Processed training data file not found at {train_file_path}")
        print("Please ensure data_preprocessing.py has been run successfully.")
        return None, None
    try:
        df = pd.read_csv(train_file_path)
        X_train = df.drop(columns=[config.TARGET_COLUMN], errors='ignore')
        y_train = df[config.TARGET_COLUMN]
        print("Processed training data loaded successfully.")
        return X_train, y_train
    except Exception as e:
        print(f"Error loading processed training data: {e}")
        return None, None

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    if X_train is None or y_train is None:
        print("Training data not available for Logistic Regression.")
        return None
    print("\n--- Training Logistic Regression Model ---")
    try:
        model = LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000) # Increased max_iter for convergence
        model.fit(X_train, y_train)
        print("Logistic Regression model trained successfully.")
        joblib.dump(model, config.LOGISTIC_REGRESSION_MODEL_PATH)
        print(f"Logistic Regression model saved to: {config.LOGISTIC_REGRESSION_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error training Logistic Regression model: {e}")
        return None

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Classifier model."""
    if X_train is None or y_train is None:
        print("Training data not available for Random Forest.")
        return None
    print("\n--- Training Random Forest Model ---")
    try:
        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RF_RANDOM_STATE
        )
        model.fit(X_train, y_train)
        print("Random Forest model trained successfully.")
        joblib.dump(model, config.RANDOM_FOREST_MODEL_PATH)
        print(f"Random Forest model saved to: {config.RANDOM_FOREST_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error training Random Forest model: {e}")
        return None

def train_decision_tree(X_train, y_train):
    """Trains a Decision Tree Classifier model."""
    if X_train is None or y_train is None:
        print("Training data not available for Decision Tree.")
        return None
    print("\n--- Training Decision Tree Model ---")
    try:
        model = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)
        print("Decision Tree model trained successfully.")
        joblib.dump(model, config.DECISION_TREE_MODEL_PATH)
        print(f"Decision Tree model saved to: {config.DECISION_TREE_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error training Decision Tree model: {e}")
        return None

def train_ann_model(X_train, y_train):
    """Placeholder for training an Artificial Neural Network model."""
    if X_train is None or y_train is None:
        print("Training data not available for ANN.")
        return None
    print("\n--- Training ANN Model (Placeholder) ---")
    print("ANN model training placeholder. Full implementation requires network architecture details.")
    # Placeholder for model saving. For a real Keras model, you'd use model.save()
    # Using joblib.dump for None maintains consistency with other placeholders if absolutely needed,
    # but ideally, this path would only be written to if a model was actually trained.
    try:
        # Example: model.save(config.ANN_MODEL_PATH) if an actual Keras model 'model' existed
        # For now, to signify a placeholder and avoid Keras-specific save for a non-existent model:
        with open(config.ANN_MODEL_PATH, 'w') as f:
            f.write("ANN model placeholder - not a trained model.")
        print(f"ANN model placeholder saved to: {config.ANN_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving ANN model placeholder: {e}")
    return None

# Example of how this script might be run (for testing purposes)
if __name__ == '__main__':
    print("Running train_models.py as a script (for testing purposes)...")

    # 1. Load processed training data
    # This assumes data_preprocessing.py has run and created train.csv
    X_train, y_train = load_processed_data(config.TRAIN_DATA_FILE)

    if X_train is not None and y_train is not None:
        # 2. Train models
        print(f"Using target column: {config.TARGET_COLUMN}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        lr_model = train_logistic_regression(X_train, y_train)
        rf_model = train_random_forest(X_train, y_train)
        dt_model = train_decision_tree(X_train, y_train)
        ann_model = train_ann_model(X_train, y_train) # Added ANN call

        if lr_model:
            print("Logistic Regression model training initiated.")
        if rf_model:
            print("Random Forest model training initiated.")
        if dt_model:
            print("Decision Tree model training initiated.")
        if ann_model is None: # ann_model is always None from placeholder
            print("ANN model training (placeholder) initiated.")
    else:
        print("Processed training data not loaded. Model training skipped.")
        print(f"Please ensure '{config.TRAIN_DATA_FILE}' exists and is correctly formatted.")

    print("\nModel training script finished.")
