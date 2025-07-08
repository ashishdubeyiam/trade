# src/config.py

import os

# --- Project Root ---
# Assuming this config.py file is in 'fraud_detection_project/src/'
# and the project root is 'fraud_detection_project/'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Placeholder for the dataset path.
# User will need to provide the actual dataset and update this path.
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_FILE = os.path.join(DATA_DIR, 'dummy_fraud_dataset.csv') # Example filename, user to confirm/update
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# --- Model Output Paths ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
DECISION_TREE_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_model.joblib')
ANN_MODEL_PATH = os.path.join(MODEL_DIR, 'ann_model.h5') # For TensorFlow/Keras

# --- Model Parameters ---
# Example: Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# Example: ANN parameters
ANN_EPOCHS = 50
ANN_BATCH_SIZE = 32

# --- Feature Engineering ---
TIMESTAMP_COLUMN = 'Timestamp'
COLUMNS_TO_DROP = ['Transaction_ID', 'User_ID']
CATEGORICAL_FEATURES = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method', 'IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']
FEATURES_TO_DROP_EXPERIMENT2 = ['nameOrig', 'nameDest'] # Features to drop for experiment 2

# --- Evaluation ---
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EVALUATION_METRICS_FILE = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions') # Path for saving prediction CSVs

# --- Other ---
RANDOM_SEED = 42 # For reproducibility
TARGET_COLUMN = 'Fraud_Label'


# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True) # Ensure predictions directory is created

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Models Directory: {MODEL_DIR}")
print(f"Results Directory: {RESULTS_DIR}")
print(f"Predictions Directory: {PREDICTIONS_DIR}") # Added print for verification
