# src/evaluate_models.py

import pandas as pd
import joblib
import os
import json
import numpy as np # Added numpy import
# import tensorflow as tf # Commented out for optional import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    print("Make sure config.py is in the src/ directory or PYTHONPATH includes src")
    # Fallback for simple execution
    class config: # Basic fallback
        TEST_DATA_FILE = '../data/processed/test.csv' # Adjust if necessary
        MODEL_DIR = '../models'
        RESULTS_DIR = '../results'
        CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
        EVALUATION_METRICS_FILE = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
        LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
        RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
        DECISION_TREE_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_model.joblib')
        ANN_MODEL_PATH = os.path.join(MODEL_DIR, 'ann_model.h5') # Added for ANN
        TARGET_COLUMN = 'isFraud' # Ensure this matches your target column name

        # Create fallback directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)

# Optional TensorFlow import
try:
    import tensorflow as tf
    # TENSORFLOW_AVAILABLE = True # Not strictly needed here if tf is not used otherwise
except ImportError:
    pass # Silently pass if not found, as it's not critical for non-ANN model evaluation
    # TENSORFLOW_AVAILABLE = False

def load_test_data(test_file_path):
    """Loads processed test data."""
    print(f"Loading processed test data from: {test_file_path}")
    if not os.path.exists(test_file_path):
        print(f"ERROR: Processed test data file not found at {test_file_path}")
        print("Please ensure data_preprocessing.py has been run successfully and created the test set.")
        return None, None
    try:
        df = pd.read_csv(test_file_path)
        X_test = df.drop(columns=[config.TARGET_COLUMN], errors='ignore')
        y_test = df[config.TARGET_COLUMN]
        print("Processed test data loaded successfully.")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading processed test data: {e}")
        return None, None

def load_model(model_path):
    """Loads a trained model from a file."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please ensure train_models.py has been run successfully and saved the model.")
        return None
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculates various evaluation metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Could not calculate ROC AUC (e.g. only one class in y_true): {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist() # Convert numpy array to list for JSON serialization

    # From confusion matrix: TN, FP, FN, TP
    # Ensure cm is 2x2, otherwise these specific metrics might be ill-defined
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        # Specificity = TN / (TN + FP)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # Geometric Mean = sqrt(Recall * Specificity)
        metrics['geometric_mean'] = (metrics['recall'] * metrics['specificity'])**0.5 if metrics['recall'] is not None and metrics['specificity'] is not None else None
    else: # Handle cases where CM might not be 2x2 (e.g. if only one class predicted or present in y_true)
        print(f"Confusion matrix is not 2x2 ({cm.shape}), cannot reliably calculate TN, FP, FN, TP, Specificity, G-Mean directly from ravel().")
        # You might want to set these to None or handle them based on the specific shape of cm
        metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = None
        metrics['specificity'] = None
        metrics['geometric_mean'] = None


    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"  {key}:")
            for row in value:
                print(f"    {row}")
        else:
            print(f"  {key}: {value}")
    return metrics

def plot_confusion_matrix(cm, model_name):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')

    # Ensure directory exists
    os.makedirs(config.CONFUSION_MATRIX_DIR, exist_ok=True)

    plot_path = os.path.join(config.CONFUSION_MATRIX_DIR, f'{model_name.lower().replace(" ", "_")}_cm.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to: {plot_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a single model and returns its metrics."""
    if model is None or X_test is None or y_test is None:
        print(f"Cannot evaluate {model_name} due to missing model or test data.")
        return None

    print(f"\n--- Evaluating {model_name} ---")
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        if 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], list): # Use the list form
             # Convert list back to numpy array for plotting
            cm_array = np.array(metrics['confusion_matrix']) # Corrected: Use numpy directly
            plot_confusion_matrix(cm_array, model_name)
        return metrics
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

# Example of how this script might be run (for testing purposes)
if __name__ == '__main__':
    print("Running evaluate_models.py as a script (for testing purposes)...")

    # 1. Load test data
    # This assumes data_preprocessing.py has run and created test.csv
    X_test, y_test = load_test_data(config.TEST_DATA_FILE)

    all_metrics = {}

    if X_test is not None and y_test is not None:
        print(f"Using target column: {config.TARGET_COLUMN}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # 2. Load models and evaluate
        model_paths = {
            "Logistic Regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
            "Random Forest": config.RANDOM_FOREST_MODEL_PATH,
            "Decision Tree": config.DECISION_TREE_MODEL_PATH,
            "ANN": config.ANN_MODEL_PATH # Added ANN
        }

        for model_name, model_path in model_paths.items():
            model = load_model(model_path)
            if model:
                metrics = evaluate_model(model, X_test, y_test, model_name)
                if metrics:
                    all_metrics[model_name] = metrics
            else:
                print(f"Skipping evaluation for {model_name} as model could not be loaded.")

        # 3. Save all metrics to a JSON file
        if all_metrics:
            try:
                # Ensure results directory exists
                os.makedirs(config.RESULTS_DIR, exist_ok=True)
                with open(config.EVALUATION_METRICS_FILE, 'w') as f:
                    json.dump(all_metrics, f, indent=4)
                print(f"\nAll evaluation metrics saved to: {config.EVALUATION_METRICS_FILE}")
            except Exception as e:
                print(f"Error saving metrics to JSON: {e}")
        else:
            print("No metrics were generated to save.")

    else:
        print("Processed test data not loaded. Model evaluation skipped.")
        print(f"Please ensure '{config.TEST_DATA_FILE}' exists and is correctly formatted.")

    print("\nModel evaluation script finished.")
