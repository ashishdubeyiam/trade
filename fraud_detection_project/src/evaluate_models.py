# src/evaluate_models.py

import pandas as pd
import joblib
import os
import json
import numpy as np # Added numpy import
# import tensorflow as tf # No longer needed directly here if ANN loading is handled by joblib/placeholder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    # This assumes evaluate_models.py is in src/, and config.py is also in src/
    import config
except ModuleNotFoundError:
    print("Attempting to import 'config' failed. Ensure config.py is in the 'src/' directory or adjust PYTHONPATH.")
    print("Using fallback configuration for standalone execution.")
    # Fallback for simple execution if config.py is not found
    class config: # Basic fallback
        PROJECT_ROOT_FALLBACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        DATA_DIR = os.path.join(PROJECT_ROOT_FALLBACK, 'data')
        PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
        TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

        MODEL_DIR = os.path.join(PROJECT_ROOT_FALLBACK, 'models')
        RESULTS_DIR = os.path.join(PROJECT_ROOT_FALLBACK, 'results')
        CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
        PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions') # Added PREDICTIONS_DIR
        EVALUATION_METRICS_FILE = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')

        LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
        RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
        DECISION_TREE_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree_model.joblib')
        ANN_MODEL_PATH = os.path.join(MODEL_DIR, 'ann_model.h5') # Path for ANN model
        TARGET_COLUMN = 'Fraud_Label' # Standardized target column name

        # Create fallback directories if they don't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
        os.makedirs(PREDICTIONS_DIR, exist_ok=True) # Ensure predictions dir is created for fallback


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
        return None
    try:
        # Generic loading, assuming joblib for scikit-learn models.
        # For ANN, if it's a Keras model, specific loading might be needed if not a placeholder.
        # If ANN_MODEL_PATH points to a dummy file, this will likely fail gracefully or load None.
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
             # This is a placeholder for actual Keras model loading if ANN is implemented
             # For now, if it's just a dummy text file, joblib.load will fail, caught by except.
            print(f"Attempting to load Keras model from {model_path}. This may fail if it's a placeholder or not a Keras model.")
            # from tensorflow.keras.models import load_model as load_keras_model
            # model = load_keras_model(model_path) # Uncomment if using actual Keras models
            # For placeholder, let it fall to generic joblib load or fail
            model = joblib.load(model_path) # This will fail for a non-joblib Keras model
        else:
            model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}. Might be a placeholder or incompatible format.")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculates various evaluation metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    if y_pred_proba is not None and len(np.unique(y_true)) > 1: # ROC AUC needs at least two classes in y_true
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Could not calculate ROC AUC (e.g. only one class in y_true or y_pred_proba issue): {e}")
            metrics['roc_auc'] = None # Or 'N/A'
    else:
        metrics['roc_auc'] = None # Or 'N/A' if only one class or no proba

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix_components_tn_fp_fn_tp'] = cm.ravel().tolist() # Save as a flat list for easier JSON storage if always 2x2

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['geometric_mean'] = (metrics['recall'] * metrics['specificity'])**0.5 if metrics['recall'] is not None and metrics['specificity'] is not None else 0.0
    else:
        print(f"Confusion matrix is not 2x2 ({cm.shape}), cannot reliably calculate TN, FP, FN, TP, Specificity, G-Mean directly from ravel(). Setting to None.")
        metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = None
        metrics['specificity'] = None
        metrics['geometric_mean'] = None

    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        if key == 'confusion_matrix_components_tn_fp_fn_tp':
            print(f"  {key} (TN, FP, FN, TP): {value}")
        else:
            print(f"  {key}: {value}")
    return metrics

def plot_confusion_matrix(cm_array, model_name): # cm_array is expected to be a numpy array
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    os.makedirs(config.CONFUSION_MATRIX_DIR, exist_ok=True)
    plot_path = os.path.join(config.CONFUSION_MATRIX_DIR, f'{model_name.lower().replace(" ", "_").replace("[^a-zA-Z0-9_]", "")}_cm.png')
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
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class
            except Exception as e_proba:
                print(f"Could not get predict_proba for {model_name}: {e_proba}. Proceeding without probabilities.")
                y_pred_proba = None

        # Save predictions along with features and true labels
        if isinstance(X_test, pd.DataFrame):
            predictions_df = X_test.copy()
        else:
            print("Warning: X_test is not a DataFrame. Predictions CSV will have generic column names.")
            predictions_df = pd.DataFrame(X_test)

        predictions_df['true_label'] = y_test.values
        predictions_df['predicted_label'] = y_pred

        if y_pred_proba is not None:
            predictions_df['predicted_probability_fraud'] = y_pred_proba
        else:
            predictions_df['predicted_probability_fraud'] = np.nan

        safe_filename_base = model_name.lower().replace(" ", "_")
        safe_filename_base = "".join(c if c.isalnum() or c == "_" else "" for c in safe_filename_base)
        predictions_filename = f"predictions_{safe_filename_base}.csv"

        predictions_filepath = os.path.join(config.PREDICTIONS_DIR, predictions_filename)

        try:
            # No need to os.makedirs here if config.py handles it, but good for fallback safety
            os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
            predictions_df.to_csv(predictions_filepath, index=False)
            print(f"Predictions for {model_name} saved to: {predictions_filepath}")
        except AttributeError:
             print(f"Error saving predictions for {model_name}: config.PREDICTIONS_DIR not found. Ensure config.py is correctly set up.")
        except Exception as e_save:
            print(f"Error saving predictions for {model_name}: {e_save}")

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # Plot confusion matrix using the components if available
        if metrics.get('tn') is not None: # Check if TN,FP,FN,TP were calculated
            cm_array_for_plot = np.array([
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']]
            ])
            plot_confusion_matrix(cm_array_for_plot, model_name)
        elif 'confusion_matrix_components_tn_fp_fn_tp' in metrics: # Fallback if only ravelled list is there
             #This case should ideally be handled by ensuring tn,fp,fn,tp are populated correctly
            cm_flat_list = metrics['confusion_matrix_components_tn_fp_fn_tp']
            if len(cm_flat_list) == 4:
                 cm_array_for_plot = np.array(cm_flat_list).reshape(2,2)
                 plot_confusion_matrix(cm_array_for_plot, model_name)
            else:
                 print(f"Could not reconstruct 2x2 confusion matrix for plotting for {model_name}.")
        else:
            print(f"Confusion matrix components not available in metrics for {model_name}, skipping plot.")

        return metrics
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

if __name__ == '__main__':
    print("Running evaluate_models.py as a script (for testing purposes)...")
    X_test, y_test = load_test_data(config.TEST_DATA_FILE)
    all_metrics = {}

    if X_test is not None and y_test is not None:
        print(f"Using target column: {config.TARGET_COLUMN}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        model_paths = {
            "Logistic Regression_all_features": config.LOGISTIC_REGRESSION_MODEL_PATH,
            "Random Forest_all_features": config.RANDOM_FOREST_MODEL_PATH,
            "Decision Tree_all_features": config.DECISION_TREE_MODEL_PATH,
            "ANN_all_features": config.ANN_MODEL_PATH
        }
        # Ensure model names in `main.py` for `all_metrics` keys match this format if used by UI

        for model_name_key, model_path in model_paths.items():
            print(f"\nProcessing model: {model_name_key} from path: {model_path}")
            model = load_model(model_path)
            if model:
                # Use model_name_key for consistency in naming outputs
                metrics = evaluate_model(model, X_test, y_test, model_name_key)
                if metrics:
                    all_metrics[model_name_key] = metrics
            else:
                print(f"Skipping evaluation for {model_name_key} as model could not be loaded.")
                # Add placeholder metrics for models that fail to load, so UI can still list them
                all_metrics[model_name_key] = {
                    'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A',
                    'roc_auc': 'N/A', 'tn': 'N/A', 'fp': 'N/A', 'fn': 'N/A', 'tp': 'N/A',
                    'specificity': 'N/A', 'geometric_mean': 'N/A', 'error': 'Model loading failed'
                }


        if all_metrics:
            try:
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
