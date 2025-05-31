# src/main.py

import os
import argparse
import json # Added for loading/saving metrics json

# Attempt to import project modules
# This structure assumes main.py is in src/ and can import other src/ modules directly
try:
    import config
    import data_preprocessing
    import train_models
    import evaluate_models
except ModuleNotFoundError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running main.py from a location where 'src' is discoverable,")
    print("or that 'src' is in your PYTHONPATH.")
    print("For example, run from the 'fraud_detection_project' directory as 'python src/main.py'")
    # Provide very basic fallbacks if run directly and modules not found,
    # though this is not ideal for a structured project.
    if 'config' not in globals():
        class config:
            RAW_DATA_FILE = '../data/fraud_dataset.csv'
            TRAIN_DATA_FILE = '../data/processed/train.csv'
            TEST_DATA_FILE = '../data/processed/test.csv'
            RESULTS_DIR = '../results' # Added for fallback
            EVALUATION_METRICS_FILE = '../results/evaluation_metrics.json'
            TARGET_COLUMN = 'isFraud'
            LOGISTIC_REGRESSION_MODEL_PATH = '../models/logistic_regression_model.joblib'
            RANDOM_FOREST_MODEL_PATH = '../models/random_forest_model.joblib'
            DECISION_TREE_MODEL_PATH = '../models/decision_tree_model.joblib'
            # Ensure necessary dirs for fallback
            os.makedirs('../data/processed', exist_ok=True)
            os.makedirs('../models', exist_ok=True)
            os.makedirs('../results/confusion_matrices', exist_ok=True) # from evaluate_models fallback
            os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results dir itself


def run_pipeline(experiment_type='all_features'):
    """
    Runs the full pipeline:
    1. Load and preprocess data for the specified experiment type.
    2. Train models.
    3. Evaluate models.
    """
    print(f"--- Starting Fraud Detection Pipeline: Experiment Type '{experiment_type}' ---")

    # --- 1. Data Preprocessing ---
    print("\n--- Step 1: Data Preprocessing ---")
    # Ensure data_preprocessing uses its own config for paths initially
    raw_df = data_preprocessing.load_data(config.RAW_DATA_FILE)
    if raw_df is None:
        print("Halting pipeline: Raw data could not be loaded.")
        return

    data_preprocessing.inspect_data(raw_df)

    if config.TARGET_COLUMN not in raw_df.columns:
        print(f"Halting pipeline: Target column '{config.TARGET_COLUMN}' not found in raw data.")
        return

    balanced_df = data_preprocessing.handle_class_imbalance(raw_df, config.TARGET_COLUMN)
    if balanced_df is None:
        print("Halting pipeline: Class imbalance handling failed.")
        return

    X_processed, y_processed = data_preprocessing.preprocess_features(balanced_df, experiment_type=experiment_type)
    if X_processed is None or y_processed is None:
        print("Halting pipeline: Feature preprocessing failed.")
        return

    X_train, X_test, y_train, y_test = data_preprocessing.split_data(X_processed, y_processed)
    if X_train is None or y_train is None or X_test is None or y_test is None: # Check if split was successful
        print("Halting pipeline: Data splitting failed or returned None.")
        return

    # Save the processed data (can be useful for inspection or direct use later)
    # These paths are read by the current train_models/evaluate_models if not passed data directly
    train_file = config.TRAIN_DATA_FILE
    test_file = config.TEST_DATA_FILE
    data_preprocessing.save_processed_data(X_train, X_test, y_train, y_test, train_file, test_file)
    print("Data preprocessing complete.")

    # --- 2. Model Training ---
    # Pass X_train, y_train directly to training functions
    print("\n--- Step 2: Model Training ---")
    train_models.train_logistic_regression(X_train, y_train)
    train_models.train_random_forest(X_train, y_train)
    train_models.train_decision_tree(X_train, y_train)
    train_models.train_ann_model(X_train, y_train) # Added ANN training call
    print("Model training complete.")

    # --- 3. Model Evaluation ---
    # Pass X_test, y_test directly to evaluation functions
    print("\n--- Step 3: Model Evaluation ---")
    all_model_metrics = {}
    models_to_evaluate = {
        "Logistic Regression": config.LOGISTIC_REGRESSION_MODEL_PATH,
        "Random Forest": config.RANDOM_FOREST_MODEL_PATH,
        "Decision Tree": config.DECISION_TREE_MODEL_PATH,
        "ANN": config.ANN_MODEL_PATH # Added ANN
    }

    for model_name, model_path in models_to_evaluate.items():
        model = evaluate_models.load_model(model_path) # evaluate_models still loads the model file
        if model:
            metrics = evaluate_models.evaluate_model(model, X_test, y_test, model_name) # Pass test data
            if metrics:
                all_model_metrics[f"{model_name}_{experiment_type}"] = metrics
        else:
            print(f"Skipping evaluation for {model_name} (experiment: {experiment_type}) as model could not be loaded.")

    if all_model_metrics:
        eval_file_path = config.EVALUATION_METRICS_FILE

        existing_metrics = {}
        if os.path.exists(eval_file_path):
            try:
                with open(eval_file_path, 'r') as f:
                    existing_metrics = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing metrics file {eval_file_path}. It might be corrupted. Overwriting.")

        existing_metrics.update(all_model_metrics)

        try:
            with open(eval_file_path, 'w') as f:
                json.dump(existing_metrics, f, indent=4)
            print(f"All evaluation metrics for experiment '{experiment_type}' saved/updated in: {eval_file_path}")
        except Exception as e:
            print(f"Error saving metrics to JSON for experiment '{experiment_type}': {e}")
    else:
        print(f"No metrics were generated to save for experiment '{experiment_type}'.")

    print(f"--- Fraud Detection Pipeline: Experiment Type '{experiment_type}' Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the fraud detection pipeline.")
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['all_features', 'selected_features'],
        default='all_features',
        help="Type of experiment to run: 'all_features' or 'selected_features' (based on proposal's Experiment 1 and 2)."
    )
    args = parser.parse_args()

    # Check if data file exists before running
    # This check should ideally use the config from the imported module, not the fallback.
    # The fallback config within main.py is only for extreme cases where module imports fail.
    try:
        data_file_to_check = config.RAW_DATA_FILE
    except NameError: # config module itself might not have been imported
        # Use the fallback path for the check if config module failed to load
        print("Warning: config module not loaded, using fallback path for raw data check.")
        data_file_to_check = '../data/fraud_dataset.csv' # Fallback path

    if not os.path.exists(data_file_to_check):
        print(f"ERROR: Raw data file not found at {data_file_to_check}")
        print("Please ensure the dataset is available and the path in src/config.py (RAW_DATA_FILE) is correct.")
        print("You might need to download it and place it in the 'data/' directory and name it appropriately.")
    else:
        run_pipeline(experiment_type=args.experiment)

    # Example: To run both experiments as per proposal
    # print("\nRunning pipeline for 'all_features' experiment...")
    # run_pipeline(experiment_type='all_features')
    # print("\nRunning pipeline for 'selected_features' experiment...")
    # run_pipeline(experiment_type='selected_features')
