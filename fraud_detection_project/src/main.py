# src/main.py

import os
import argparse
import json # Added for loading/saving metrics json
import sys # Make sure sys is imported

# Ensure the 'src' directory (where main.py and its sibling modules reside) is in sys.path
# This helps Python find modules like config, data_preprocessing, etc., when main.py is run as a script.
current_script_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_script_path)

if src_directory not in sys.path:
    sys.path.insert(0, src_directory)

# Direct imports after sys.path modification
import config
import data_preprocessing
import train_models
import evaluate_models

def run_pipeline(experiment_type='all_features', input_csv_path=None):
    """
    Runs the full pipeline:
    1. Load and preprocess data for the specified experiment type.
    2. Train models.
    3. Evaluate models.
    """
    print(f"--- Starting Fraud Detection Pipeline: Experiment Type '{experiment_type}' ---")

    # --- 1. Data Preprocessing ---
    print("\n--- Step 1: Data Preprocessing ---")

    current_data_file = input_csv_path if input_csv_path else config.RAW_DATA_FILE
    if not current_data_file or not os.path.exists(current_data_file):
        print(f"ERROR: Data file not found at specified path: {current_data_file}")
        print("Please ensure the file exists, either via --input-file argument or RAW_DATA_FILE in config.py.")
        return # Or raise an error
    print(f"Attempting to load data from: {current_data_file}")
    raw_df = data_preprocessing.load_data(current_data_file)

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
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help="Path to the input CSV data file (overrides config.RAW_DATA_FILE)."
    )
    args = parser.parse_args()

    # Determine data file path: command-line arg > config default
    # The actual loading and detailed error for not found is handled in run_pipeline now.
    # This pre-check is just to guide the user if they run main.py directly.
    data_file_to_run = args.input_file if args.input_file else config.RAW_DATA_FILE

    # Simplified pre-run check, detailed check is now inside run_pipeline
    if not data_file_to_run or not os.path.exists(data_file_to_run):
        print(f"ERROR: Raw data file not found.")
        if args.input_file: # If --input-file was provided and not found
            print(f"  Checked path from --input-file: {args.input_file}")
        # If --input-file was not provided OR it was but not found, AND config.RAW_DATA_FILE is different and relevant
        if not args.input_file or \
           (args.input_file and not os.path.exists(args.input_file) and \
            config.RAW_DATA_FILE and config.RAW_DATA_FILE != args.input_file):
            print(f"  Checked path from config.RAW_DATA_FILE: {config.RAW_DATA_FILE}")
        print("Please ensure the dataset is available and the path is correct.")
    else:
        run_pipeline(experiment_type=args.experiment, input_csv_path=data_file_to_run)

    # Example: To run both experiments as per proposal
    # print("\nRunning pipeline for 'all_features' experiment...")
    # run_pipeline(experiment_type='all_features')
    # print("\nRunning pipeline for 'selected_features' experiment...")
    # run_pipeline(experiment_type='selected_features')
