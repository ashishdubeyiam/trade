## Explanation of `fraud_detection_project/src/evaluate_models.py`

The `evaluate_models.py` script is the final analytical stage in the `fraud_detection_project`'s machine learning pipeline. Its primary function is to assess the performance of the previously trained machine learning models using the unseen test dataset. It calculates a variety of performance metrics, generates visual aids like confusion matrices, and saves these evaluation outputs for review and interpretation.

**1. Overall Purpose:**

The script is designed to:
*   Load the preprocessed test data.
*   Load the serialized (trained) machine learning models.
*   For each model, make predictions on the test data.
*   Calculate a comprehensive suite of classification performance metrics.
*   Generate and save confusion matrix plots for each model.
*   Aggregate and save all calculated metrics into a structured JSON file, which serves as the main quantitative output of the pipeline.

**2. Key Libraries Used:**

*   **`pandas`:** Used in the standalone execution mode by `load_test_data` to read the processed test data CSV.
*   **`joblib`:** Used by `load_model` to deserialize and load the saved scikit-learn model objects.
*   **`json`:** Used to save the aggregated performance metrics dictionary into a JSON file.
*   **`numpy`:** Used for numerical operations, particularly for manipulating the confusion matrix array (e.g., extracting TN, FP, FN, TP).
*   **`matplotlib.pyplot`:** Used for creating and managing figures and plots, specifically for confusion matrices.
*   **`seaborn`:** Used for creating more visually appealing heatmaps for the confusion matrices.
*   **`sklearn.metrics`:** This module from scikit-learn is heavily used for calculating various performance metrics:
    *   `accuracy_score`
    *   `precision_score`
    *   `recall_score`
    *   `f1_score`
    *   `roc_auc_score`
    *   `confusion_matrix`

**3. Core Functions and their Logic:**

*   **`load_test_data(test_file_path: str) -> tuple[pd.DataFrame, pd.Series]`:**
    *   **Purpose for Standalone Execution:** Enables the script to be run independently. It loads the processed test data from the CSV file specified by `test_file_path` (typically `config.TEST_DATA_FILE`).
    *   **Logic:** Reads the CSV into a pandas DataFrame, then separates it into features `X_test` and the target variable `y_test`.
    *   **Bypass by `main.py`:** In the integrated pipeline orchestrated by `main.py`, `X_test` and `y_test` are already available in memory (from `data_preprocessing.py`) and are passed directly to relevant functions, making this load redundant.

*   **`load_model(model_path: str) -> object`:**
    *   Takes `model_path` (from `config.py`) as input.
    *   Uses `joblib.load(model_path)` to load the serialized model object.
    *   **Error Handling:** Includes `try-except` blocks to handle potential issues:
        *   `FileNotFoundError`: If the model file at `model_path` does not exist.
        *   Other exceptions (e.g., `joblib.externals.loky.process_executor.TerminatedWorkerError`, `EOFError`, `pickle.UnpicklingError`): Catches errors that might occur if the file is not a valid joblib file or is corrupted. This is particularly relevant for the placeholder `ann_model.h5` which is not a joblib file; this function would likely fail to load it, and the error handling would prevent a crash, allowing the script to (ideally) skip or note the failure for that specific model.
    *   Returns the loaded model object or `None` if loading fails.

*   **`calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict`:**
    *   **Inputs:**
        *   `y_true`: True labels from the test set.
        *   `y_pred`: Predicted labels from the model.
        *   `y_pred_proba`: Predicted probabilities for the positive class (typically from `model.predict_proba(X_test)[:, 1]`), required for ROC-AUC score. Defaults to `None`.
    *   **Metrics Calculation:**
        *   `accuracy = accuracy_score(y_true, y_pred)`
        *   `precision = precision_score(y_true, y_pred, zero_division=0)` (sets precision to 0 if no positive predictions are made)
        *   `recall = recall_score(y_true, y_pred, zero_division=0)` (sets recall to 0 if no actual positives exist or none are predicted)
        *   `f1 = f1_score(y_true, y_pred, zero_division=0)` (sets F1 to 0 under same conditions as precision/recall)
        *   `roc_auc = roc_auc_score(y_true, y_pred_proba)` if `y_pred_proba` is provided and valid; otherwise, it might be set to "N/A" or a default value like 0.0.
    *   **Confusion Matrix Components:**
        *   `cm = confusion_matrix(y_true, y_pred)`
        *   Extracts True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) from the confusion matrix `cm`. Handles cases where `cm` might not be 2x2 (e.g., if the model predicts only one class, `cm.ravel()` is used to safely extract values).
    *   **Additional Derived Metrics:**
        *   `specificity = TN / (TN + FP)` if `(TN + FP) > 0` else 0.0.
        *   `geometric_mean = np.sqrt(recall * specificity)` if `recall` and `specificity` are valid.
    *   Returns a dictionary containing all these calculated metric names and their values.

*   **`plot_confusion_matrix(cm: np.ndarray, model_name: str, results_dir: str = config.CONFUSION_MATRIX_DIR)`:**
    *   Takes the confusion matrix NumPy array `cm`, the `model_name` string, and the output directory path as input.
    *   Uses `seaborn.heatmap()` to create a plot of the confusion matrix, annotating cells with the counts (TN, FP, FN, TP) and including labels for axes and a title.
    *   Saves the plot as a PNG image file (e.g., `logistic_regression_cm.png`) in the specified `results_dir`. The filename includes the `model_name`.
    *   Uses `plt.close()` to close the plot figure and free up memory, which is important when generating multiple plots in a loop.

*   **`evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict`:**
    *   Orchestrates the evaluation process for a single loaded `model`.
    *   **Predictions:**
        *   `y_pred = model.predict(X_test)`: Gets discrete class predictions.
        *   `y_pred_proba = model.predict_proba(X_test)[:, 1]`: Attempts to get probability estimates for the positive class. Includes a `try-except AttributeError` block because some models (or malformed/placeholder models like the dummy ANN) might not have a `predict_proba` method. If it fails, `y_pred_proba` is set to `None`.
    *   Calls `calculate_metrics(y_test, y_pred, y_pred_proba)` to get the metrics dictionary.
    *   Calls `plot_confusion_matrix(metrics_dict['confusion_matrix_components'], model_name)` to generate and save the plot (assuming `calculate_metrics` returns the raw `cm` array as `confusion_matrix_components` or similar).
    *   Returns the `metrics_dict`.

**4. Workflow (`if __name__ == '__main__':` block):**

The script's standalone execution mode (`python src/evaluate_models.py`) typically performs:
1.  Loads the test data (`X_test`, `y_test`) using `load_test_data(config.TEST_DATA_FILE)`.
2.  Initializes an empty dictionary `all_metrics = {}`.
3.  Defines a list of model names and their corresponding paths from `config.py` (e.g., `models_to_evaluate = [("Logistic Regression", config.LOGISTIC_REGRESSION_MODEL_PATH), ...]`).
4.  Iterates through this list:
    *   Prints which model is being evaluated.
    *   Loads the model using `load_model(model_path)`.
    *   If the model is loaded successfully:
        *   Calls `evaluate_model(model, X_test, y_test, model_name)` to get its performance metrics.
        *   Stores these metrics in the `all_metrics` dictionary, keyed by a descriptive name (e.g., `Logistic Regression_all_features`).
    *   If model loading fails (e.g., for the ANN placeholder or a corrupted file), it prints an error message and skips evaluation for that model.
5.  After evaluating all models, saves the `all_metrics` dictionary to `config.EVALUATION_METRICS_FILE` as a JSON file using `json.dump()`.
6.  Prints a confirmation that evaluation is complete and metrics are saved.

**5. Interaction with `main.py`:**

When `main.py` orchestrates the pipeline:
*   `X_test` and `y_test` (from `data_preprocessing.py`) are already in memory.
*   `main.py` iterates through the trained models (which it might have just finished training or knows their paths from `config.py`). For each model object already loaded (or loaded by `main.py` itself):
    *   It directly calls `evaluate_model(loaded_model_object, X_test_df, y_test_series, model_name_string)`.
*   The `load_test_data()` function within `evaluate_models.py` is thus bypassed.
*   `main.py` would then collect the metrics from each `evaluate_model` call and handle the saving of the aggregated `all_metrics` JSON.

**6. Dependencies and Interactions:**

*   **`config.py`:** Crucial for obtaining:
    *   Path to the test data CSV file (`config.TEST_DATA_FILE`) for standalone mode.
    *   Paths to all trained model files (e.g., `config.LOGISTIC_REGRESSION_MODEL_PATH`).
    *   Output paths for results (`config.RESULTS_DIR`, `config.EVALUATION_METRICS_FILE`, `config.CONFUSION_MATRIX_DIR`).
*   **`train_models.py`:** Consumes the serialized model files (e.g., `.joblib` files) that were created and saved by `train_models.py`.
*   **Input Data (from `data_preprocessing.py`):**
    *   Uses the processed test data (`X_test`, `y_test`). In standalone mode, it reads this from `test.csv`; in the integrated pipeline, it receives these as arguments from `main.py`.
*   **Outputs:**
    *   **`evaluation_metrics.json`:** A JSON file saved to the path specified by `config.EVALUATION_METRICS_FILE`. This is the primary quantitative output, containing a dictionary of metrics for each evaluated model. This file is subsequently read by `fraud_detection_ui/app.py` to display results in the web interface.
    *   **Confusion Matrix Images:** PNG image files for each model's confusion matrix, saved in the directory specified by `config.CONFUSION_MATRIX_DIR`.

In summary, `evaluate_models.py` is responsible for rigorously testing the trained models, quantifying their performance using a wide range of metrics, and generating outputs that are essential for understanding model efficacy and for communicating results through the UI.
