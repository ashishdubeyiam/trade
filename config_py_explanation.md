## Explanation of `fraud_detection_project/src/config.py`

The `config.py` script within the `fraud_detection_project/src/` directory serves as a centralized configuration hub for the entire machine learning pipeline. Its primary purpose is to enhance modularity, maintainability, and ease of modification by consolidating all critical settings, paths, and parameters into a single, easily accessible file.

**1. Overall Purpose:**

By defining all configurable aspects of the project in one place, `config.py` allows developers and users to:
*   Quickly understand and modify project settings without searching through multiple scripts.
*   Ensure consistency in paths and parameters across different modules of the pipeline.
*   Adapt the pipeline to different datasets or experimental setups by changing values in this file.
*   Reduce hardcoding of values within the core logic scripts, making them more reusable and cleaner.

**2. Key Libraries Used:**

*   **`os`:** This standard Python library is used extensively for operating system-dependent functionalities, primarily for path manipulation (e.g., joining path components, creating absolute paths) and directory creation.

**3. Key Configuration Variables and Sections:**

The `config.py` script typically defines several categories of configuration variables:

*   **Project Root Directory (`PROJECT_ROOT`):**
    *   This variable dynamically determines the absolute path to the root directory of the `fraud_detection_project`.
    *   It is usually calculated using `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, which navigates two levels up from the `src` directory (where `config.py` resides) to reach the project's root. This makes the paths robust to where the project is cloned or executed from.

*   **Data Paths:** These define the locations for input, intermediate, and processed data.
    *   `DATA_DIR`: Path to the main data directory (e.g., `PROJECT_ROOT / 'data'`).
    *   `RAW_DATA_DIR`: Path to the directory containing raw input data (e.g., `DATA_DIR / 'raw'`).
    *   `RAW_DATA_FILE`: Full path to the primary raw dataset CSV file (e.g., `RAW_DATA_DIR / 'synthetic_fraud_dataset.csv'` or `dummy_fraud_dataset.csv`). **Crucially, for integration with the UI, the `main.py` script needs to be adapted to potentially override this path with the file path provided via the `--input-file` command-line argument from the UI.**
    *   `PROCESSED_DATA_DIR`: Path to the directory where processed data (e.g., training and testing sets) will be saved (e.g., `DATA_DIR / 'processed'`).
    *   `TRAIN_DATA_FILE`: Full path for saving the processed training data CSV file (e.g., `PROCESSED_DATA_DIR / 'train.csv'`).
    *   `TEST_DATA_FILE`: Full path for saving the processed testing data CSV file (e.g., `PROCESSED_DATA_DIR / 'test.csv'`).

*   **Model Output Paths:** These specify where trained models will be serialized and saved.
    *   `MODEL_DIR`: Path to the directory for storing trained model files (e.g., `PROJECT_ROOT / 'models'`).
    *   Individual Model File Paths: Specific paths for each model, typically constructed using `MODEL_DIR`. Examples:
        *   `LOGISTIC_REGRESSION_MODEL_PATH = MODEL_DIR / 'logistic_regression_model.joblib'`
        *   `RANDOM_FOREST_MODEL_PATH = MODEL_DIR / 'random_forest_model.joblib'`
        *   `DECISION_TREE_MODEL_PATH = MODEL_DIR / 'decision_tree_model.joblib'`
        *   `ANN_MODEL_PATH = MODEL_DIR / 'ann_model.h5'` (even as a placeholder, its path is defined).

*   **Model Parameters:** These define hyperparameters for the machine learning models. While extensive tuning would use dedicated scripts, `config.py` often holds default or baseline parameters.
    *   Examples from `implementation_guide.md` context:
        *   `RF_N_ESTIMATORS = 100` (Number of trees in Random Forest)
        *   `RF_MAX_DEPTH = 10` (Maximum depth of trees in Random Forest)
        *   `LR_MAX_ITER = 1000` (Maximum iterations for Logistic Regression solver)
        *   `ANN_EPOCHS = 10` (Placeholder epochs for ANN)
        *   `ANN_BATCH_SIZE = 32` (Placeholder batch size for ANN)
        *   `TEST_SIZE = 0.2` (Proportion of data to allocate to the test set)

*   **Feature Engineering Configuration:** Variables controlling how features are processed.
    *   `TARGET_COLUMN = 'Fraud_Label'` (Name of the target variable column in the dataset).
    *   `TIMESTAMP_COLUMN = 'Timestamp'` (Name of the timestamp column, if present, for time-based feature engineering).
    *   `COLUMNS_TO_DROP = ['Transaction_ID', 'User_ID', 'Account_Number']` (List of columns to be removed before feature processing, if they exist).
    *   `CATEGORICAL_FEATURES = ['Transaction_Type', 'Payment_Method', 'Device_Type']` (List of column names to be treated as categorical features for encoding, e.g., One-Hot Encoding).
    *   `FEATURES_TO_DROP_EXPERIMENT2 = ['Feature_X', 'Feature_Y']` (Specific features to drop for the 'selected_features' experiment type, if defined).

*   **Evaluation Output Paths:** Locations for saving model evaluation results.
    *   `RESULTS_DIR`: Path to the main directory for storing results (e.g., `PROJECT_ROOT / 'results'`).
    *   `EVALUATION_METRICS_FILE`: Full path to the JSON file where performance metrics for all models will be saved (e.g., `RESULTS_DIR / 'evaluation_metrics.json'`). This file is read by the UI.
    *   `CONFUSION_MATRIX_DIR`: Path to the directory where confusion matrix plots will be saved as images (e.g., `RESULTS_DIR / 'confusion_matrices'`).

*   **Other Settings:**
    *   `RANDOM_SEED = 42`: A seed value used for random operations (e.g., train/test split, model initialization, undersampling) to ensure reproducibility of results.

**4. Directory Creation Logic:**

To prevent errors if output directories do not exist, `config.py` typically includes logic to create them automatically at the start of a pipeline run. This is often achieved using:
```python
# Example:
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
```
The `exist_ok=True` argument ensures that `os.makedirs` does not raise an error if the directory already exists. This makes the pipeline robust to being run multiple times.

**5. Interaction with Other Project Components:**

The `config.py` module is imported by most other Python scripts in the `src/` directory:
*   **`data_preprocessing.py`:** Uses `RAW_DATA_FILE`, `PROCESSED_DATA_DIR`, `TRAIN_DATA_FILE`, `TEST_DATA_FILE`, `TARGET_COLUMN`, `TIMESTAMP_COLUMN`, `COLUMNS_TO_DROP`, `CATEGORICAL_FEATURES`, `RANDOM_SEED`, `TEST_SIZE`, etc.
*   **`train_models.py`:** Uses `MODEL_DIR` and specific model paths, `RANDOM_SEED`, and model-specific hyperparameters like `RF_N_ESTIMATORS`.
*   **`evaluate_models.py`:** Uses `MODEL_DIR` (to load models), `RESULTS_DIR`, `EVALUATION_METRICS_FILE`, `CONFUSION_MATRIX_DIR`.
*   **`main.py`:** Imports and uses various configurations to orchestrate the pipeline steps, and importantly, it's the script that would need modification to allow `RAW_DATA_FILE` to be overridden by a command-line argument from the UI.

This centralized approach ensures all parts of the pipeline refer to the same settings, paths, and parameters.

**6. Debugging Aids:**

As mentioned in the `implementation_guide.md`, `config.py` often includes `print()` statements at the end of the script. These statements typically display the resolved absolute paths for key directories (e.g., `PROJECT_ROOT`, `DATA_DIR`, `MODEL_DIR`, `RESULTS_DIR`). This is a simple but effective debugging aid, allowing the user to quickly verify that the paths are being resolved correctly based on their current environment and project location.

In summary, `config.py` is a cornerstone of the `fraud_detection_project`, promoting organized, maintainable, and easily adaptable code by centralizing all configuration aspects of the machine learning pipeline.
