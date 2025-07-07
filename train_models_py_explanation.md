## Explanation of `fraud_detection_project/src/train_models.py`

The `train_models.py` script in the `fraud_detection_project/src/` directory is specifically responsible for the model training phase of the machine learning pipeline. It defines functions to train various classification models using the preprocessed training data and then saves these trained models to disk for later evaluation and use.

**1. Overall Purpose:**

The primary objectives of this script are:
*   To encapsulate the training logic for different machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, and a placeholder for an ANN).
*   To use preprocessed training data to fit these models.
*   To serialize and save the trained model objects to disk, allowing them to be loaded and used by other parts of the pipeline (primarily for evaluation).

**2. Key Libraries Used:**

*   **`pandas`:** Used in the standalone execution mode by the `load_processed_data` function to read the processed training data CSV into a DataFrame.
*   **`joblib`:** The standard Python library for serializing and deserializing Python objects, particularly effective for saving and loading scikit-learn models efficiently.
*   **`sklearn.linear_model.LogisticRegression`:** The scikit-learn class for Logistic Regression.
*   **`sklearn.ensemble.RandomForestClassifier`:** The scikit-learn class for Random Forest models.
*   **`sklearn.tree.DecisionTreeClassifier`:** The scikit-learn class for Decision Tree models.
*   **`os` (implicitly):** Although not always directly imported in every function, path joining for model saving often relies on `os.path.join` indirectly through `config.py` which constructs full paths.

**3. Core Functions and their Logic:**

*   **`load_processed_data(train_file_path: str) -> tuple[pd.DataFrame, pd.Series]`:**
    *   **Purpose for Standalone Execution:** This function is designed to allow `train_models.py` to be run as a standalone script. It loads the processed training data from the CSV file specified by `train_file_path` (typically `config.TRAIN_DATA_FILE`).
    *   **Logic:**
        1.  Reads the CSV into a pandas DataFrame using `pd.read_csv()`.
        2.  Separates the DataFrame into features `X_train` (all columns except the target) and the target variable `y_train` (the `config.TARGET_COLUMN`).
        3.  Returns `X_train` and `y_train`.
    *   **Bypass by `main.py`:** As noted in the `implementation_guide.md`, when the pipeline is orchestrated by `main.py`, `main.py` already has `X_train` and `y_train` as in-memory DataFrames/Series (output from `data_preprocessing.py` functions). It passes these directly to the training functions below, making this `load_processed_data` call redundant and bypassed in the integrated workflow for efficiency.

*   **`train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series)`:**
    *   Initializes a `LogisticRegression` model.
        *   `random_state=config.RANDOM_SEED` is used for reproducibility.
        *   `max_iter=config.LR_MAX_ITER` (from `config.py`) is set to ensure convergence for some solvers/datasets.
    *   Fits the model to the provided training data: `model.fit(X_train, y_train)`.
    *   Saves the trained model to disk using `joblib.dump(model, config.LOGISTIC_REGRESSION_MODEL_PATH)`. The path is retrieved from `config.py`.
    *   Prints a confirmation message indicating the model has been trained and saved.

*   **`train_random_forest(X_train: pd.DataFrame, y_train: pd.Series)`:**
    *   Initializes a `RandomForestClassifier` model.
        *   Hyperparameters are sourced from `config.py`:
            *   `n_estimators=config.RF_N_ESTIMATORS`
            *   `max_depth=config.RF_MAX_DEPTH`
            *   `random_state=config.RANDOM_SEED`
        *   Other parameters like `min_samples_split` or `min_samples_leaf` might also be set here if defined in `config.py`.
    *   Fits the model: `model.fit(X_train, y_train)`.
    *   Saves the trained model: `joblib.dump(model, config.RANDOM_FOREST_MODEL_PATH)`.
    *   Prints a confirmation message.

*   **`train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series)`:**
    *   Initializes a `DecisionTreeClassifier` model.
        *   `random_state=config.RANDOM_SEED` is used for reproducibility.
        *   Other parameters like `max_depth` could be configured here if specified in `config.py`.
    *   Fits the model: `model.fit(X_train, y_train)`.
    *   Saves the trained model: `joblib.dump(model, config.DECISION_TREE_MODEL_PATH)`.
    *   Prints a confirmation message.

*   **`train_ann_model(X_train: pd.DataFrame, y_train: pd.Series)`:**
    *   **Placeholder Status (Crucial Emphasis):** As highlighted in the `implementation_guide.md`, this function in its current state **does not train an actual Artificial Neural Network.**
    *   **Logic:**
        1.  It typically prints a message to the console stating that the ANN model training is a placeholder and that a full implementation (e.g., using Keras/TensorFlow) is a future enhancement.
        2.  Instead of training a real model, it saves a dummy text string or a simple empty file to the path specified by `config.ANN_MODEL_PATH`. This is done to ensure that the `evaluate_models.py` script (if it attempts to load all model files listed in `config.py`) doesn't crash when it encounters the ANN model path, although it won't be able to use it for meaningful evaluation.
    *   The future enhancement would involve replacing this placeholder logic with actual Keras/TensorFlow model definition, compilation, training (`model.fit()`), and saving using `model.save()`.

**4. Workflow (`if __name__ == '__main__':` block):**

The script includes a main execution block (`if __name__ == '__main__':`) that allows it to be run standalone (e.g., `python src/train_models.py`). This workflow typically performs the following:
1.  Calls `load_processed_data(config.TRAIN_DATA_FILE)` to load `X_train` and `y_train` from the processed training CSV file.
2.  Sequentially calls each of the training functions:
    *   `train_logistic_regression(X_train, y_train)`
    *   `train_random_forest(X_train, y_train)`
    *   `train_decision_tree(X_train, y_train)`
    *   `train_ann_model(X_train, y_train)` (which executes the placeholder logic).
This standalone capability is useful for debugging the training process for each model independently or for retraining models if the processed data already exists.

**5. Workflow Note (Interaction with `main.py`):**

It's important to reiterate that when `main.py` orchestrates the entire pipeline:
*   The `X_train` and `y_train` DataFrames/Series are generated by `data_preprocessing.py` functions and are already available in memory.
*   `main.py` then passes these in-memory data structures directly to the respective training functions in `train_models.py` (e.g., `train_logistic_regression(X_train_df, y_train_series)`).
*   In this integrated scenario, the `load_processed_data()` function within `train_models.py` is **not called**, making the process more efficient by avoiding unnecessary disk I/O (reading the CSV file again).

**6. Dependencies and Interactions:**

*   **`config.py`:** This script is heavily dependent on `config.py` for:
    *   Model saving paths (e.g., `config.LOGISTIC_REGRESSION_MODEL_PATH`, `config.RANDOM_FOREST_MODEL_PATH`, etc.).
    *   Model hyperparameters (e.g., `config.RF_N_ESTIMATORS`, `config.RF_MAX_DEPTH`, `config.LR_MAX_ITER`, `config.RANDOM_SEED`).
    *   The path to the processed training data (`config.TRAIN_DATA_FILE`) when run in standalone mode.
    *   The target column name (`config.TARGET_COLUMN`) if `load_processed_data` is used.
*   **Input Data (from `data_preprocessing.py`):**
    *   The script consumes the processed training data (`X_train`, `y_train`). In standalone mode, it reads this from `train.csv`; in the integrated pipeline, it receives these as arguments from `main.py`.
*   **Output (Serialized Models):**
    *   The primary outputs of `train_models.py` are the serialized model files (e.g., `.joblib` files for scikit-learn models, a dummy file for the ANN placeholder) saved in the directory specified by `config.MODEL_DIR`.
    *   These saved model files are subsequently loaded and used by `evaluate_models.py` to assess their performance on the test dataset.

In summary, `train_models.py` plays a focused role in the ML pipeline: taking prepared training data, fitting various predefined machine learning models with specified configurations, and persisting these trained models for downstream evaluation.
