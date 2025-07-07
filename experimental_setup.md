# Experimental Setup

This section details the experimental setup for the fraud detection project, covering the datasets, data preprocessing methodologies, machine learning models employed, evaluation strategies, and the software environment.

## 1. Datasets

The project is designed to work with transactional data in CSV format.

*   **`dummy_fraud_dataset.csv`:**
    *   Located in `fraud_detection_project/data/raw/`.
    *   This is a very small, synthetic dataset (15 rows, as per the `implementation_guide.md`'s EDA notebook description) primarily used for testing the structural integrity and execution flow of the ML pipeline. It allows developers to run the entire pipeline quickly and verify that all scripts connect and produce outputs in the expected formats and locations.
    *   **It is not suitable for drawing meaningful conclusions about model performance or actual fraud patterns due to its limited size and likely simplistic nature.**
*   **User-Uploaded CSV Data (via UI):**
    *   The primary mode of operation for the integrated system involves users uploading their own CSV files through the `fraud_detection_ui/`.
    *   The UI (`app.py`) saves this uploaded file temporarily and passes its path to the ML pipeline (`main.py` via the `--input-file` argument).
    *   The entire ML pipeline (preprocessing, training, evaluation) is then executed using this user-provided data. This means the "experiment" is effectively re-run on each new dataset uploaded.
*   **`synthetic_fraud_dataset.csv` (Placeholder/User-Provided):**
    *   The `implementation_guide.md` and `config.py` mention a `synthetic_fraud_dataset.csv` (expected at `fraud_detection_project/data/raw/synthetic_fraud_dataset.csv`). This appears to be the intended target dataset for more realistic experimentation, which the user of the codebase would provide. The project structure is set up to use this file if `config.RAW_DATA_FILE` is pointed to it. When using the UI, the uploaded file takes precedence for that specific run.

## 2. Data Preprocessing (`fraud_detection_project/src/data_preprocessing.py`)

Before model training, the selected dataset undergoes several preprocessing steps:

*   **Initial Data Loading and Inspection:** Data is loaded from the source CSV. Basic inspection (shape, data types, missing values, descriptive statistics) is performed (though output of `inspect_data` is mostly for console logging during pipeline execution).
*   **Class Imbalance Handling:**
    *   The `handle_class_imbalance()` function is applied to address the common issue of imbalanced classes in fraud detection (where non-fraudulent transactions vastly outnumber fraudulent ones).
    *   **Method Used:** Undersampling of the majority class. The number of majority class samples is reduced to match the number of minority class samples, using `sklearn.utils.resample`. This creates a balanced dataset for training.
*   **Feature Engineering (`preprocess_features()`):**
    *   **Timestamp Processing:** If a timestamp column (specified in `config.TIMESTAMP_COLUMN`) exists, it's converted to datetime objects. New features are extracted: 'Hour_of_Day', 'Day_of_Week', and 'Month_of_Year'. The original timestamp column is then dropped.
    *   **Column Dropping:** Predefined irrelevant columns (e.g., 'Transaction_ID', 'User_ID' listed in `config.COLUMNS_TO_DROP`) are removed.
    *   **Categorical Feature Encoding:** Features listed in `config.CATEGORICAL_FEATURES` (e.g., 'Transaction_Type', 'Device_Type') are transformed using One-Hot Encoding via `sklearn.compose.ColumnTransformer` and `OneHotEncoder`. This converts categorical string data into a numerical format suitable for ML models. `handle_unknown='ignore'` is used to manage new categories at test time.
    *   **Numerical Feature Scaling:** All numerical features (including newly created time-based features and one-hot encoded features) are scaled using `StandardScaler` from scikit-learn. This standardizes features by removing the mean and scaling to unit variance, which helps improve the performance of many ML algorithms.
*   **Data Splitting (`split_data()`):**
    *   The preprocessed dataset is split into training and testing sets (typically 80% training, 20% testing as per `config.TEST_SIZE`) using `sklearn.model_selection.train_test_split`.
    *   `stratify=y` is used to ensure that the proportion of the target classes is maintained in both train and test splits, which is important for classification tasks.
    *   The `config.RANDOM_SEED` is used throughout preprocessing (undersampling, train/test split) to ensure reproducibility.

## 3. Machine Learning Models (`fraud_detection_project/src/train_models.py`)

The project trains and evaluates several common classification models:

*   **Logistic Regression:** Implemented using `sklearn.linear_model.LogisticRegression` with parameters from `config.py` (e.g., `LR_MAX_ITER`).
*   **Decision Tree:** Implemented using `sklearn.tree.DecisionTreeClassifier` with `random_state=config.RANDOM_SEED`.
*   **Random Forest:** Implemented using `sklearn.ensemble.RandomForestClassifier`. Hyperparameters like `n_estimators` (`RF_N_ESTIMATORS`), `max_depth` (`RF_MAX_DEPTH`), and `random_state` (`RANDOM_SEED`) are configurable via `config.py`.
*   **Artificial Neural Network (ANN):**
    *   This is currently a **placeholder** (`train_ann_model`). The function does not define, compile, or train an actual neural network. It merely creates a dummy text file at the specified model path (`config.ANN_MODEL_PATH`). For meaningful ANN experimentation, this function would need to be fully implemented using a library like Keras/TensorFlow.

All models are trained on the processed training data derived from the input CSV.

## 4. Evaluation Strategy (`fraud_detection_project/src/evaluate_models.py`)

The performance of each trained model is assessed using the processed test set.

*   **Metrics Calculated:** A comprehensive suite of metrics is computed using `sklearn.metrics`:
    *   **Accuracy:** Overall correctness of the model.
    *   **Precision:** Ability of the model to avoid false positives (important for minimizing incorrect fraud alerts). Calculated with `zero_division=0`.
    *   **Recall (Sensitivity):** Ability of the model to identify actual positive cases (important for catching as much fraud as possible). Calculated with `zero_division=0`.
    *   **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure. Calculated with `zero_division=0`.
    *   **ROC-AUC Score:** Area Under the Receiver Operating Characteristic Curve, measuring the model's ability to distinguish between classes. Requires probability scores from the model.
    *   **Confusion Matrix Components:**
        *   True Positives (TP): Actual frauds correctly identified.
        *   False Positives (FP): Non-frauds incorrectly labeled as fraud.
        *   True Negatives (TN): Non-frauds correctly identified.
        *   False Negatives (FN): Actual frauds missed by the model.
    *   **Specificity (True Negative Rate):** Proportion of actual negatives correctly identified (`TN / (TN + FP)`).
    *   **Geometric Mean (G-Mean):** `sqrt(Recall * Specificity)`, useful for balanced performance assessment on imbalanced data.
*   These metrics are calculated for each model and stored in a structured format (JSON file: `config.EVALUATION_METRICS_FILE`).
*   **Confusion Matrix Plots:** For each model, a confusion matrix is plotted using `matplotlib` and `seaborn` and saved as a PNG image in the directory specified by `config.CONFUSION_MATRIX_DIR`.

## 5. Experiment Configuration (`fraud_detection_project/src/main.py`)

*   The `main.py` script can be run with an `--experiment` argument, which can be:
    *   `all_features` (default): Uses all features remaining after initial cleaning and feature engineering.
    *   `selected_features`: Allows for dropping an additional list of features specified in `config.FEATURES_TO_DROP_EXPERIMENT2` before one-hot encoding and scaling. (Note: The `implementation_guide.md` points out that with the `dummy_fraud_dataset.csv` and the default `FEATURES_TO_DROP_EXPERIMENT2` list, this experiment might not differ from `all_features` unless `FEATURES_TO_DROP_EXPERIMENT2` is updated for relevant columns in the actual dataset used).
*   The choice of experiment type is passed to `data_preprocessing.preprocess_features()`.
*   The UI (`app.py`) currently hardcodes the experiment type to `all_features` when calling `main.py`.

## 6. Software and Libraries (`fraud_detection_project/requirements.txt`)

The project relies on a Python environment with several key libraries:
*   **Python:** Core programming language (version as specified in project, e.g., 3.8+).
*   **Pandas:** For data manipulation and analysis (loading CSVs, DataFrame operations).
*   **NumPy:** For numerical operations, often used by Pandas and Scikit-learn.
*   **Scikit-learn (sklearn):** For machine learning tasks (preprocessing, models, metrics, splitting).
*   **Joblib:** For saving and loading trained scikit-learn models.
*   **Matplotlib & Seaborn:** For generating plots (used in `evaluate_models.py` for confusion matrices and in the EDA notebook).
*   **Flask:** For the web user interface (`fraud_detection_ui/`).
*   **(For ANN development, TensorFlow/Keras would be added to requirements)**.

This setup provides a reproducible framework for experimenting with different models and feature sets on user-provided transactional data, with a clear process for preprocessing, training, and evaluation. The results of these experiments (metrics and plots) are saved to the `results/` directory within `fraud_detection_project/`.
