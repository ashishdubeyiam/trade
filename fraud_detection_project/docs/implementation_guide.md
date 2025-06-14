# Implementation Guide: Machine Learning Pipeline for Fraud Detection

## 1. Introduction

This guide provides a detailed walkthrough of the implementation of the machine learning pipeline for fraud detection. It covers the structure of the project, the role of each Python script, configuration management, data handling processes, model training, and evaluation strategies. The pipeline is designed to be modular, allowing for relatively straightforward modifications and extensions.

The primary goal of this project is to establish an end-to-end workflow that takes raw transactional data, processes it, trains various classification models, evaluates their performance, and saves the results and trained models. This document serves as a technical reference for understanding and potentially extending the existing codebase.

## 2. Configuration Management: `src/config.py`

The `src/config.py` script is the central hub for all project configurations. Centralizing these settings makes it easy to adapt the pipeline to different datasets, change model parameters, or adjust file paths without modifying the core logic of the other scripts.

### Key Configuration Variables:

*   **Project Root (`PROJECT_ROOT`):**
    *   Dynamically determines the absolute path to the project's root directory (`fraud_detection_project/`). This ensures that all other paths constructed from it are robust to changes in the project's location on the filesystem.
    *   Calculated as: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (assuming `config.py` is in `src/`).

*   **Data Paths:**
    *   `DATA_DIR`: Path to the `data/` directory (e.g., `/app/fraud_detection_project/data`).
    *   `RAW_DATA_FILE`: Full path to the raw input dataset CSV file. This is a critical path that users must verify. It is configured to point to `dummy_fraud_dataset.csv` for initial testing or `synthetic_fraud_dataset.csv` for the user's actual data.
        *   Example: `os.path.join(DATA_DIR, 'dummy_fraud_dataset.csv')`
    *   `PROCESSED_DATA_DIR`: Path to the directory where processed data (train/test splits) will be saved (e.g., `data/processed/`).
    *   `TRAIN_DATA_FILE`: Full path for saving the processed training data CSV.
    *   `TEST_DATA_FILE`: Full path for saving the processed testing data CSV.

*   **Model Output Paths:**
    *   `MODEL_DIR`: Path to the `models/` directory where trained models will be serialized and stored.
    *   Individual model paths are defined for each model type:
        *   `LOGISTIC_REGRESSION_MODEL_PATH`
        *   `RANDOM_FOREST_MODEL_PATH`
        *   `DECISION_TREE_MODEL_PATH`
        *   `ANN_MODEL_PATH` (for the placeholder Artificial Neural Network model, typically an `.h5` file if using Keras/TensorFlow).

*   **Model Parameters (Examples):**
    *   `RF_N_ESTIMATORS`: Number of trees for the Random Forest model (e.g., `100`).
    *   `RF_MAX_DEPTH`: Maximum depth for Random Forest trees (e.g., `10`).
    *   `RF_RANDOM_STATE`: Random state for reproducibility in Random Forest (e.g., `42`).
    *   `ANN_EPOCHS`: Number of epochs for ANN training (e.g., `50`).
    *   `ANN_BATCH_SIZE`: Batch size for ANN training (e.g., `32`).
    *   *(Note: These are illustrative; actual ANN parameters would be used in a full implementation.)*

*   **Feature Engineering Configuration:**
    *   `TIMESTAMP_COLUMN`: Name of the column containing timestamp information (e.g., `'Timestamp'`).
    *   `COLUMNS_TO_DROP`: A list of column names to be dropped from the dataset early in the preprocessing phase (e.g., `['Transaction_ID', 'User_ID']`).
    *   `CATEGORICAL_FEATURES`: A list of column names to be treated as categorical features for one-hot encoding.
        *   Example: `['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method', 'IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']`
    *   `FEATURES_TO_DROP_EXPERIMENT2`: A list of features to be dropped specifically for the 'selected_features' experiment type (e.g., `['nameOrig', 'nameDest']`). This allows for testing different feature sets as per the original research proposal.

*   **Evaluation Output Paths:**
    *   `RESULTS_DIR`: Path to the `results/` directory.
    *   `EVALUATION_METRICS_FILE`: Full path to the JSON file where evaluation metrics for all models will be saved.
    *   `CONFUSION_MATRIX_DIR`: Path to the subdirectory within `results/` where confusion matrix plots will be stored.

*   **Other Settings:**
    *   `RANDOM_SEED`: A global random seed (e.g., `42`) used in various parts of the pipeline (like data splitting, model initialization, undersampling) to ensure reproducibility of results.
    *   `TARGET_COLUMN`: The name of the target variable in the dataset (e.g., `'Fraud_Label'`).

*   **Directory Creation:**
    *   The script also includes `os.makedirs(..., exist_ok=True)` calls to automatically create the `PROCESSED_DATA_DIR`, `MODEL_DIR`, `RESULTS_DIR`, and `CONFUSION_MATRIX_DIR` if they do not already exist when the `config.py` module is loaded or run.
    *   Print statements at the end of `config.py` display the resolved absolute paths for `PROJECT_ROOT`, `DATA_DIR`, `MODEL_DIR`, and `RESULTS_DIR`, which can be helpful for debugging path issues.

**Usage:** Other scripts (`data_preprocessing.py`, `train_models.py`, `evaluate_models.py`, `main.py`) import the `config` module to access these settings. This ensures consistency across the pipeline.

## 3. Data Handling and Preparation: `src/data_preprocessing.py`

This script is responsible for all aspects of data loading, inspection, cleaning, feature engineering, transformation, and splitting into training and testing sets.

### Core Functions:

*   **`load_data(file_path)`:**
    *   Takes a file path (typically `config.RAW_DATA_FILE`) as input.
    *   Attempts to load the dataset from the CSV file using `pd.read_csv()`.
    *   Includes basic error handling:
        *   Checks if the file exists at the given path. If not, it prints an error and returns `None`.
        *   Catches general exceptions during CSV loading and returns `None`.
    *   Prints status messages about loading progress.

*   **`inspect_data(df)`:**
    *   Takes a pandas DataFrame `df` as input.
    *   Prints comprehensive initial information about the dataset:
        *   `df.shape`: Dimensions (number of rows and columns).
        *   `df.head()`: First 5 rows to show a sample of the data.
        *   `df.info()`: Data types of each column and non-null counts.
        *   `df.describe()`: Descriptive statistics for numerical columns.
        *   `df.isnull().sum()`: Count of missing values for each column.
        *   Target variable distribution (`config.TARGET_COLUMN`): Value counts and normalized proportions of the target classes (e.g., Fraud vs. Non-Fraud).
        *   Timestamp column inspection: If `config.TIMESTAMP_COLUMN` is defined and present, it prints the column's data type and a few unique values to help diagnose parsing issues.

*   **`handle_class_imbalance(df, target_column)`:**
    *   Addresses the common issue of class imbalance in fraud datasets.
    *   **Method Used:** Undersampling of the majority class.
    *   **Process:**
        1.  Separates the DataFrame into two: one for the majority class (non-fraudulent) and one for the minority class (fraudulent) based on the `target_column`.
        2.  Prints the original counts of fraudulent and non-fraudulent samples.
        3.  If no fraudulent samples are found or if non-fraudulent samples are already less than or equal to fraudulent ones, it returns the original DataFrame (or a shuffled version).
        4.  Uses `sklearn.utils.resample` to randomly select a subset from the majority class DataFrame. The size of this subset (`n_samples`) is set to be equal to the number of samples in the minority class. `replace=False` ensures sampling without replacement. `random_state=config.RANDOM_SEED` ensures reproducibility.
        5.  Concatenates the undersampled majority class DataFrame with the original minority class DataFrame to form a new, balanced DataFrame.
        6.  Prints the counts after undersampling.
        7.  Shuffles the resulting balanced DataFrame (`.sample(frac=1, random_state=config.RANDOM_SEED)`) and resets its index before returning it.

*   **`preprocess_features(df, experiment_type='all_features')`:**
    *   This is the main workhorse function for feature engineering and transformation.
    *   Takes the DataFrame `df` (typically the output of `handle_class_imbalance`) and an `experiment_type` string as input.
    *   **Feature (X) and Target (y) Separation:**
        *   `X` is created by dropping the `config.TARGET_COLUMN` from `df`.
        *   `y` is extracted as the `config.TARGET_COLUMN`.
    *   **Timestamp Feature Engineering:**
        1.  Checks if `config.TIMESTAMP_COLUMN` exists in `X`.
        2.  Converts the timestamp column to datetime objects: `X[config.TIMESTAMP_COLUMN] = pd.to_datetime(X[config.TIMESTAMP_COLUMN], errors='coerce')`. `errors='coerce'` will turn unparseable dates into `NaT` (Not a Time).
        3.  Creates new time-based features:
            *   `X['Hour_of_Day'] = X[config.TIMESTAMP_COLUMN].dt.hour`
            *   `X['Day_of_Week'] = X[config.TIMESTAMP_COLUMN].dt.dayofweek` (Monday=0, Sunday=6)
            *   `X['Month_of_Year'] = X[config.TIMESTAMP_COLUMN].dt.month`
        4.  Drops the original `config.TIMESTAMP_COLUMN` from `X`.
    *   **Dropping Specified Columns:**
        1.  Checks if `config.COLUMNS_TO_DROP` is defined and not empty.
        2.  Drops these columns from `X` using `X.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')`. `errors='ignore'` prevents failure if a column to drop is not present.
    *   **Experiment-Specific Feature Dropping:**
        1.  If `experiment_type == 'selected_features'`, it checks for `config.FEATURES_TO_DROP_EXPERIMENT2`.
        2.  If defined, these additional features are dropped from `X`. This allows for testing different feature sets.
    *   **Identifying Numerical and Categorical Features (Post-Transformation):**
        1.  `numerical_features = X.select_dtypes(include=np.number).columns.tolist()`: Automatically identifies all columns with numerical data types after the above transformations. This will include newly created time features and any original numerical features that were kept.
        2.  `categorical_features = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]`: Filters the predefined `config.CATEGORICAL_FEATURES` list to include only those that are still present in `X` after dropping operations.
    *   **ColumnTransformer Setup:**
        1.  A `sklearn.compose.ColumnTransformer` is defined to apply different transformations to different types of columns.
        2.  It has two main transformers:
            *   `('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)`: Applies one-hot encoding to the identified categorical features. `handle_unknown='ignore'` prevents errors if new categories appear in test data (they will be all zeros). `sparse_output=False` results in a dense numpy array.
            *   `('scaler', StandardScaler(), numerical_features)`: Applies standard scaling (zero mean, unit variance) to all identified numerical features.
        3.  `remainder='passthrough'`: This ensures that any columns not explicitly handled by the 'onehot' or 'scaler' transformers are passed through without modification. Ideally, all relevant features should be covered by the transformers.
    *   **Applying Transformations:**
        1.  `X_processed = preprocessor.fit_transform(X)`: The `ColumnTransformer` is fitted on `X` and then `X` is transformed. For a robust pipeline, `fit_transform` should ideally be done only on the training data, and `transform` on the test data. However, in this script, if run standalone, it processes the entire `X` provided. When called from `main.py`, `X_processed` is derived from the full balanced dataset, then split.
    *   **Reconstructing Feature Names (Post-Transformation):**
        1.  This part of the code attempts to get the new feature names after one-hot encoding. `preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)` is used.
        2.  If this fails (which can happen, or if the list of names doesn't match the number of columns), it falls back to generic names or prints a warning. *This is a known area of potential brittleness in scikit-learn pipelines if not handled carefully, especially with complex transformers.* For the dummy data, it resulted in integer column names.
        3.  The scaled numerical feature names and any remainder column names are appended.
        4.  A new DataFrame `X_processed_df` is created from the transformed numpy array `X_processed` and the reconstructed column names. If names cannot be perfectly reconstructed, a warning is printed.
    *   Returns `X_processed_df` and the original `y`.

*   **`split_data(X, y, test_size=0.2)`:**
    *   Takes the processed features `X` and target `y`.
    *   Uses `sklearn.model_selection.train_test_split` to divide the data.
    *   `test_size=0.2`: Allocates 20% of the data to the test set and 80% to the training set.
    *   `random_state=config.RANDOM_SEED`: Ensures the split is the same every time the script runs.
    *   `stratify=y`: If `y` is not None and has more than one class, this attempts to preserve the proportion of target classes in both the train and test sets. This is important for classification tasks.
    *   Prints the shapes of the resulting `X_train`, `X_test`, `y_train`, `y_test`.
    *   Returns these four DataFrames/Series.

*   **`save_processed_data(X_train, X_test, y_train, y_test, train_file, test_file)`:**
    *   Takes the split data components and file paths for saving.
    *   Ensures the output directories (`os.path.dirname(train_file)` and `test_file`) exist using `os.makedirs(..., exist_ok=True)`.
    *   Combines features and target for saving: `pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)`. This creates DataFrames where the target column is re-attached to its respective feature set.
    *   Saves the combined training and testing DataFrames to CSV files using `df.to_csv(path, index=False)`.
    *   Prints confirmation messages with the save paths.

*   **`if __name__ == '__main__':` block:**
    *   This block allows `data_preprocessing.py` to be run as a standalone script for testing its functionality.
    *   It sequentially calls `load_data`, `inspect_data`, `handle_class_imbalance`, `preprocess_features`, `split_data`, and `save_processed_data`, using paths and settings from `config.py`.
    *   This is useful for debugging the preprocessing steps independently of the full pipeline.

### Data Flow Summary:
Raw CSV -> `load_data` -> DataFrame -> `inspect_data` (prints info) -> `handle_class_imbalance` (undersamples) -> Balanced DataFrame -> `preprocess_features` (feature engineering, OHE, scaling) -> Processed `X` and `y` -> `split_data` -> `X_train, X_test, y_train, y_test` -> `save_processed_data` -> Saved CSVs.

## 4. Model Training: `src/train_models.py`

This script is dedicated to training the various machine learning models selected for the project. It loads the processed training data, initializes each model with parameters from `config.py` (where applicable), trains them, and then saves the trained models to disk.

### Core Functions:

*   **`load_processed_data(train_file_path)`:**
    *   Similar to `load_data` in `data_preprocessing.py`, but specifically for the processed training data.
    *   Takes the `train_file_path` (e.g., `config.TRAIN_DATA_FILE`) as input.
    *   Loads the CSV file into a pandas DataFrame.
    *   Separates features (`X_train`) from the target (`y_train`) by dropping `config.TARGET_COLUMN` to get `X_train` and selecting it for `y_train`.
    *   Includes error handling for file not found or other loading issues.
    *   *(Note: When `train_models.py` functions are called from `main.py`, `X_train` and `y_train` are passed directly, so this function is primarily used if `train_models.py` is run as a standalone script).*

*   **`train_logistic_regression(X_train, y_train)`:**
    *   Takes `X_train` (features) and `y_train` (target) as input.
    *   Initializes `sklearn.linear_model.LogisticRegression` with `random_state=config.RANDOM_SEED` and `max_iter=1000` (increased for potential convergence issues with some solvers).
    *   Fits the model: `model.fit(X_train, y_train)`.
    *   Saves the trained model using `joblib.dump(model, config.LOGISTIC_REGRESSION_MODEL_PATH)`. `joblib` is efficient for saving scikit-learn models.
    *   Prints confirmation messages.
    *   Returns the trained model object.

*   **`train_random_forest(X_train, y_train)`:**
    *   Takes `X_train` and `y_train` as input.
    *   Initializes `sklearn.ensemble.RandomForestClassifier` with parameters sourced from `config.py`:
        *   `n_estimators=config.RF_N_ESTIMATORS`
        *   `max_depth=config.RF_MAX_DEPTH`
        *   `random_state=config.RF_RANDOM_STATE`
    *   Fits the model: `model.fit(X_train, y_train)`.
    *   Saves the trained model using `joblib.dump(model, config.RANDOM_FOREST_MODEL_PATH)`.
    *   Prints confirmation messages.
    *   Returns the trained model object.

*   **`train_decision_tree(X_train, y_train)`:**
    *   Takes `X_train` and `y_train` as input.
    *   Initializes `sklearn.tree.DecisionTreeClassifier` with `random_state=config.RANDOM_SEED`.
    *   Fits the model: `model.fit(X_train, y_train)`.
    *   Saves the trained model using `joblib.dump(model, config.DECISION_TREE_MODEL_PATH)`.
    *   Prints confirmation messages.
    *   Returns the trained model object.

*   **`train_ann_model(X_train, y_train)`:**
    *   This function serves as a **placeholder** for training an Artificial Neural Network (ANN).
    *   Takes `X_train` and `y_train` as input.
    *   Currently, it does not define or train an actual TensorFlow/Keras ANN model.
    *   It prints a message indicating that it's a placeholder and that full implementation requires network architecture details.
    *   It "saves" a placeholder by writing a text string into the file specified by `config.ANN_MODEL_PATH` (e.g., `ann_model.h5`). This is done to ensure the file path is used and tested in the pipeline, but the content is not a valid serialized model.
    *   Returns `None`.
    *   *(Future Enhancement: This function would be replaced with actual Keras/TensorFlow model definition, compilation, training (`model.fit`), and saving (`model.save(config.ANN_MODEL_PATH)`).)*

*   **`if __name__ == '__main__':` block:**
    *   Allows `train_models.py` to be run as a standalone script.
    *   It first calls `load_processed_data(config.TRAIN_DATA_FILE)` to get `X_train` and `y_train`.
    *   Then, it calls each of the training functions (`train_logistic_regression`, `train_random_forest`, `train_decision_tree`, `train_ann_model`) in sequence.
    *   Prints messages indicating which model training has been initiated.

### Workflow Note:
When the main pipeline (`src/main.py`) is executed, it directly passes the `X_train` and `y_train` DataFrames (obtained from `data_preprocessing.py`) to these training functions. The `load_processed_data` function within `train_models.py` is thus bypassed in that scenario, making the pipeline slightly more efficient by avoiding redundant data loading from disk.

## 5. Model Performance Assessment: `src/evaluate_models.py`

This script is responsible for evaluating the performance of the trained machine learning models using the processed test dataset. It loads the models, makes predictions, calculates a range of metrics, generates confusion matrix plots, and saves all results.

### Core Functions:

*   **`load_test_data(test_file_path)`:**
    *   Analogous to `load_processed_data` in `train_models.py`, but for the test set.
    *   Takes `test_file_path` (e.g., `config.TEST_DATA_FILE`) as input.
    *   Loads the CSV, separates features (`X_test`) from the target (`y_test`).
    *   Includes error handling.
    *   *(Note: When called from `main.py`, `X_test` and `y_test` are passed directly to `evaluate_model` function, so this loader is mainly for standalone execution of `evaluate_models.py`.)*

*   **`load_model(model_path)`:**
    *   Takes `model_path` (e.g., `config.LOGISTIC_REGRESSION_MODEL_PATH`) as input.
    *   Uses `joblib.load(model_path)` to load a serialized scikit-learn model.
    *   Includes error handling:
        *   Checks if the model file exists.
        *   Catches exceptions during loading (e.g., if the file is not a valid joblib file, which happens with the ANN placeholder).
    *   Returns the loaded model object or `None` if loading fails.

*   **`calculate_metrics(y_true, y_pred, y_pred_proba=None)`:**
    *   Calculates a dictionary of performance metrics.
    *   Inputs:
        *   `y_true`: Actual target values.
        *   `y_pred`: Predicted target values from the model.
        *   `y_pred_proba`: Predicted probabilities for the positive class (used for ROC-AUC). Optional.
    *   Metrics calculated using `sklearn.metrics`:
        *   `accuracy_score`
        *   `precision_score` (with `zero_division=0` to handle cases with no positive predictions)
        *   `recall_score` (with `zero_division=0`)
        *   `f1_score` (with `zero_division=0`)
        *   `roc_auc_score` (if `y_pred_proba` is provided and valid)
        *   `confusion_matrix`: Returns a NumPy array, which is converted to a list of lists for JSON serialization.
    *   **Detailed Confusion Matrix Components:**
        *   If the confusion matrix `cm` is 2x2 (binary classification standard case):
            *   Extracts `tn, fp, fn, tp = cm.ravel()`.
            *   Calculates `specificity = tn / (tn + fp)` (True Negative Rate).
            *   Calculates `geometric_mean = (recall * specificity)**0.5`.
        *   If `cm` is not 2x2 (e.g., if the model predicts only one class on the test set), these specific components are set to `None` or 0.0, and a warning is printed.
    *   Prints the calculated metrics to the console.
    *   Returns the `metrics` dictionary.

*   **`plot_confusion_matrix(cm, model_name)`:**
    *   Takes a confusion matrix (`cm`, typically a NumPy array) and `model_name` string.
    *   Uses `matplotlib.pyplot` and `seaborn.heatmap` to create a plot of the confusion matrix.
    *   Includes labels for axes and title. Annotations (`annot=True`) show the counts in each cell.
    *   Ensures the output directory `config.CONFUSION_MATRIX_DIR` exists.
    *   Saves the plot to a PNG file named `{model_name}_cm.png` within that directory.
    *   Closes the plot figure (`plt.close()`) to free memory.

*   **`evaluate_model(model, X_test, y_test, model_name)`:**
    *   Orchestrates the evaluation for a single model.
    *   Inputs: trained `model` object, `X_test` DataFrame, `y_test` Series, `model_name` string.
    *   Makes predictions: `y_pred = model.predict(X_test)`.
    *   If the model has a `predict_proba` method, it gets `y_pred_proba = model.predict_proba(X_test)[:, 1]` (probabilities for the positive class).
    *   Calls `calculate_metrics` to get the metrics dictionary.
    *   If a confusion matrix is present in the metrics, it converts the list version back to a NumPy array and calls `plot_confusion_matrix`.
    *   Returns the `metrics` dictionary for this model.
    *   Includes error handling for the evaluation process.

*   **`if __name__ == '__main__':` block:**
    *   Allows `evaluate_models.py` to be run as a standalone script.
    *   Loads test data using `load_test_data(config.TEST_DATA_FILE)`.
    *   Defines a dictionary `model_paths` mapping model names to their file paths from `config.py` (including the ANN placeholder).
    *   Iterates through `model_paths`:
        *   Loads each model using `load_model()`.
        *   If the model loads successfully, calls `evaluate_model()` to get its metrics.
        *   Stores the metrics in an `all_metrics` dictionary, keyed by model name.
    *   If `all_metrics` is not empty, it saves this dictionary to `config.EVALUATION_METRICS_FILE` as a JSON string with indentation for readability.
    *   Prints status messages.

## 6. Pipeline Orchestration: `src/main.py`

The `src/main.py` script is the master script that orchestrates the entire fraud detection pipeline, from data loading through model evaluation. It integrates the functionalities of `config.py`, `data_preprocessing.py`, `train_models.py`, and `evaluate_models.py`.

### Key Components and Workflow:

*   **Imports:**
    *   Standard libraries: `os`, `argparse`, `json`.
    *   Project modules: `config`, `data_preprocessing`, `train_models`, `evaluate_models`.
    *   Includes a `try-except ModuleNotFoundError` block for project module imports to provide helpful error messages if `main.py` is run from an incorrect location or if the `src` directory is not in `PYTHONPATH`. It also defines very basic fallback `config` class attributes if the main `config.py` fails to load, though this is for extreme cases.

*   **`run_pipeline(experiment_type='all_features')` function:**
    *   This is the core function that executes the pipeline steps.
    *   **Step 1: Data Preprocessing**
        1.  Calls `data_preprocessing.load_data(config.RAW_DATA_FILE)` to load the raw dataset. Halts if data loading fails.
        2.  Calls `data_preprocessing.inspect_data()` to print initial dataset statistics.
        3.  Checks if `config.TARGET_COLUMN` is in the loaded data. Halts if not.
        4.  Calls `data_preprocessing.handle_class_imbalance()` to perform undersampling. Halts if this step fails.
        5.  Calls `data_preprocessing.preprocess_features()` with the balanced DataFrame and the `experiment_type` argument. This performs timestamp engineering, column dropping, OHE, and scaling. Halts if this fails.
        6.  Calls `data_preprocessing.split_data()` to get `X_train, X_test, y_train, y_test`. Halts if this fails.
        7.  Calls `data_preprocessing.save_processed_data()` to save these splits to files specified in `config.py`.
    *   **Step 2: Model Training**
        1.  Calls the training functions from `train_models.py` directly, passing the in-memory `X_train` and `y_train` DataFrames:
            *   `train_models.train_logistic_regression(X_train, y_train)`
            *   `train_models.train_random_forest(X_train, y_train)`
            *   `train_models.train_decision_tree(X_train, y_train)`
            *   `train_models.train_ann_model(X_train, y_train)` (the placeholder)
        2.  This approach of passing data directly avoids reloading the processed training data from disk within `train_models.py` functions.
    *   **Step 3: Model Evaluation**
        1.  Defines a dictionary `models_to_evaluate` mapping model names (e.g., "Logistic Regression", "ANN") to their respective model file paths from `config.py`.
        2.  Iterates through this dictionary:
            *   Loads each model using `evaluate_models.load_model(model_path)`.
            *   If a model is loaded successfully, it calls `evaluate_models.evaluate_model(model, X_test, y_test, model_name)`, passing the in-memory `X_test`, `y_test`, and the loaded model.
            *   Stores the returned metrics in an `all_model_metrics` dictionary, keyed by a combination of model name and `experiment_type` (e.g., "Logistic Regression_all_features").
            *   If a model fails to load (like the ANN placeholder), it prints a message and skips its evaluation.
        3.  **Saving Aggregated Metrics:**
            *   If `all_model_metrics` is not empty, it attempts to load any existing metrics from `config.EVALUATION_METRICS_FILE`. This allows results from different experiments to be appended/updated in the same file.
            *   It updates the `existing_metrics` with the `all_model_metrics` from the current run.
            *   Saves the combined metrics dictionary back to `config.EVALUATION_METRICS_FILE` as a formatted JSON.

*   **`if __name__ == '__main__':` block:**
    *   This makes the script executable from the command line.
    *   **Argument Parsing:** Uses `argparse` to define and handle command-line arguments.
        *   `--experiment`: Takes a string, choices are `'all_features'` or `'selected_features'`, with `'all_features'` as the default. This argument is passed to `run_pipeline`.
    *   **Raw Data File Check:**
        *   Before calling `run_pipeline`, it checks if `config.RAW_DATA_FILE` (using the imported `config` module's value) exists.
        *   If the file is not found, it prints an informative error message and does not proceed with the pipeline execution. This is a crucial pre-flight check.
    *   Calls `run_pipeline(experiment_type=args.experiment)` if the data file exists.
    *   Includes commented-out example lines for running both experiments sequentially.

### Execution Flow:
When `python src/main.py` is run:
1.  `config.py` is imported (printing its resolved paths). Other modules are imported.
2.  The `if __name__ == '__main__':` block in `main.py` is executed.
3.  Command-line arguments are parsed.
4.  The existence of the raw data file (specified in the imported `config.py`) is checked.
5.  If the file exists, `run_pipeline()` is called with the specified experiment type.
6.  `run_pipeline()` executes data preprocessing, model training, and model evaluation steps sequentially, printing status messages throughout the process.
7.  Results (models, processed data CSVs, metrics JSON, plots) are saved to their respective directories.

## 7. Exploratory Data Analysis: `notebooks/eda_analysis_dummy_data.ipynb`

The project includes a Jupyter Notebook (`eda_analysis_dummy_data.ipynb`) located in the `notebooks/` directory. This notebook is designed to perform Exploratory Data Analysis (EDA) primarily on the `dummy_fraud_dataset.csv`.

### Purpose and Structure:

*   **Objective:** To provide a template and demonstration of common EDA techniques that can be applied to the fraud detection dataset. This helps in understanding data characteristics, identifying patterns, and informing feature engineering or modeling choices.
*   **Data Source:** The notebook is pre-configured to load `../data/dummy_fraud_dataset.csv`. Users intending to analyze the actual `synthetic_fraud_dataset.csv` would need to change the file path within the notebook and ensure the dataset is available.
*   **Key Sections in the Notebook:**
    1.  **Title:** "Exploratory Data Analysis (on Dummy Data)".
    2.  **Imports:** Imports standard data science libraries: `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`. Sets a plot style using `sns.set_style()`.
    3.  **Load Data:** Loads the CSV into a pandas DataFrame and displays the first few rows using `df.head()`.
    4.  **Basic Data Inspection:**
        *   `df.info()`: To view column data types and non-null counts.
        *   `df.describe(include='all')`: To get descriptive statistics for all columns (numerical and categorical).
        *   `df.isnull().sum()`: To check for missing values in each column.
    5.  **Numerical Feature Analysis (Markdown Section):**
        *   **Histograms:** Code cells to generate histograms for selected numerical columns (e.g., 'Transaction_Amount', 'Account_Balance', 'Risk_Score', 'Card_Age') to visualize their distributions.
        *   **Box Plots:** Code cells to create box plots for the same numerical features, often compared by 'Fraud_Label', to identify outliers and differences in distributions between classes.
    6.  **Categorical Feature Analysis (Markdown Section):**
        *   **Count Plots:** Code cells to generate count plots (bar charts) for selected categorical columns (e.g., 'Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category') to show the frequency of each category.
        *   **Categorical Features vs. Target:** Code cells to create count plots showing the distribution of the 'Fraud_Label' within each category of selected features (e.g., `sns.countplot(x='Transaction_Type', hue='Fraud_Label', data=df)`). This helps visualize how fraud incidence might vary across different categories.
    7.  **Correlation Analysis (Markdown Section):**
        *   **Correlation Heatmap:** A code cell that selects only numerical columns from the DataFrame, calculates their pairwise Pearson correlation matrix (`df.corr()`), and then visualizes this matrix as a heatmap using `seaborn.heatmap()`. Annotations are included to show correlation coefficients.
    8.  **Summary/Observations (Placeholder Markdown Cell):**
        *   A concluding markdown cell that emphasizes the limitations of EDA on the dummy data. It provides examples of insights one would typically seek with a real, larger dataset (e.g., identifying skewness, outliers, high-cardinality features, strong correlations, relationships with the target variable).

### Usage and Limitations:

*   **Running the Notebook:** Users can run this notebook in a Jupyter environment (Jupyter Lab, Jupyter Notebook classic, VS Code with Jupyter extension) provided they have installed the necessary dependencies (pandas, matplotlib, seaborn).
*   **Illustrative Nature:** As repeatedly emphasized in the notebook's content and this guide, the EDA performed on the `dummy_fraud_dataset.csv` (15 rows) is for **demonstration of the process only**. The plots and statistics generated from this tiny, synthetic dataset will not yield meaningful insights into actual fraud patterns or real data characteristics.
*   **Adaptation for Real Data:** Users should adapt this notebook to run on the full `synthetic_fraud_dataset.csv` to perform a meaningful EDA. This will involve changing the file path in the "Load Data" section and potentially adjusting plot parameters or feature selections based on the actual data's properties. The real EDA is crucial for understanding the dataset's intricacies and for guiding further refinements to the preprocessing and modeling strategies.

## 8. Experimental Setup and Results

The project is designed to facilitate experimentation, primarily through the `experiment_type` argument in `src/main.py` and by modifying settings in `src/config.py`.

### Experiment Types:

*   **`all_features` (Default):**
    *   This experiment uses all features that remain after the initial data cleaning and defined feature engineering steps (timestamp processing, dropping `Transaction_ID` and `User_ID`).
    *   The categorical features listed in `config.CATEGORICAL_FEATURES` are one-hot encoded, and numerical features are scaled.
*   **`selected_features`:**
    *   This experiment builds upon the `all_features` setup.
    *   Additionally, it drops columns specified in `config.FEATURES_TO_DROP_EXPERIMENT2` (which defaults to `['nameOrig', 'nameDest']` based on an earlier version of a dataset proposal).
    *   *Note:* The columns `nameOrig` and `nameDest` are not present in the `dummy_fraud_dataset.csv` or the user-provided schema for `synthetic_fraud_dataset.csv`. Therefore, in the current project state, running the `selected_features` experiment will produce the same feature set and results as `all_features` unless the `FEATURES_TO_DROP_EXPERIMENT2` list in `config.py` is modified to include relevant columns from the actual dataset being used.

### Output and Interpretation:

*   **Models:** Trained models are saved in the `models/` directory. These can be loaded for further analysis or deployment.
*   **Processed Data:** The exact `train.csv` and `test.csv` files used for a particular run are saved in `data/processed/`. This is useful for reproducibility and debugging.
*   **Evaluation Metrics (`results/evaluation_metrics.json`):**
    *   This JSON file stores a dictionary where keys are model names (appended with experiment type, e.g., "Random Forest_all_features") and values are dictionaries of their performance metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix values, specificity, geometric mean).
    *   The file is updated, so results from multiple experiments can accumulate.
    *   **Dummy Data Caveat:** When run with `dummy_fraud_dataset.csv`, the metrics in this file (especially for the 2-sample test set) are not indicative of real model performance but confirm the evaluation pipeline works. For example, perfect scores (1.0) or zero scores are common due to the tiny test set size.
*   **Confusion Matrix Plots (`results/confusion_matrices/`):**
    *   Visual plots for each model provide a quick understanding of classification accuracy, true positives, false positives, etc., for that specific run.

### Running Further Experiments:

Users can conduct further experiments by:

1.  **Modifying `config.py`:**
    *   Changing model hyperparameters (e.g., `RF_N_ESTIMATORS`, `RF_MAX_DEPTH`).
    *   Adjusting `CATEGORICAL_FEATURES`, `COLUMNS_TO_DROP`, or `TIMESTAMP_COLUMN` if different feature engineering is desired.
    *   Altering `FEATURES_TO_DROP_EXPERIMENT2` to test different feature subsets for the `selected_features` experiment.
2.  **Modifying `src/data_preprocessing.py`:**
    *   Implementing different feature engineering techniques (e.g., polynomial features, interaction terms).
    *   Trying alternative encoding for categorical features (e.g., target encoding, especially if `Location` proves to be high-cardinality in real data).
    *   Using different class imbalance strategies (e.g., SMOTE, ADASYN instead of undersampling).
3.  **Modifying `src/train_models.py`:**
    *   Adding new models to the training pipeline.
    *   Implementing the ANN model fully.
4.  **Analyzing `evaluation_metrics.json`:** After each experimental run, this file provides the quantitative results needed to compare different approaches.

The most critical step for meaningful results is to run the pipeline with the actual `synthetic_fraud_dataset.csv` provided by the user. This will establish a true baseline performance and highlight areas where the default configurations might need tuning based on real data characteristics (e.g., high cardinality of `Location`, actual distribution of `Timestamp`, etc.).
