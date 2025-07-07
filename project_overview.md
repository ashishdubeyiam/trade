# Fraud Detection Pipeline Project Overview

This document provides a detailed description of the Fraud Detection Pipeline project, covering its frontend and backend implementations, as well as how prediction models are prepared and integrated.

## 1. Frontend Implementation (`fraud_detection_ui/`)

The frontend of the "Fraud Detection Pipeline" project is responsible for user interaction, specifically allowing users to upload transaction data and view the results of the machine learning pipeline processing that data. It is built using Flask's templating engine to render HTML pages.

*   **`index.html` (Main Upload Page):**
    *   **Purpose:** This is the landing page of the web application. Its primary function is to provide a user interface for uploading a CSV file containing transaction data.
    *   **Structure and Form Elements:**
        *   It displays a clear title: "Upload Transaction Data for Fraud Detection."
        *   It includes a form (`<form>`) that uses the `POST` method and `enctype="multipart/form-data"` which is necessary for file uploads. The form submits data to the `/upload` URL endpoint, handled by the Flask backend.
        *   **File Input:** A file input field (`<input type="file" id="datafile" name="datafile" accept=".csv" required>`) allows users to select a CSV file from their local system. The `accept=".csv"` attribute restricts file selection to CSV files in the browser's file dialog, and `required` makes the field mandatory for form submission.
        *   **Submit Button:** A submit button (`<input type="submit" value="Upload and Process">`) triggers the form submission.
    *   **User Feedback (Flash Messages):**
        *   The template includes a section `{% with messages = get_flashed_messages(with_categories=true) %}` to display flash messages passed from the Flask backend. These messages are used to inform the user about the status of their actions (e.g., "No file part in the request," "No selected file," "Invalid file type," or success/error messages after processing). Messages are styled based on their category (e.g., `error`, `success`).
    *   **Styling:** Basic inline CSS is used within `<style>` tags for layout and appearance (e.g., font, colors, padding, container styling).

*   **`results.html` (Results Display Page):**
    *   **Purpose:** This page is rendered after the backend has processed the uploaded file. It displays either the results of the fraud detection pipeline or an error message if something went wrong.
    *   **Structure and Content Display:**
        *   **Title:** "Fraud Detection Pipeline Results."
        *   **Error Handling:** If an `error_message` is passed from the backend, it's prominently displayed, often with a distinct style (e.g., red text, specific background). It also shows the filename associated with the error and can display `stderr_details` if available.
        *   **Metrics Display:** If `metrics_data` (a dictionary parsed from `evaluation_metrics.json`) is available, the page iterates through it.
            *   For each model's results in `metrics_data` (e.g., "Logistic Regression_all_features"), it displays a section header (e.g., "Metrics for: Logistic Regression_all_features").
            *   A table is used to list various performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP), Specificity, and Geometric Mean. Values are rounded to four decimal places where applicable, and 'N/A' is shown if a metric is not available.
        *   **Confusion Matrix Information:**
            *   The page informs the user about the path where the confusion matrix image for each model is saved (e.g., "Confusion Matrix plot saved as: results/confusion_matrices/logistic_regression_cm.png").
            *   It explicitly states: "(To view images, check the 'fraud_detection_project/results/confusion_matrices/' directory within your project. Direct image display in this UI is a future enhancement.)" This means the images themselves are not rendered on the web page.
        *   **No Data Message:** If no `error_message` and no `metrics_data` are available, a message like "No metrics data available to display for file '{{ filename }}'" is shown.
    *   **Navigation:**
        *   A "Upload Another File" link (`<a href="{{ url_for('index') }}">`) allows the user to easily return to the main upload page.
    *   **Styling:** Similar to `index.html`, it uses inline CSS for styling tables, sections, and messages.

*   **Flask Templating:**
    *   Both HTML files utilize Flask's Jinja2 templating engine features, such as:
        *   `{{ url_for('...') }}` for generating URLs dynamically.
        *   `{% with messages = ... %}` and `{% for ... %}` loops for displaying flash messages and iterating through results.
        *   `{{ variable }}` for rendering variables passed from the Flask backend (e.g., `metrics.accuracy`, `error_message`).
        *   Conditional rendering with `{% if ... %}`, `{% elif ... %}`, `{% else %}`.

In summary, the frontend provides a functional, if basic, interface for users to engage with the fraud detection pipeline by uploading data and reviewing the performance metrics generated by the backend processing. It relies on server-side rendering via Flask and provides essential user feedback through messages and a structured display of results.

## 2. Backend Implementation

The backend of the "Fraud Detection Pipeline" project is composed of two main parts: the Web Backend responsible for handling user requests and interacting with the ML pipeline, and the ML Backend which performs the core data processing and machine learning tasks.

### A. Web Backend (`fraud_detection_ui/app.py`)

The web backend is a Flask application that serves the user interface and orchestrates the execution of the machine learning pipeline.

*   **Flask Web Server:**
    *   It initializes a Flask app (`app = Flask(__name__)`).
    *   It defines a `secret_key` for enabling Flask's flash messaging system, used for user feedback.
    *   It configures an `UPLOAD_FOLDER` (set to `uploads/` relative to `app.py`) and ensures this directory exists.

*   **Route Handling:**
    *   **`@app.route('/')` (index):**
        *   Handles requests to the root URL.
        *   Renders and returns `index.html`, which is the main page for file uploads.
    *   **`@app.route('/upload', methods=['POST'])` (upload_file):**
        *   This is the core endpoint for processing user uploads. It only accepts `POST` requests.
        *   **File Validation and Saving:**
            *   Checks if a file part (`datafile`) is present in the request. If not, flashes an error and redirects to `index`.
            *   Checks if a filename is present. If empty, flashes an error and redirects.
            *   Uses a helper function `allowed_file(filename)` to verify if the file extension is in `ALLOWED_EXTENSIONS` (which is `{'csv'}`). If not allowed, flashes an error and redirects.
            *   If the file is valid, it uses `werkzeug.utils.secure_filename()` to sanitize the filename and saves the file to the configured `UPLOAD_FOLDER`. Errors during file saving are caught, flashed, and the user is redirected.
        *   **Pipeline Execution (Subprocess):**
            *   This is the most critical function of the web backend. It does **not** load ML models directly within the Flask app to make predictions. Instead, it triggers the entire ML pipeline located in the `fraud_detection_project/` directory as a separate subprocess.
            *   **Path Construction:** It dynamically constructs paths:
                *   `ui_root`: Absolute path to the `fraud_detection_ui` directory.
                *   `project_root_dir`: Absolute path to the parent directory containing both `fraud_detection_ui` and `fraud_detection_project`.
                *   `main_script_path`: Full path to `fraud_detection_project/src/main.py`.
                *   `absolute_uploaded_file_path`: Full path to the temporarily saved uploaded CSV.
                *   `ml_project_root_for_cwd`: Full path to `fraud_detection_project/`, used as the current working directory for the subprocess. This is crucial for `main.py` to correctly resolve its internal relative paths (e.g., to `config.py`, `data/`, `models/`).
            *   **Command Construction:** It prepares a command list to execute the ML pipeline:
                `['python', main_script_path, '--input-file', absolute_uploaded_file_path, '--experiment', 'all_features']`
                This command runs `main.py`, passing the path to the uploaded data and specifying the 'all_features' experiment type.
            *   **Subprocess Execution:**
                *   `subprocess.run(command, capture_output=True, text=True, check=False, timeout=300, cwd=ml_project_root_for_cwd)` is used.
                    *   `capture_output=True`: Captures `stdout` and `stderr`.
                    *   `text=True`: Decodes `stdout` and `stderr` as text.
                    *   `check=False`: Prevents raising an exception for non-zero exit codes (handled manually).
                    *   `timeout=300`: Sets a 5-minute timeout for the pipeline execution.
                    *   `cwd=ml_project_root_for_cwd`: Sets the working directory for `main.py`.
        *   **Results Handling from Subprocess:**
            *   `stdout` and `stderr` from the pipeline are printed to the Flask app's console for debugging.
            *   **Success Case (returncode 0):**
                *   Flashes a success message.
                *   Attempts to read the `evaluation_metrics.json` file (path constructed relative to `ml_project_root_for_cwd`).
                *   If the metrics file is found, its JSON content is loaded into `metrics_data`.
                *   If not found, an error message is prepared and flashed.
                *   Any content in `stderr` (even on success) is flashed as "Pipeline warnings/info."
            *   **Failure Case (non-zero returncode):**
                *   An error message including the return code is flashed. `stderr` content is captured as `stderr_details` and a snippet is flashed.
            *   **Timeout Case (`subprocess.TimeoutExpired`):**
                *   An error message about the timeout is flashed.
            *   **Other Exceptions:** Generic exceptions during subprocess execution are caught and flashed.
        *   **File Cleanup:** The uploaded CSV file in `app.config['UPLOAD_FOLDER']` is removed after the pipeline attempt (regardless of success, failure, or timeout) before rendering the results.
        *   **Rendering Results:** Finally, it renders `results.html`, passing `metrics_data`, `error_message`, `stderr_details`, and the original `filename`.
    *   **Error Handling and User Feedback:** The application uses Flask's `flash()` mechanism extensively to provide feedback to the user (e.g., file upload errors, pipeline status, success messages). These messages are displayed in the HTML templates.

*   **Execution Context:**
    *   The script can be run directly using `if __name__ == '__main__': app.run(debug=True)`, which starts a development server.

### B. ML Backend (`fraud_detection_project/src/`)

The ML backend consists of a set of Python scripts that implement the machine learning pipeline. The UI (`app.py`) triggers this backend via `main.py`.

*   **`config.py` (Configuration Management):**
    *   Centralizes all project configurations: paths to data, models, results; model parameters; feature definitions (e.g., `TIMESTAMP_COLUMN`, `COLUMNS_TO_DROP`, `CATEGORICAL_FEATURES`); and settings like `RANDOM_SEED` and `TARGET_COLUMN`.
    *   Creates necessary directories if they don't exist.
    *   **Crucial for UI Integration:** For the UI's `--input-file` argument to work, `config.py` or `main.py` needs to be adaptable. The `RAW_DATA_FILE` path in `config.py` would likely need to be dynamically set or overridden by the path passed from `main.py` when the `--input-file` argument is used.

*   **`data_preprocessing.py` (Data Handling and Preparation):**
    *   **`load_data(file_path)`:** Loads data from a CSV file (this would use the path passed via `--input-file` from `main.py`).
    *   **`inspect_data(df)`:** Prints basic information and statistics about the DataFrame.
    *   **`handle_class_imbalance(df, target_column)`:** Implements undersampling of the majority class to address class imbalance.
    *   **`preprocess_features(df, experiment_type)`:** Performs feature engineering (timestamp conversion, creating time-based features), drops specified columns, handles categorical features (One-Hot Encoding), and scales numerical features (StandardScaler). Uses `ColumnTransformer` for applying different transformations.
    *   **`split_data(X, y, test_size)`:** Splits data into training and testing sets.
    *   **`save_processed_data(...)`:** Saves the processed train and test sets to CSV files in `data/processed/`.

*   **`train_models.py` (Model Training):**
    *   Contains functions to train different classifiers:
        *   `train_logistic_regression(X_train, y_train)`
        *   `train_random_forest(X_train, y_train)`
        *   `train_decision_tree(X_train, y_train)`
        *   `train_ann_model(X_train, y_train)`: This is currently a **placeholder** and does not train a real ANN model.
    *   Models are initialized with parameters (some from `config.py`) and fitted on the training data.
    *   Trained models are serialized and saved to the `models/` directory using `joblib` (or a dummy file for ANN).

*   **`evaluate_models.py` (Model Performance Assessment):**
    *   **`load_test_data(test_file_path)`:** Loads the processed test data.
    *   **`load_model(model_path)`:** Loads a serialized model from disk.
    *   **`calculate_metrics(y_true, y_pred, y_pred_proba)`:** Calculates various metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix components like TN, FP, FN, TP, specificity, geometric mean).
    *   **`plot_confusion_matrix(cm, model_name)`:** Generates and saves a plot of the confusion matrix to `results/confusion_matrices/`.
    *   **`evaluate_model(model, X_test, y_test, model_name)`:** Orchestrates the evaluation for a single model.
    *   The script iterates through trained models, evaluates them, and aggregates all metrics into `all_metrics`. This dictionary is then saved as a JSON file to `config.EVALUATION_METRICS_FILE` (`results/evaluation_metrics.json`).

*   **`main.py` (Pipeline Orchestration):**
    *   This is the entry point for the ML pipeline, executed by the Flask backend's subprocess call.
    *   **Argument Parsing:** It needs to be adapted to parse the `--input-file <filepath>` command-line argument passed by `fraud_detection_ui/app.py`. This is a **critical modification required** for the system to work as intended by the UI.
        *   When `--input-file` is provided, `main.py` should use this file path as the source for `data_preprocessing.load_data()`, likely by updating the `config.RAW_DATA_FILE` variable dynamically or passing the path through its internal function calls.
    *   **Pipeline Execution Steps:**
        1.  Loads raw data (using the path from `--input-file`).
        2.  Performs data preprocessing (imbalance handling, feature engineering, splitting) using functions from `data_preprocessing.py`. Saves processed data.
        3.  Trains all models (Logistic Regression, Random Forest, Decision Tree, ANN placeholder) using functions from `train_models.py`, saving the trained models. This means models are **retrained on the uploaded data in each run**.
        4.  Evaluates the newly trained models using functions from `evaluate_models.py`, saving metrics to `evaluation_metrics.json` and confusion matrix plots.
    *   The `evaluation_metrics.json` file created by this script is then read by `fraud_detection_ui/app.py` to display results.

**Summary of Backend Interaction and Data Flow for a UI Request:**
1.  User uploads a CSV via `index.html`.
2.  `fraud_detection_ui/app.py` saves the CSV to `uploads/`.
3.  `app.py` calls `python fraud_detection_project/src/main.py --input-file <path_to_uploaded_csv> ...` as a subprocess.
4.  `main.py` (ML Backend):
    *   Reads the specified input CSV.
    *   Preprocesses the data.
    *   Trains models on this data from scratch.
    *   Evaluates these models on a test split of this data.
    *   Saves trained models, processed data, evaluation metrics (JSON), and confusion matrix plots.
5.  `app.py` (after subprocess completion):
    *   Reads `evaluation_metrics.json`.
    *   Deletes the uploaded CSV from `uploads/`.
    *   Renders `results.html` with the metrics or error information.

This architecture means the system effectively performs a full train-and-evaluate cycle on the user-provided dataset with each upload, rather than using pre-trained models for inference on new data.

## 3. Prediction Model Preparation and Integration

The "Prediction Model Preparation and Integration" in this project has a unique characteristic: models are not pre-trained and then loaded by the UI for inference on new data. Instead, the entire process of model preparation (training and evaluation) is triggered by the UI for each dataset uploaded by the user.

### A. Model Preparation (as per the current system)

Model preparation occurs dynamically every time a user uploads a CSV file and the `fraud_detection_ui/app.py` script invokes the `fraud_detection_project/src/main.py` pipeline.

1.  **Data Reception (via UI-triggered `main.py`):**
    *   The process begins when `main.py` is executed with the `--input-file <path_to_uploaded_csv>` argument. This uploaded CSV becomes the source dataset for the current run.
    *   The `data_preprocessing.load_data()` function is called with this path to load the user's data.

2.  **Data Preprocessing (`data_preprocessing.py`):**
    *   **Inspection:** Basic properties of the uploaded data are printed.
    *   **Class Imbalance Handling:** The `handle_class_imbalance()` function is applied, which performs undersampling of the majority class to create a more balanced dataset for training. This is done on the entire uploaded dataset before splitting.
    *   **Feature Engineering & Transformation:** `preprocess_features()` is called:
        *   Timestamp columns are converted, and features like 'Hour_of_Day', 'Day_of_Week', 'Month_of_Year' are extracted.
        *   Predefined columns (from `config.COLUMNS_TO_DROP`) are removed.
        *   Categorical features (defined in `config.CATEGORICAL_FEATURES`) are one-hot encoded using `sklearn.compose.ColumnTransformer` and `OneHotEncoder`.
        *   Numerical features are scaled using `StandardScaler`.
    *   **Data Splitting:** The processed dataset (features `X` and target `y`) is split into training and testing sets using `split_data()` (typically an 80/20 split).
    *   **Saving Processed Data:** The resulting `X_train, y_train, X_test, y_test` are saved as `train.csv` and `test.csv` in the `fraud_detection_project/data/processed/` directory.

3.  **Model Training (`train_models.py`):**
    *   The `X_train` and `y_train` (derived from the user's uploaded file) are passed to various training functions:
        *   **`train_logistic_regression(X_train, y_train)`:** Trains a Logistic Regression model.
        *   **`train_random_forest(X_train, y_train)`:** Trains a Random Forest Classifier using parameters from `config.py` (e.g., `RF_N_ESTIMATORS`, `RF_MAX_DEPTH`).
        *   **`train_decision_tree(X_train, y_train)`:** Trains a Decision Tree Classifier.
        *   **`train_ann_model(X_train, y_train)`:** This function is a **placeholder**. It does not define or train an actual Artificial Neural Network.
    *   **Saving Trained Models:** Each successfully trained model (Logistic Regression, Random Forest, Decision Tree) is serialized using `joblib.dump()` and saved into the `fraud_detection_project/models/` directory. The ANN placeholder also creates a file. These saved models are overwritten with each new pipeline run.

4.  **Model Evaluation (`evaluate_models.py`):**
    *   The models trained in the previous step are loaded back using `joblib.load()`.
    *   The `X_test` and `y_test` (from the split of the user's uploaded data) are used for evaluation.
    *   For each model, predictions are made, metrics calculated (accuracy, precision, recall, F1, ROC-AUC, confusion matrix components), and confusion matrix plots are saved.
    *   **Aggregating Results:** All metrics are compiled into `evaluation_metrics.json`.

### B. Model Integration (UI with ML Pipeline)

The integration between the UI (`fraud_detection_ui/app.py`) and the model preparation pipeline (`fraud_detection_project/src/main.py`) is not for performing predictions with pre-trained models. Instead, the UI triggers the entire model preparation lifecycle.

1.  **Triggering Mechanism:**
    *   User uploads CSV via `index.html`.
    *   `app.py` receives the file.
    *   `app.py` executes `main.py` as a subprocess, passing the uploaded CSV path via `--input-file`.

2.  **Data Flow for "Prediction" (actually, full pipeline execution):**
    *   `main.py` uses the `--input-file` to read the user's data.
    *   The data goes through the full preprocessing, training-on-this-data, and evaluation-on-this-data cycle. Models are trained from scratch using the uploaded data.
    *   The output is `evaluation_metrics.json` and saved model files, reflecting performance *on the specific dataset the user just uploaded*.

3.  **Displaying Results (Not Predictions):**
    *   `app.py` reads `evaluation_metrics.json` after `main.py` completes.
    *   It passes these metrics to `results.html`.
    *   The user sees the performance metrics of models trained on their data.

4.  **No Persistent Trained Models for General Prediction:**
    *   Models are retrained on each upload; there isn't a persistent model for general, quick predictions. Each request is a "train and evaluate on this new dataset" request.

**Implications of this Approach:**

*   **Purpose:** Suitable for demonstrating an end-to-end ML pipeline, allowing users to see model performance on their data, and testing script adaptability.
*   **Not for Standard Inference:** Not designed for production scenarios requiring fast inference with pre-trained models.
*   **ANN Functionality:** The ANN model is a placeholder and needs full implementation for genuine use.

## 4. Concluding Summary

The Fraud Detection Pipeline project provides a web interface for users to upload transaction data (CSV files). Upon upload, the backend Flask application triggers a comprehensive machine learning pipeline as a subprocess. This pipeline, located in a separate project directory, performs data preprocessing (handling imbalance, feature engineering, scaling), then trains several models (Logistic Regression, Random Forest, Decision Tree, and a placeholder ANN) from scratch using the uploaded data. After training, these models are evaluated on a test split of the same data, and performance metrics (accuracy, precision, recall, F1-score, ROC-AUC, etc.) along with confusion matrices are generated.

The results, specifically the evaluation metrics, are then displayed back to the user on a results page. A critical aspect of the current design is that models are not pre-trained and then used for inference. Instead, the entire train-and-evaluate cycle is executed for each new dataset provided by the user. This makes the system more of a demonstration and testing platform for the ML pipeline itself, rather than a tool for making predictions on new instances with established models. The system highlights the structure of an ML project but would require significant changes (e.g., implementing actual ANN, modifying `main.py` to correctly use the input file path from the UI, and potentially decoupling model training from prediction) for a production-like predictive service.
