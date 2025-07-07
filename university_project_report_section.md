# Project Experimental Design, Outcomes, and Operational Characteristics

This section details the comprehensive experimental setup adopted for the fraud detection project. It covers the datasets utilized, the methodologies for data preprocessing, the suite of machine learning models employed, and the strategies for evaluating their performance. Furthermore, it describes how results are captured and visualized, and discusses the integration and deployment model of the system, highlighting how the machine learning pipeline and user interface components interact.

## 1. Experimental Setup

This sub-section details the experimental setup for the fraud detection project, covering the datasets, data preprocessing methodologies, machine learning models employed, evaluation strategies, and the software environment.

### 1.1. Datasets

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

### 1.2. Data Preprocessing (`fraud_detection_project/src/data_preprocessing.py`)

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

### 1.3. Machine Learning Models (`fraud_detection_project/src/train_models.py`)

The project trains and evaluates several common classification models:

*   **Logistic Regression:** Implemented using `sklearn.linear_model.LogisticRegression` with parameters from `config.py` (e.g., `LR_MAX_ITER`).
*   **Decision Tree:** Implemented using `sklearn.tree.DecisionTreeClassifier` with `random_state=config.RANDOM_SEED`.
*   **Random Forest:** Implemented using `sklearn.ensemble.RandomForestClassifier`. Hyperparameters like `n_estimators` (`RF_N_ESTIMATORS`), `max_depth` (`RF_MAX_DEPTH`), and `random_state` (`RANDOM_SEED`) are configurable via `config.py`.
*   **Artificial Neural Network (ANN):**
    *   This is currently a **placeholder** (`train_ann_model`). The function does not define, compile, or train an actual neural network. It merely creates a dummy text file at the specified model path (`config.ANN_MODEL_PATH`). For meaningful ANN experimentation, this function would need to be fully implemented using a library like Keras/TensorFlow.

All models are trained on the processed training data derived from the input CSV.

### 1.4. Evaluation Strategy (`fraud_detection_project/src/evaluate_models.py`)

The performance of each trained model is assessed using the processed test set.

*   **Metrics Calculated:** A comprehensive suite of metrics is computed using `sklearn.metrics`:
    *   **Accuracy:** Overall correctness of the model.
    *   **Precision:** Ability of the model to avoid false positives. Calculated with `zero_division=0`.
    *   **Recall (Sensitivity):** Ability of the model to identify actual positive cases. Calculated with `zero_division=0`.
    *   **F1-Score:** The harmonic mean of precision and recall. Calculated with `zero_division=0`.
    *   **ROC-AUC Score:** Area Under the Receiver Operating Characteristic Curve. Requires probability scores.
    *   **Confusion Matrix Components:** True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
    *   **Specificity (True Negative Rate):** Proportion of actual negatives correctly identified.
    *   **Geometric Mean (G-Mean):** `sqrt(Recall * Specificity)`.
*   These metrics are stored in `config.EVALUATION_METRICS_FILE`.
*   **Confusion Matrix Plots:** Generated using `matplotlib` and `seaborn`, saved in `config.CONFUSION_MATRIX_DIR`.

### 1.5. Experiment Configuration (`fraud_detection_project/src/main.py`)

*   `main.py` supports an `--experiment` argument (`all_features` or `selected_features`).
*   The UI currently hardcodes the experiment to `all_features`.

### 1.6. Software and Libraries (`fraud_detection_project/requirements.txt`)

Key libraries include Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, Seaborn, and Flask (for the UI). TensorFlow/Keras would be needed for a functional ANN.

This setup provides a reproducible framework for experimenting with models and feature sets on user-provided data.

## 2. Results & Visualizations

This sub-section outlines how results from the fraud detection pipeline are captured, stored, and visualized.

### 2.1. Quantitative Results (`fraud_detection_project/results/evaluation_metrics.json`)

*   **Storage:** Metrics are saved in `evaluation_metrics.json` in the `results/` directory. Keys are model/experiment names, values are dictionaries of metrics. Each run of `main.py` overwrites this file.
*   **Key Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, TP, FP, TN, FN, Specificity, G-Mean. These are vital for understanding model performance, especially in imbalanced fraud data.
*   **Interpretation via UI:** `results.html` reads `evaluation_metrics.json` and displays metrics in tables, allowing users to see performance on their uploaded data.

### 2.2. Visualizations

*   **Confusion Matrix Plots:**
    *   **Purpose:** Visually detail classifier performance (TP, FP, TN, FN).
    *   **Generation:** `evaluate_models.py` uses `matplotlib` and `seaborn` for each model.
    *   **Storage:** Saved as PNGs (e.g., `logistic_regression_cm.png`) in `results/confusion_matrices/`.
    *   **UI Indication:** `results.html` states their save path but doesn't display them (future enhancement).
*   **Exploratory Data Analysis (EDA) Visualizations (`fraud_detection_project/notebooks/eda_analysis_dummy_data.ipynb`):**
    *   **Purpose:** The Jupyter notebook aids in understanding data distributions, outliers, feature relationships, and correlations via plots like histograms, box plots, count plots, and heatmaps.
    *   **Caveat:** The notebook is pre-configured for `dummy_fraud_dataset.csv`. For meaningful insights, it must be adapted for larger, realistic datasets.

### 2.3. Interpreting Results in the Project's Context

When a user uploads data via the UI:
1.  The ML pipeline runs, training models *on that specific data*.
2.  `evaluation_metrics.json` is populated with performance on a test split *of that uploaded data*.
3.  The UI displays these metrics.
The "results" reflect how the pipeline modeled that particular dataset, not inference from a fixed, pre-trained model. For university experiments, this involves running the pipeline on suitable datasets, analyzing `evaluation_metrics.json` and plots, potentially modifying `config.py` for different parameters, and using EDA on actual data.

## 3. Deployment and Integration

This sub-section describes how the ML pipeline and UI are structured, integrated, and "deployed."

### 3.1. ML Pipeline (`fraud_detection_project/`) as a Standalone Unit

*   **Structure:** A self-contained pipeline with `src/` (scripts), `data/`, `models/`, `results/`, `notebooks/`, and `requirements.txt`.
*   **Execution:** Can run independently via `python src/main.py`, configured by `config.py` and CLI arguments.

### 3.2. Web User Interface (`fraud_detection_ui/`)

*   **Structure:** A Flask app with `app.py`, `templates/`, `static/`, and `uploads/`.
*   **Functionality:** UI for CSV upload, processing feedback, and viewing model performance metrics from the uploaded data.

### 3.3. Integration Mechanism: UI Orchestrating the ML Pipeline

*   **Trigger:** User uploads CSV via `index.html`.
*   **`app.py` Logic:**
    1.  Validates and saves the uploaded file to `uploads/`.
    2.  Constructs paths to `main.py` and the uploaded file.
    3.  **Subprocess Call:** Executes `main.py` using `subprocess.run()`:
        `['python', '<path_to_main.py>', '--input-file', '<path_to_uploaded_file.csv>', '--experiment', 'all_features']`
        `cwd` is set to `fraud_detection_project/` for correct path resolution within the pipeline.
    4.  Waits for subprocess completion (timeout of 300s).
*   **`main.py` Execution:**
    1.  **Crucial Adaptation:** `main.py` must parse `--input-file` and use this path for data loading.
    2.  Runs the full pipeline: load, preprocess, train, evaluate, save models/results.
*   **Returning Results to UI:**
    1.  `app.py` checks subprocess return code.
    2.  If successful, reads `evaluation_metrics.json`.
    3.  Renders `results.html` with metrics or errors.
    4.  Deletes the uploaded CSV from `uploads/`.

### 3.4. Current "Deployment" Model

*   **Local Flask Development Server:** Runs via `app.run(debug=True)`, accessible locally (e.g., `http://127.0.0.1:5000/`).
*   **"Retrain-on-Upload" Model:** Each upload triggers a full model retraining and evaluation cycle on that dataset.
    *   **Pros:** Demonstrates pipeline performance on user-specific data; educational.
    *   **Cons:** Not scalable; resource-intensive; results are dataset-specific, not from a generalized model.

### 3.5. Alternative Deployment Model (for University Project Context/Discussion)

A typical prediction service would involve:
*   **Offline Model Training & Selection:** Train, tune, and select the best model(s) on representative data offline.
*   **Prediction Endpoint in UI for Inference:**
    *   Flask app loads pre-trained model(s) at startup.
    *   An endpoint (e.g., `/predict`) accepts new transaction data.
    *   Preprocesses new data (consistent with training).
    *   Uses loaded model(s) for prediction.
    *   Returns prediction.
This separates training from inference, unlike the current project's combined approach.

## 4. Summary of Experimental Design and Operational Characteristics

The project is designed as an experimental platform where the entire machine learning pipeline—from data preprocessing to model training and evaluation—is executed dynamically upon user data submission through a Flask web interface. Key aspects of its design and operation include:

*   **"Retrain-on-Upload" System:** The core characteristic is that models (Logistic Regression, Decision Tree, Random Forest, and a placeholder ANN) are not pre-trained but are instead trained from scratch on the dataset provided by the user during each session. This allows for assessing the pipeline's behavior and model performance on diverse datasets but differs from a production system where models are trained offline and deployed for inference.
*   **UI-Pipeline Integration:** The user interface (`fraud_detection_ui`) acts as an orchestrator, triggering the machine learning scripts (`fraud_detection_project`) as a subprocess. For this integration to function correctly, the `main.py` script within the ML project is critically required to accept an `--input-file` argument, allowing it to dynamically process the path to the user-uploaded data.
*   **ANN Placeholder:** The Artificial Neural Network (ANN) component included in the model training phase is currently a placeholder. It does not implement actual neural network training or evaluation, and its inclusion is for structural completeness, indicating where such a model could be integrated.
*   **Results and Evaluation:** Performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC, etc.) and confusion matrix visualizations are generated for each model based on the test split of the user-uploaded data. These results are then presented back to the user, offering insights into how the models performed on that specific dataset.

This setup is valuable for educational purposes, allowing for the demonstration of an end-to-end ML workflow and its response to different data inputs. However, for practical fraud detection deployment, a shift towards offline training of robust models and dedicated inference endpoints would be necessary.
