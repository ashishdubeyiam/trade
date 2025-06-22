# Fraud Detection Project

## 1. Project Description
This project aims to develop a machine learning pipeline for detecting fraudulent transactions. It is based on a research proposal that outlines data preprocessing steps, model training, and evaluation. The pipeline includes:
- Data loading and initial inspection.
- Feature engineering, including timestamp parsing and dropping irrelevant identifiers.
- Preprocessing steps like one-hot encoding for categorical features and scaling for numerical features.
- Handling class imbalance using undersampling.
- Training several classification models: Logistic Regression, Random Forest, Decision Tree.
- A placeholder for an Artificial Neural Network (ANN) model.
- Evaluating model performance using various metrics and generating confusion matrices.

## 2. Dataset
- The project is designed to work with a dataset named `synthetic_fraud_dataset.csv`. This file is expected to be provided by the user and placed in the `fraud_detection_project/data/` directory.
- For testing the pipeline's execution flow and structure, a `dummy_fraud_dataset.csv` is included in the `data/` directory.
- The expected dataset schema (based on the dummy data and user proposal) should include columns such as:
  `Transaction_ID`, `User_ID`, `Transaction_Amount`, `Transaction_Type`, `Timestamp`, `Account_Balance`, `Device_Type`, `Location`, `Merchant_Category`, `IP_Address_Flag`, `Previous_Fraudulent_Activity`, `Daily_Transaction_Count`, `Avg_Transaction_Amount_7d`, `Failed_Transaction_Count_7d`, `Card_Type`, `Card_Age`, `Transaction_Distance`, `Authentication_Method`, `Risk_Score`, `Is_Weekend`.
- The target variable for fraud classification is `Fraud_Label`.

## 3. Setup Instructions
- **Prerequisites**: Python 3.7+ is recommended.
- **Create a Virtual Environment** (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- **Install Dependencies**:
  Navigate to the `fraud_detection_project` root directory (where `requirements.txt` is located) and run:
  ```bash
  pip install -r requirements.txt
  ```

## 4. Running the Pipeline
- The main script to run the end-to-end pipeline is `src/main.py`.

- **Configuration**:
  - Dataset configuration is managed in `src/config.py`. The `RAW_DATA_FILE` variable specifies the input dataset.
  - After cloning, it might point to `dummy_fraud_dataset.csv`. To use your actual data, change this to `synthetic_fraud_dataset.csv` (or your specific filename) and ensure the file is in the `data/` directory.
    ```python
    # Example in src/config.py
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'synthetic_fraud_dataset.csv')
    ```

- **Execution**:
  Navigate to the root directory of the project (`fraud_detection_project/`).
  Run the pipeline using the following command:
  ```bash
  python src/main.py --experiment <experiment_type>
  ```
  - `<experiment_type>` can be:
    - `all_features`: Uses all features as processed by `data_preprocessing.py` (default). This includes the engineered time-based features and drops configured ID columns.
    - `selected_features`: In addition to the default processing, this option drops features specified in `config.FEATURES_TO_DROP_EXPERIMENT2` (e.g., 'nameOrig', 'nameDest'). Note: These specific columns ('nameOrig', 'nameDest') are from an older proposal and are not present in the current `dummy_fraud_dataset.csv` or the user-provided schema.

- **Output**:
  - **Trained Models**: Saved as `.joblib` files (or `.h5` for ANN placeholder) in the `models/` directory.
  - **Evaluation Metrics**: Stored in JSON format in `results/evaluation_metrics.json`. This file is updated if it already exists.
  - **Confusion Matrices**: Plots are saved as PNG images in the `results/confusion_matrices/` directory.
  - **Processed Data**: The training and testing datasets generated after preprocessing are saved as CSV files in `data/processed/`.

## 4.5. Running the Web UI (Optional)

This project also includes a basic web user interface (UI) built with Flask for uploading data and viewing results. The UI code is located in the `fraud_detection_ui/` directory, which should be a sibling directory to this `fraud_detection_project/` directory at the root of the repository.

**Prerequisites for UI:**
- All prerequisites for the main pipeline must be met (Python, virtual environment, and all dependencies in `requirements.txt` installed, including `Flask`).

**Steps to Run the Web UI:**

1.  **Ensure ML Pipeline is Ready:** The UI calls the ML pipeline.
2.  **Navigate to UI Directory:** From the root directory of the repository (e.g., where `fraud_detection_project/` and `fraud_detection_ui/` are located), change to the UI directory:
    ```bash
    cd fraud_detection_ui
    ```
3.  **Activate Virtual Environment:** Ensure the virtual environment containing Flask and other project dependencies is active. If your virtual environment was created at the repository root or within `fraud_detection_project/`, ensure your `PYTHONPATH` or activation makes `flask` available.
4.  **Run the Flask App:**
    ```bash
    python app.py
    ```
5.  **Access the UI:** Open your web browser and go to the address shown in the Flask development server terminal, usually:
    ```
    http://127.0.0.1:5000/
    ```
6.  **Using the UI:**
    - The **Dashboard** page (usually the new home page or accessible via `/dashboard`) provides summary statistics, a view of the latest run's F1 score chart, and lists recent pipeline runs.
    - The **New Upload** page (`/` or linked as "New Upload") allows you to upload a CSV transaction data file.
    - The **History** page (`/history`) displays a comprehensive list of all past pipeline runs stored in the database.
    - From the Dashboard or History page, you can navigate to a **Detailed Results** view for each specific run, showing all model metrics and confusion matrix plots.
    - Upon uploading a file, the backend triggers the ML pipeline located in `fraud_detection_project/src/main.py` using the uploaded file. Results and run information are saved to the configured MS SQL database.
    - Uploaded files are temporarily stored in `fraud_detection_ui/uploads/` and are cleaned up after processing.
    - ML models and results (JSON, plots) generated by the pipeline will still be saved within the `fraud_detection_project/models/` and `fraud_detection_project/results/` directories, respectively. The database stores the metrics and paths to these plots.

### Database Setup for UI Results Storage (MS SQL Server)

The Web UI now supports saving pipeline run results and model metrics to a database, allowing for history viewing and a dashboard. This feature is configured to target an MS SQL Server database.

**Additional Dependencies:**
- Ensure `Flask-SQLAlchemy` and `pyodbc` are installed (they are included in `requirements.txt`).
- You may need to install appropriate Microsoft ODBC drivers for SQL Server on the machine where the Flask Web UI (`app.py`) will run, if they are not already present. The specific driver depends on your SQL Server version (e.g., 'SQL Server Native Client 11.0' for SQL Server 2014, or 'ODBC Driver 17 for SQL Server' for newer versions).

**Configuration:**
1.  **MS SQL Server Instance:** You must have an MS SQL Server instance running and accessible. You also need to create a database on this server for the application to use.
2.  **Connection String:** The Flask application (`fraud_detection_ui/app.py`) needs a connection string to connect to your MS SQL Server. This is configured via the `SQLALCHEMY_DATABASE_URI`.
    - **Recommended:** Set an environment variable named `DATABASE_URL` with your full MS SQL Server connection string.
      Example format (using `pyodbc`):
      `mssql+pyodbc://USERNAME:PASSWORD@SERVER_ADDRESS/DATABASE_NAME?driver=ODBC+Driver+17+for+SQL+Server`
      (Replace `USERNAME`, `PASSWORD`, `SERVER_ADDRESS`, `DATABASE_NAME`, and the `driver` name with your actual details. For SQL Server 2014, the driver might be `SQL Server Native Client 11.0` or `SQL Server`).
    - Alternatively (less secure for credentials), you can directly modify the placeholder URI in `fraud_detection_ui/app.py`, but this is not advised for production or shared environments.
3.  **Table Creation:** When `fraud_detection_ui/app.py` is first run with a correct database connection, SQLAlchemy will attempt to create the necessary tables (`pipeline_run`, `model_result`) if they don't already exist.

**Note on UI and Database:**
- The UI's `app.py` correctly locates the ML pipeline's `main.py` script assuming `fraud_detection_ui/` and `fraud_detection_project/` are sibling directories at the repository root.
- The development of database interaction features by the AI agent was done without direct access to an MS SQL Server. Therefore, successful database operations depend entirely on the user's correct setup and configuration of their MS SQL Server environment and the connection string.

## 5. Project Structure
```
fraud_detection_project/
├── data/
│   ├── dummy_fraud_dataset.csv   # Small dummy dataset for testing the pipeline
│   ├── processed/                # Stores train.csv and test.csv after preprocessing
│   └── .gitkeep                  # (User should place synthetic_fraud_dataset.csv here)
├── models/                       # Stores trained model files (e.g., .joblib, .h5)
├── notebooks/
│   └── eda_analysis_dummy_data.ipynb # Jupyter notebook for EDA on the dummy data
├── results/
│   ├── confusion_matrices/       # Stores confusion matrix plots (e.g., .png)
│   └── evaluation_metrics.json   # JSON file with performance metrics of models
├── src/
│   ├── config.py                 # Central configuration file for paths, parameters, feature lists
│   ├── data_preprocessing.py     # Script for data loading, cleaning, feature engineering, and splitting
│   ├── evaluate_models.py        # Script for evaluating trained models
│   ├── main.py                   # Main executable script to run the entire pipeline
│   └── train_models.py           # Script for training different classification models
├── README.md                     # This project documentation file
└── requirements.txt              # List of Python dependencies for the project
```

## 6. Key Scripts Description
- **`src/config.py`**: Manages all critical configurations: paths to data, models, and results; model hyperparameters (examples provided); lists of features for processing (categorical, columns to drop, timestamp column).
- **`src/data_preprocessing.py`**:
    - Loads the raw dataset specified in `config.py`.
    - Performs data inspection (shape, info, missing values, target distribution).
    - Implements feature engineering:
        - Parses the `Timestamp` column to create `Hour_of_Day`, `Day_of_Week`, `Month_of_Year`.
        - Drops `Transaction_ID` and `User_ID` as per `config.COLUMNS_TO_DROP`.
    - Handles class imbalance using undersampling of the majority class.
    - Applies `ColumnTransformer` for preprocessing: one-hot encoding for categorical features and `StandardScaler` for numerical features.
    - Splits the processed data into training and testing sets and saves them.
- **`src/train_models.py`**:
    - Takes processed training data.
    - Trains Logistic Regression, Random Forest, and Decision Tree classifiers using parameters from `config.py`.
    - Includes a placeholder function for training an ANN model (currently saves a placeholder file).
    - Saves the trained models to the directory specified in `config.MODEL_DIR`.
- **`src/evaluate_models.py`**:
    - Loads the trained models and the processed test data.
    - Predicts on the test set.
    - Calculates a suite of metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix (TN, FP, FN, TP), specificity, and geometric mean.
    - Saves confusion matrix plots.
    - Aggregates metrics from all models into a JSON file.
- **`src/main.py`**:
    - Parses command-line arguments (currently for `experiment_type`).
    - Orchestrates the pipeline by calling the relevant functions from `data_preprocessing.py`, `train_models.py`, and `evaluate_models.py` in sequence.

## 7. Notes
- **Feature Engineering**: The current feature engineering strategy involves creating time-based features (`Hour_of_Day`, `Day_of_Week`, `Month_of_Year`) from the `Timestamp` column and dropping `Transaction_ID` and `User_ID`.
- **Categorical Feature Encoding**: Categorical features (e.g., `Location`, `Transaction_Type`) are handled using one-hot encoding. If a feature like `Location` has very high cardinality in the real dataset, this could lead to a very large feature space, potentially causing performance issues or memory errors. For such cases, alternative encoding strategies (e.g., target encoding, feature hashing, or dimensionality reduction) might be necessary. This aspect was not fully testable with the low-cardinality dummy data.
- **ANN Model**: The Artificial Neural Network (ANN) model component in `train_models.py` and `evaluate_models.py` is currently a placeholder. Full implementation would require defining the network architecture (layers, activations, etc.) and using TensorFlow/Keras for training and saving/loading the model.
- **Dataset**: The project's functionality with the actual `synthetic_fraud_dataset.csv` is dependent on the user providing this file in the `data/` directory and ensuring its schema aligns with the processing logic (especially column names defined in `config.py`).
