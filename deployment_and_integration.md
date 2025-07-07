# Deployment and Integration

This section describes how the machine learning pipeline (`fraud_detection_project/`) and the web user interface (`fraud_detection_ui/`) are structured and integrated, as well as the current "deployment" model of the system.

## 1. ML Pipeline (`fraud_detection_project/`) as a Standalone Unit

*   **Structure:** The `fraud_detection_project/` directory contains a complete, self-contained machine learning pipeline designed to be runnable independently.
    *   `src/`: Houses all core Python scripts for configuration (`config.py`), data preprocessing (`data_preprocessing.py`), model training (`train_models.py`), model evaluation (`evaluate_models.py`), and the main pipeline orchestration script (`main.py`).
    *   `data/`: Subdirectories for `raw` input data (like `dummy_fraud_dataset.csv` or the user-provided `synthetic_fraud_dataset.csv`) and `processed` data (train/test splits created during preprocessing).
    *   `models/`: Stores serialized trained machine learning models (e.g., `logistic_regression_model.joblib`).
    *   `results/`: Stores evaluation outputs, including `evaluation_metrics.json` for performance metrics and a `confusion_matrices/` subdirectory for saved PNG plots.
    *   `notebooks/`: Includes Jupyter notebooks for tasks like Exploratory Data Analysis (EDA), e.g., `eda_analysis_dummy_data.ipynb`.
    *   `requirements.txt`: Lists Python dependencies specifically for the ML pipeline (e.g., pandas, scikit-learn, matplotlib, seaborn).
*   **Execution:** The pipeline can be executed independently from the command line using `python src/main.py`. It can be configured via settings in `src/config.py` and command-line arguments (e.g., `--input-file <path_to_data>` and `--experiment <experiment_name>`). This modularity allows for development, testing, and batch processing of datasets without requiring the UI.

## 2. Web User Interface (`fraud_detection_ui/`)

*   **Structure:** The `fraud_detection_ui/` directory contains a Flask web application responsible for user interaction.
    *   `app.py`: The main Flask application file. It defines routes, handles HTTP requests, manages file uploads, orchestrates the ML pipeline execution, and prepares data for rendering in templates.
    *   `templates/`: Stores HTML templates (`index.html` for the file upload page, `results.html` for displaying processing outcomes) that are rendered by Flask using the Jinja2 templating engine.
    *   `static/`: Intended for static assets like CSS or JavaScript files. Currently, CSS is mostly inline in the HTML templates.
    *   `uploads/`: A temporary directory (created by `app.py` if it doesn't exist) to store CSV files uploaded by the user before they are passed to the ML pipeline.
*   **Functionality:** The UI provides a simple web interface for users to:
    *   Upload a CSV file containing transaction data.
    *   Receive feedback on the upload and processing status via flashed messages.
    *   View the performance metrics (read from the ML pipeline's output) of models that were trained and evaluated on their uploaded data.

## 3. Integration Mechanism: UI Orchestrating the ML Pipeline

The integration between the UI and the ML pipeline is achieved by the Flask application (`fraud_detection_ui/app.py`) invoking the ML pipeline's main script (`fraud_detection_project/src/main.py`) as an external subprocess.

*   **Trigger:** A user uploads a CSV file through the form on the `index.html` page, which sends a POST request to the `/upload` endpoint in `app.py`.
*   **`app.py` Logic:**
    1.  Receives and validates the uploaded file (checks for presence, filename, and `.csv` extension).
    2.  Saves the file temporarily to the `fraud_detection_ui/uploads/` directory using a secure filename.
    3.  Constructs the absolute paths to `fraud_detection_project/src/main.py` and the temporarily saved uploaded CSV file.
    4.  **Subprocess Call:** Executes `main.py` using `subprocess.run()`. A typical command constructed is:
        `['python', '<path_to_fraud_detection_project/src/main.py>', '--input-file', '<path_to_uploaded_file.csv>', '--experiment', 'all_features']`
        *   The `cwd` (current working directory) for the subprocess is critically set to the root of the `fraud_detection_project/` directory. This ensures that `main.py` can correctly resolve its internal relative paths to access `config.py`, and its `data/`, `models/`, and `results/` subdirectories.
    5.  `app.py` waits for the subprocess to complete, with a configurable timeout (e.g., 300 seconds).
*   **`main.py` (ML Pipeline) Execution:**
    1.  **Crucial Adaptation for Integration:** `main.py` must be implemented to parse the `--input-file` command-line argument. It then uses the provided file path as the source for data loading in `data_preprocessing.load_data()`, overriding any static path in `config.py`.
    2.  The pipeline then runs end-to-end: loads the specified user data, preprocesses it (including imbalance handling, feature engineering, scaling), trains all defined models from scratch using this data, evaluates them on a test split of this data, and saves the trained models, processed data, evaluation metrics (to `results/evaluation_metrics.json`), and confusion matrix plots.
*   **Returning Results to UI:**
    1.  After `main.py` finishes, `app.py` checks the return code of the subprocess.
    2.  If the subprocess was successful (return code 0), `app.py` attempts to read the `fraud_detection_project/results/evaluation_metrics.json` file to fetch the performance metrics.
    3.  It then renders the `results.html` template, passing these metrics (or error information if the pipeline failed or the metrics file wasn't found) for display to the user.
    4.  Regardless of success or failure, the temporarily uploaded CSV file in `fraud_detection_ui/uploads/` is deleted by `app.py` to clean up.

## 4. Current "Deployment" Model

*   **Local Flask Development Server:** The "deployment" of this integrated system, as structured, involves running the Flask application using its built-in development server (triggered by `app.run(debug=True)` in `app.py` when executed as the main script). This makes the UI accessible locally in a web browser, typically at `http://127.0.0.1:5000/`.
*   **"Retrain-on-Upload" Model:** This is not a traditional deployment where a pre-trained model is served for inference on new data. Instead, each user interaction involving a file upload triggers a full retraining and re-evaluation cycle of the ML models on that specific dataset provided by the user.
    *   **Pros:** Allows users to see how the defined pipeline performs on their unique data; excellent for demonstrating the end-to-end functionality of the ML pipeline and its components; useful for educational purposes or initial pipeline testing with various datasets.
    *   **Cons:** Not scalable for frequent prediction requests due to the time taken for model training; resource-intensive; results are specific to the characteristics of the uploaded data and the models trained on it, not from a generalized, robustly validated model. The models generated are overwritten with each new upload.

## 5. Alternative Deployment Model (for University Project Context/Discussion)

For a university project report, it's valuable to discuss the current model and then contrast it with a more typical deployment scenario for a real-world prediction service:

*   **Offline Model Training & Selection:** Models would be trained and rigorously evaluated offline on a large, representative, and well-curated dataset (e.g., an enhanced `synthetic_fraud_dataset.csv`). This process would involve hyperparameter tuning, cross-validation, and selection of the best-performing, most robust model(s). These final model(s) would then be saved.
*   **Prediction Endpoint in UI for Inference:** The Flask application (`app.py`) would:
    *   Load the pre-trained and selected model(s) once at startup (or on demand).
    *   Define a specific API endpoint (e.g., `/predict`) that accepts new, individual transaction data for inference (perhaps as a JSON payload or a small CSV with a single or few records).
    *   Preprocess this new incoming data using the *exact same* preprocessing steps and transformations (e.g., fitted scalers, encoders) that were used during the offline training of the selected model.
    *   Use the loaded model(s) to make a fraud prediction on the preprocessed new data.
    *   Return the prediction result (e.g., "Fraud" or "Not Fraud", possibly with a confidence score or explanation) to the client.
*   This alternative approach clearly separates the model training lifecycle from the prediction/inference lifecycle, which is standard practice for operational ML systems. The current project, however, tightly couples them, making it primarily an experimental or evaluative platform for the pipeline itself.

In summary, the integration in this project is achieved via process-level calls from the web UI (Flask) to the ML pipeline script. The current deployment model is geared towards local execution and demonstration, running the entire ML workflow on user-provided data. Understanding this distinction and discussing potential alternative deployment strategies is key when evaluating the project's practical applicability beyond its educational and demonstrative purpose.
