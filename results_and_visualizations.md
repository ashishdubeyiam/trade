# Results & Visualizations

This section outlines how the results from the fraud detection pipeline are captured, stored, and visualized. The primary goal is to assess and compare the performance of different machine learning models trained on the input data.

## 1. Quantitative Results (`fraud_detection_project/results/evaluation_metrics.json`)

The core quantitative outputs of the pipeline are the performance metrics for each trained model.

*   **Storage:**
    *   These metrics are saved in a JSON file named `evaluation_metrics.json` within the `fraud_detection_project/results/` directory.
    *   The file stores a dictionary where keys are typically a combination of the model name and the experiment type (e.g., `"Random Forest_all_features"`).
    *   The values are dictionaries containing the calculated performance metrics for that specific model and experiment run.
    *   The `evaluate_models.py` script (called by `main.py`) writes this file. Each run of `main.py` overwrites or creates this file with the metrics from the current dataset and models. When the UI triggers a run, it effectively evaluates models on the new dataset, and these new metrics are written.

*   **Key Metrics and Their Significance in Fraud Detection:**
    *   **Accuracy:** (`(TP + TN) / (TP + TN + FP + FN)`) - Represents the proportion of total transactions correctly classified. While a general measure, it can be misleading in imbalanced datasets (common in fraud) if the model simply predicts the majority class.
    *   **Precision:** (`TP / (TP + FP)`) - Of all transactions predicted as fraudulent, what proportion actually were fraudulent? High precision is crucial to minimize investigating legitimate transactions (false positives), which can be costly and damage customer trust. `zero_division=0` is used in its calculation.
    *   **Recall (Sensitivity):** (`TP / (TP + FN)`) - Of all actual fraudulent transactions, what proportion did the model correctly identify? High recall is vital to ensure as many fraudulent activities as possible are detected. `zero_division=0` is used in its calculation.
    *   **F1-Score:** (`2 * (Precision * Recall) / (Precision + Recall)`) - The harmonic mean of Precision and Recall. It provides a single score that balances the trade-off between catching fraud (recall) and avoiding false alarms (precision). Particularly useful for imbalanced datasets. `zero_division=0` is used in its calculation.
    *   **ROC-AUC Score:** (Area Under the Receiver Operating Characteristic Curve) - Measures the model's ability to distinguish between fraudulent and non-fraudulent transactions across all classification thresholds. An AUC of 1.0 represents a perfect classifier, while 0.5 suggests a model with no discriminative ability. Requires probability predictions from models.
    *   **True Positives (TP):** Fraudulent transactions correctly identified as fraud.
    *   **False Positives (FP):** Legitimate transactions incorrectly identified as fraud (Type I error).
    *   **True Negatives (TN):** Legitimate transactions correctly identified as legitimate.
    *   **False Negatives (FN):** Fraudulent transactions missed by the model and incorrectly identified as legitimate (Type II error) – often the most critical errors to minimize.
    *   **Specificity (True Negative Rate):** (`TN / (TN + FP)`) - Proportion of actual legitimate transactions that were correctly identified.
    *   **Geometric Mean (G-Mean):** (`sqrt(Recall * Specificity)`) - Aims to balance performance on both the majority (non-fraud) and minority (fraud) classes. It's particularly sensitive to performance on the minority class, making it a good indicator when dealing with imbalanced data.

*   **Interpretation via UI:**
    *   The web UI (`fraud_detection_ui/templates/results.html`) reads this `evaluation_metrics.json` file after a pipeline run (triggered by user data upload).
    *   It then presents these metrics in a tabular format for each model, allowing the user to see how the models performed on the specific dataset they uploaded. Values are typically rounded for display.

## 2. Visualizations

The project generates visualizations to aid in understanding model performance and data characteristics.

*   **Confusion Matrix Plots:**
    *   **Purpose:** To provide a detailed visual breakdown of a classifier's performance, showing the counts of True Positives (TPs), False Positives (FPs), True Negatives (TNs), and False Negatives (FNs). This helps in understanding the types and magnitudes of errors the model is making.
    *   **Generation:** The `evaluate_models.py` script uses `matplotlib.pyplot` and `seaborn.heatmap` to create a confusion matrix plot for each trained model. The plot includes labels for axes, a title indicating the model, and annotations for the counts within each cell of the heatmap.
    *   **Storage:** These plots are saved as PNG image files (e.g., `logistic_regression_cm.png`, `random_forest_cm.png`) in the `fraud_detection_project/results/confusion_matrices/` directory, as specified by `config.CONFUSION_MATRIX_DIR`.
    *   **UI Indication:** The `results.html` page informs the user of the path where these images are saved but currently does not display them directly in the browser (noted as a future enhancement). Users need to access the project's file system to view them.

*   **Exploratory Data Analysis (EDA) Visualizations (`fraud_detection_project/notebooks/eda_analysis_dummy_data.ipynb`):**
    *   **Purpose:** The Jupyter notebook is provided as a template to perform EDA on the transaction data. EDA helps in understanding data distributions, identifying outliers, relationships between features, feature importance, and correlations with the target variable (`Fraud_Label`).
    *   **Types of Plots (Examples from the notebook using `matplotlib` and `seaborn`):**
        *   **Histograms:** For numerical features (e.g., 'Transaction_Amount', 'Account_Balance') to show their distributions and skewness.
        *   **Box Plots:** For numerical features, often compared by 'Fraud_Label', to identify outliers and differences in central tendency and spread between fraudulent and non-fraudulent transactions.
        *   **Count Plots (Bar Charts):** For categorical features (e.g., 'Transaction_Type', 'Device_Type', 'Location') to show category frequencies and their relationship with fraud.
        *   **Categorical Features vs. Target:** Stacked or grouped bar charts showing the distribution of the 'Fraud_Label' within each category of selected features (e.g., fraud incidence per 'Transaction_Type').
        *   **Correlation Heatmap:** Visualizes the pairwise Pearson correlation matrix of numerical features (and potentially the target variable if numerically encoded). This helps identify multicollinearity or strong predictors.
        *   **Scatter Plots:** For visualizing relationships between two numerical features, possibly colored by the target variable.
    *   **Caveat:** The notebook is pre-configured for `dummy_fraud_dataset.csv`. As stated in the `implementation_guide.md` and the notebook itself, EDA on this tiny dataset is for **demonstration of process only**. For meaningful insights, the notebook should be adapted (e.g., file paths, feature names if different) and run on a larger, more realistic dataset (like the user's `synthetic_fraud_dataset.csv` or an uploaded dataset). The visualizations from the dummy data will not be representative of real-world fraud patterns.

## 3. Interpreting Results in the Project's Context

When a user uploads data via the UI:
1.  The ML pipeline runs, training models *on that specific data*.
2.  The `evaluation_metrics.json` is populated with the performance of these newly trained models on a test split *of that same uploaded data*.
3.  The UI displays these metrics, and confusion matrix plots are saved to disk.

Therefore, the "results" shown are a direct reflection of how well the pipeline could model the particular dataset provided by the user at that moment. They are not results from a fixed, pre-trained model being applied to new data for inference in a production sense. This is valuable for understanding dataset characteristics, pipeline behavior, and relative model performance on that data, but it's different from typical inference results.

For a university project, this means that experiments would involve:
*   Preparing one or more suitable datasets (e.g., the `synthetic_fraud_dataset.csv`, ensuring it has the features expected by `config.py`).
*   Running the pipeline (either manually via `main.py --input-file <path_to_data>` or via the UI by uploading the dataset).
*   Analyzing the generated `evaluation_metrics.json` and the saved confusion matrix plots to compare model performances on that specific dataset.
*   Potentially modifying `config.py` (e.g., model hyperparameters, feature sets for the `selected_features` experiment by updating `FEATURES_TO_DROP_EXPERIMENT2`) and re-running to observe changes in performance metrics.
*   Using the EDA notebook on the actual dataset to derive insights that might inform feature engineering choices, understand data quality issues, or guide model selection.

The framework provides the tools to generate these results and visualizations; the interpretation depends heavily on the quality, characteristics, and representativeness of the input dataset used for the experiments.
