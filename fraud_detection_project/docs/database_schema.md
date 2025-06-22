# Database Schema for Fraud Detection Project UI Results

This document outlines the database schema used by the Flask Web UI (`fraud_detection_ui/`) to store results from ML pipeline runs. The schema is defined using SQLAlchemy and is intended to be used with an MS SQL Server database (though SQLAlchemy provides abstraction for other databases too).

## 1. SQLAlchemy Model Definitions

These are the Python class definitions located in `fraud_detection_ui/app.py` that define the database tables and their relationships. `db` is the SQLAlchemy instance initialized with the Flask application.

```python
# In fraud_detection_ui/app.py, after db = SQLAlchemy(app):

class PipelineRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    original_filename = db.Column(db.String(255), nullable=False)
    status_message = db.Column(db.Text, nullable=True)
    # Relationship to ModelResult: One PipelineRun has many ModelResults
    results = db.relationship('ModelResult', backref='pipeline_run', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<PipelineRun {self.id} - {self.original_filename}>'

class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Foreign Key to link to the PipelineRun table
    pipeline_run_id = db.Column(db.Integer, db.ForeignKey('pipeline_run.id'), nullable=False)
    model_name = db.Column(db.String(100), nullable=False) # e.g., "Logistic Regression_all_features"

    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    roc_auc = db.Column(db.Float, nullable=True)

    tn = db.Column(db.Integer, nullable=True) # True Negatives
    fp = db.Column(db.Integer, nullable=True) # False Positives
    fn = db.Column(db.Integer, nullable=True) # False Negatives
    tp = db.Column(db.Integer, nullable=True) # True Positives

    specificity = db.Column(db.Float, nullable=True)
    geometric_mean = db.Column(db.Float, nullable=True)

    # Stores the filename of the confusion matrix plot image
    confusion_matrix_plot_path = db.Column(db.String(500), nullable=True)

    def __repr__(self):
        return f'<ModelResult {self.model_name} for Run {self.pipeline_run_id}>'
```

## 2. Pseudo-SQL DDL Representation

This is an approximate SQL Data Definition Language (DDL) representation of the tables that SQLAlchemy would generate, tailored for a typical SQL database. Specific MS SQL Server syntax (like `IDENTITY(1,1)` for auto-incrementing primary keys and `GETDATE()` for default timestamps) is used for illustration.

```sql
CREATE TABLE pipeline_run (
    id INTEGER NOT NULL PRIMARY KEY IDENTITY(1,1), -- Auto-incrementing primary key for MS SQL
    upload_timestamp DATETIME DEFAULT GETDATE(),   -- Default current timestamp in MS SQL
    original_filename VARCHAR(255) NOT NULL,
    status_message TEXT NULL                      -- Or VARCHAR(MAX) in MS SQL for long text
);

CREATE TABLE model_result (
    id INTEGER NOT NULL PRIMARY KEY IDENTITY(1,1), -- Auto-incrementing primary key for MS SQL
    pipeline_run_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT NULL,
    precision FLOAT NULL,
    recall FLOAT NULL,
    f1_score FLOAT NULL,
    roc_auc FLOAT NULL,
    tn INTEGER NULL,
    fp INTEGER NULL,
    fn INTEGER NULL,
    tp INTEGER NULL,
    specificity FLOAT NULL,
    geometric_mean FLOAT NULL,
    confusion_matrix_plot_path VARCHAR(500) NULL,
    FOREIGN KEY(pipeline_run_id) REFERENCES pipeline_run (id) ON DELETE CASCADE
);
```

### Notes on SQL DDL for MS SQL Server:
*   `IDENTITY(1,1)` is used for auto-incrementing primary keys.
*   `GETDATE()` is the MS SQL Server function for the current timestamp.
*   `TEXT` type in SQL is for large text; MS SQL Server also supports `VARCHAR(MAX)`. SQLAlchemy handles the abstraction.
*   `FLOAT` is a standard SQL type for floating-point numbers.
*   `VARCHAR(n)` is for variable-length strings.
*   `INTEGER` is for integer numbers.
*   `ON DELETE CASCADE` on the foreign key ensures that if a `pipeline_run` record is deleted, all its associated `model_result` records are also automatically deleted, maintaining data integrity.

## 3. Table Descriptions

*   **`pipeline_run` Table:**
    *   Stores one record for each invocation of the ML pipeline (typically triggered by a file upload via the UI).
    *   `id`: Unique auto-generated identifier for the run.
    *   `upload_timestamp`: Timestamp of when the run was initiated/data uploaded.
    *   `original_filename`: The name of the data file processed in this run.
    *   `status_message`: A message indicating the outcome of the run (e.g., "Success", or error details if it failed).

*   **`model_result` Table:**
    *   Stores one record for each machine learning model evaluated within a specific pipeline run.
    *   `id`: Unique auto-generated identifier for this model result entry.
    *   `pipeline_run_id`: Foreign key linking back to the `pipeline_run` table, indicating which run these results belong to.
    *   `model_name`: Name of the model (e.g., "Random Forest_all_features").
    *   Metrics columns (`accuracy` through `geometric_mean`): Store the calculated performance metrics for the model in that run. `nullable=True` as some metrics might not always be calculable.
    *   `confusion_matrix_plot_path`: Stores the filename of the saved confusion matrix plot for this model and run, allowing the UI to retrieve and display it.

This schema allows for tracking multiple pipeline runs over time and comparing the performance of different models within each run.
