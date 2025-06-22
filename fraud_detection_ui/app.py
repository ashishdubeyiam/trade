from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory # Added send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import json # Added for reading metrics file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc # For ordering query results
from sqlalchemy.orm import joinedload # For eager loading related objects
# import datetime # if using datetime for timestamps in models, db.func.current_timestamp() is used for now
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np # For np.arange

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- SQLAlchemy Configuration for MS SQL Server ---
# IMPORTANT: Replace with your actual MS SQL Server connection string.
# Using environment variables for sensitive parts like passwords is highly recommended for production.
# Example format for pyodbc: 'mssql+pyodbc://<USERNAME>:<PASSWORD>@<SERVER_NAME>/<DATABASE_NAME>?driver=ODBC+Driver+17+for+SQL+Server'
# For Windows Authentication, it might be like: 'mssql+pyodbc://@<SERVER_NAME>/<DATABASE_NAME>?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

# Check if an environment variable is set for the DB URI, otherwise use a placeholder/default.
# This placeholder will NOT work and needs to be configured by the user.
db_uri = os.environ.get('DATABASE_URL', 'mssql+pyodbc://USER:PASSWORD@SERVER/DATABASE_NAME?driver=ODBC+Driver+17+for+SQL+Server')

app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning

db = SQLAlchemy(app)

# Define root paths for accessing other project directories (needs to be after app init for some contexts if app is used)
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # Root of the UI app (fraud_detection_ui/)
PROJECT_ROOT_DIR = os.path.dirname(APP_ROOT) # Root of the entire repository (parent of fraud_detection_ui/ and fraud_detection_project/)
CONFUSION_MATRICES_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, 'fraud_detection_project', 'results', 'confusion_matrices'))

app.secret_key = 'super secret key'

# --- Database Models ---
class PipelineRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_timestamp = db.Column(db.DateTime, default=db.func.current_timestamp()) # Using db.func for database's current time
    original_filename = db.Column(db.String(255), nullable=False)
    status_message = db.Column(db.UnicodeText, nullable=True) # For NVARCHAR(MAX)
    results = db.relationship('ModelResult', backref='pipeline_run', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<PipelineRun {self.id} - {self.original_filename}>'

class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(db.Integer, db.ForeignKey('pipeline_run.id'), nullable=False)
    model_name = db.Column(db.String(100), nullable=False) # e.g., "Logistic Regression_all_features"

    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    roc_auc = db.Column(db.Float, nullable=True)

    tn = db.Column(db.Integer, nullable=True)
    fp = db.Column(db.Integer, nullable=True)
    fn = db.Column(db.Integer, nullable=True)
    tp = db.Column(db.Integer, nullable=True)

    specificity = db.Column(db.Float, nullable=True)
    geometric_mean = db.Column(db.Float, nullable=True)

    confusion_matrix_plot_path = db.Column(db.String(500), nullable=True)

    def __repr__(self):
        return f'<ModelResult {self.model_name} for Run {self.pipeline_run_id}>'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Chart Generation Function ---
def generate_performance_chart(run_id, model_results_for_run):
    # model_results_for_run is expected to be a list of ModelResult objects for a specific run
    if not model_results_for_run:
        return None

    model_names = [res.model_name for res in model_results_for_run]
    f1_scores = [res.f1_score if res.f1_score is not None else 0 for res in model_results_for_run]

    if not any(f1_scores): # Don't generate chart if all scores are 0 or None
        print(f"No valid F1 scores to plot for run_id {run_id}")
        return None

    try:
        plt.figure(figsize=(10, 6))
        display_names = []
        for name in model_names:
            parts = name.split('_') # e.g. "Logistic Regression_all_features"
            base_name = parts[0]
            if "Random Forest" in name: base_name = "Random Forest"
            elif "Logistic Regression" in name: base_name = "Logistic Regression"
            elif "Decision Tree" in name: base_name = "Decision Tree"
            elif "ANN" in name: base_name = "ANN" # Should match the key if ANN results are stored
            display_names.append(base_name)

        bars = plt.bar(display_names, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon'])
        plt.xlabel("Model")
        plt.ylabel("F1-Score")
        plt.title(f"Model F1-Scores for Run ID: {run_id}")
        plt.xticks(rotation=15, ha="right")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(0, 1.05)
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

        charts_dir = os.path.join(app.static_folder, 'generated_charts')
        os.makedirs(charts_dir, exist_ok=True)

        chart_filename = f"run_{run_id}_f1_scores.png"
        chart_save_path = os.path.join(charts_dir, chart_filename)

        plt.savefig(chart_save_path)
        plt.close()
        print(f"Generated chart: {chart_save_path}")
        return chart_filename
    except Exception as e:
        print(f"Error generating performance chart for run_id {run_id}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(url_for('index'))

        file = request.files['datafile']

        if file.filename == '':
            flash('No selected file.', 'error')
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(file_path)
                # Flash message moved to after pipeline attempt
            except Exception as e:
                flash(f"Error saving file '{filename}': {str(e)}", 'error')
                return redirect(url_for('index'))

            ui_root = os.path.dirname(os.path.abspath(__file__))
            project_root_dir = os.path.dirname(ui_root)
            main_script_path = os.path.join(project_root_dir, 'fraud_detection_project', 'src', 'main.py')
            absolute_uploaded_file_path = os.path.abspath(file_path)
            ml_project_root_for_cwd = os.path.join(project_root_dir, 'fraud_detection_project')

            command = ['python', main_script_path, '--input-file', absolute_uploaded_file_path, '--experiment', 'all_features']

            # --- Environment setup for subprocess ---
            # Define the src directory of the ML project for PYTHONPATH
            ml_project_src_dir = os.path.abspath(os.path.join(project_root_dir, 'fraud_detection_project', 'src'))

            # Get current environment and update PYTHONPATH
            env = os.environ.copy()
            existing_python_path = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{ml_project_src_dir}{os.pathsep}{existing_python_path}"
            # --- End environment setup ---

            print(f"Executing command: {' '.join(command)}")
            print(f"Using CWD for subprocess: {ml_project_root_for_cwd}")
            print(f"Setting PYTHONPATH for subprocess to: {env['PYTHONPATH']}")

            metrics_data = None
            error_message = None
            stderr_details = None

            try:
                process = subprocess.run(command,
                                         capture_output=True,
                                         text=True,
                                         check=False,
                                         timeout=300,
                                         cwd=ml_project_root_for_cwd,
                                         env=env) # Pass modified environment

                print("--- Subprocess STDOUT ---")
                print(process.stdout)
                print("--- Subprocess STDERR ---")
                print(process.stderr)

                pipeline_run_status = ""
                new_run = None

                if process.returncode == 0:
                    pipeline_run_status = "Success"
                    flash(f"Pipeline executed successfully for {filename}!", 'success')
                    metrics_file_path = os.path.join(ml_project_root_for_cwd, 'results', 'evaluation_metrics.json')

                    if os.path.exists(metrics_file_path):
                        with open(metrics_file_path, 'r') as f:
                            metrics_data = json.load(f)
                    else:
                        pipeline_run_status = "Success (metrics file not found)"
                        error_message = "Evaluation metrics file not found. Pipeline may have completed but results are missing."
                        flash(error_message, 'error')

                    if process.stderr: # Capture warnings/info from stderr on success
                         flash(f"Pipeline warnings/info (stderr):\n{process.stderr[:200]}...", 'info')
                         pipeline_run_status += f" (Warnings: {process.stderr[:200]})"

                    new_run = PipelineRun(original_filename=filename, status_message=pipeline_run_status)
                    db.session.add(new_run)
                    db.session.flush() # Get new_run.id

                    if metrics_data:
                        for model_key, metrics_values in metrics_data.items():
                            parts = model_key.split('_') # e.g. "Logistic Regression_all_features"
                            base_model_name_for_plot = parts[0]
                            if "Random Forest" in model_key: base_model_name_for_plot = "Random Forest"
                            elif "Logistic Regression" in model_key: base_model_name_for_plot = "Logistic Regression"
                            elif "Decision Tree" in model_key: base_model_name_for_plot = "Decision Tree"
                            elif "ANN" in model_key: base_model_name_for_plot = "ANN"

                            cm_plot_filename = base_model_name_for_plot.lower().replace(' ', '_') + '_cm.png'

                            model_result_entry = ModelResult(
                                pipeline_run_id=new_run.id, model_name=model_key,
                                accuracy=metrics_values.get('accuracy'), precision=metrics_values.get('precision'),
                                recall=metrics_values.get('recall'), f1_score=metrics_values.get('f1_score'),
                                roc_auc=metrics_values.get('roc_auc'), tn=metrics_values.get('tn'),
                                fp=metrics_values.get('fp'), fn=metrics_values.get('fn'),
                                tp=metrics_values.get('tp'), specificity=metrics_values.get('specificity'),
                                geometric_mean=metrics_values.get('geometric_mean'),
                                confusion_matrix_plot_path=cm_plot_filename
                            )
                            db.session.add(model_result_entry)
                    try:
                        db.session.commit()
                        flash('Results successfully saved to database.', 'success')
                    except Exception as e_db_commit:
                        db.session.rollback()
                        flash(f'Error saving results to database: {str(e_db_commit)}', 'error')
                        if new_run: # Update status if run entry was created
                            new_run.status_message = f"Success (DB save error: {str(e_db_commit)})"
                            try: db.session.commit()
                            except: db.session.rollback()

                else: # process.returncode != 0
                    error_message = f"Pipeline execution failed for {filename}. Error code: {process.returncode}"
                    stderr_details = process.stderr
                    flash(error_message, 'error')
                    if stderr_details:
                         flash(f"Error details (see console log for full details):\n{stderr_details[:200]}...", 'error')

                    failed_run = PipelineRun(
                        original_filename=filename,
                        status_message=f"Pipeline Failed (Code: {process.returncode}): {stderr_details[:1000]}" # Store more stderr
                    )
                    db.session.add(failed_run)
                    try: db.session.commit()
                    except Exception as e_db_fail:
                        db.session.rollback()
                        flash(f'Error saving failure status to database: {str(e_db_fail)}', 'error')

                # Cleanup before rendering results
                if os.path.exists(file_path):
                    try: os.remove(file_path); print(f"Cleaned up uploaded file: {file_path}")
                    except Exception as e_cl: print(f"Error cleaning up file {file_path}: {e_cl}")
                return render_template('results.html', metrics_data=metrics_data, error_message=error_message, stderr_details=stderr_details, filename=filename)

            except subprocess.TimeoutExpired:
                error_message = f"Pipeline execution for {filename} timed out after 5 minutes."
                flash(error_message, 'error')
                print("--- Subprocess TIMEOUT ---")
                timeout_run = PipelineRun(original_filename=filename, status_message="Pipeline Timed Out")
                db.session.add(timeout_run)
                try: db.session.commit()
                except Exception as e_db_timeout:
                    db.session.rollback()
                    flash(f'Error saving timeout status to database: {str(e_db_timeout)}', 'error')

                if os.path.exists(file_path):
                    try: os.remove(file_path); print(f"Cleaned up uploaded file after timeout: {file_path}")
                    except Exception as e_cl: print(f"Error cleaning up file {file_path} after timeout: {e_cl}")
                return render_template('results.html', error_message=error_message, filename=filename)

            except Exception as e: # Catch other exceptions during subprocess phase
                error_message = f"An error occurred while trying to run the pipeline: {str(e)}"
                flash(error_message, 'error')
                print(f"--- Subprocess EXCEPTION: {e} ---")
                exception_run = PipelineRun(original_filename=filename, status_message=f"Pipeline Exception: {str(e)}")
                db.session.add(exception_run)
                try: db.session.commit()
                except Exception as e_db_exc:
                    db.session.rollback()
                    flash(f'Error saving exception status to database: {str(e_db_exc)}', 'error')

                if os.path.exists(file_path):
                    try: os.remove(file_path); print(f"Cleaned up uploaded file after exception: {file_path}")
                    except Exception as e_cl: print(f"Error cleaning up file {file_path} after exception: {e_cl}")
                return render_template('results.html', error_message=error_message, filename=filename)

            # Note: The following cleanup lines are now effectively handled above,
            # as each path leading to render_template('results.html') now includes cleanup.
            # This specific location might become unreachable if all prior paths return.
            # However, if there was a path that didn't return render_template from the try/except block,
            # this would be a fallback. Given current logic, it's mostly redundant here.
            # For safety, can be kept or removed if all return paths are confirmed to handle cleanup.
            # To be safe and explicit, it's better handled before each specific return render_template.
            # Removing these specific lines as they are now handled more locally to each return.
            # try: # This block is now removed as cleanup is done before each render_template.
            #     os.remove(file_path)
            #     print(f"Removed uploaded file: {file_path}")
            # except OSError as e:
            #     print(f"Error removing file {file_path}: {e.strerror}")

        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    summary_stats = {
        'total_runs': 0,
        'successful_runs': 0,
        'failed_runs': 0,
        'last_run_status': None,
        'last_run_filename': None,
        'last_run_timestamp': None
    }
    recent_runs_query = []
    latest_run_chart_filename = None # Changed from path to filename

    try:
        # Query for summary statistics
        summary_stats['total_runs'] = db.session.query(PipelineRun).count()
        summary_stats['successful_runs'] = db.session.query(PipelineRun).filter(PipelineRun.status_message.ilike('Success%')).count()
        summary_stats['failed_runs'] = summary_stats['total_runs'] - summary_stats['successful_runs']

        last_run = db.session.query(PipelineRun).order_by(desc(PipelineRun.upload_timestamp)).first()
        if last_run:
            summary_stats['last_run_status'] = last_run.status_message
            summary_stats['last_run_filename'] = last_run.original_filename
            summary_stats['last_run_timestamp'] = last_run.upload_timestamp

        # Query for recent runs (e.g., last 5)
        recent_runs_query = db.session.query(PipelineRun).order_by(desc(PipelineRun.upload_timestamp)).limit(5).all()

        # Placeholder for chart generation logic (to be implemented in a later step)
        # For now, latest_run_chart_path remains None.
        # In a future step, we would query ModelResult for the last_run.id,
        # generate a chart image, save it, and set a path or filename here.
        # e.g., if a chart 'latest_run_f1_scores.png' was saved in a known static/image path:
        # latest_run_chart_path = url_for('static', filename='images/latest_run_f1_scores.png') # Example if served from UI's static
        # Or if served via serve_generated_image:
        # latest_run_chart_path = url_for('serve_generated_image', filename='latest_run_f1_scores.png')

        # Generate chart for the latest successful run and get prediction counts
        latest_successful_run = db.session.query(PipelineRun).filter(PipelineRun.status_message.ilike('Success%')).order_by(desc(PipelineRun.upload_timestamp)).first()

        # Initialize new summary stats for prediction counts
        summary_stats['latest_run_predicted_fraud'] = 'N/A'
        summary_stats['latest_run_predicted_legit'] = 'N/A'
        summary_stats['latest_run_model_for_counts'] = 'N/A'

        if latest_successful_run:
            model_results_for_latest_run = db.session.query(ModelResult).filter(ModelResult.pipeline_run_id == latest_successful_run.id).all()
            if model_results_for_latest_run:
                latest_run_chart_filename = generate_performance_chart(latest_successful_run.id, model_results_for_latest_run)

                # Try to find Random Forest results, otherwise use the first model's results for prediction counts
                chosen_model_result = None
                for res in model_results_for_latest_run:
                    if "Random Forest" in res.model_name:
                        chosen_model_result = res
                        break
                if not chosen_model_result and model_results_for_latest_run:
                    chosen_model_result = model_results_for_latest_run[0] # Fallback to the first model

                if chosen_model_result:
                    summary_stats['latest_run_model_for_counts'] = chosen_model_result.model_name
                    tp = chosen_model_result.tp if chosen_model_result.tp is not None else 0
                    fp = chosen_model_result.fp if chosen_model_result.fp is not None else 0
                    tn = chosen_model_result.tn if chosen_model_result.tn is not None else 0
                    fn = chosen_model_result.fn if chosen_model_result.fn is not None else 0

                    summary_stats['latest_run_predicted_fraud'] = tp + fp
                    summary_stats['latest_run_predicted_legit'] = tn + fn

    except Exception as e:
        flash(f"Error querying database for dashboard data: {str(e)}", "error")
        print(f"DB Query Error for Dashboard: {e}") # Server log

    return render_template('dashboard.html',
                           summary_stats=summary_stats,
                           recent_runs=recent_runs_query,
                           latest_run_chart_filename=latest_run_chart_filename) # Pass filename

@app.route('/generated_images/<path:filename>')
def serve_generated_image(filename):
    # Ensure the directory exists to prevent errors if it's somehow missing
    if not os.path.isdir(CONFUSION_MATRICES_DIR):
        print(f"ERROR: Confusion matrices directory not found: {CONFUSION_MATRICES_DIR}") # Using print for server log
        return "Image directory not found.", 404

    # Basic security: ensure filename is just a filename, not a path traversal attempt
    if '..' in filename or filename.startswith('/'):
        print(f"WARNING: Potentially unsafe filename requested: {filename}") # Using print for server log
        return "Invalid filename.", 400

    print(f"INFO: Attempting to serve image: {filename} from {CONFUSION_MATRICES_DIR}") # Using print for server log
    try:
        return send_from_directory(CONFUSION_MATRICES_DIR, filename)
    except FileNotFoundError:
        print(f"ERROR: Image not found: {filename} in {CONFUSION_MATRICES_DIR}") # Using print for server log
        return "Image not found.", 404

@app.route('/history')
def history():
    all_runs = []
    try:
        # Query for all runs, ordered by newest first
        all_runs = db.session.query(PipelineRun).order_by(desc(PipelineRun.upload_timestamp)).all()
    except Exception as e:
        flash(f"Error querying database for history data: {str(e)}", "error")
        print(f"DB Query Error for History: {e}") # Server log

    return render_template('history.html', runs=all_runs)

@app.route('/history/<int:run_id>')
def run_detail(run_id):
    pipeline_run_details = None
    try:
        # Query for the specific run and its associated model results
        # Using joinedload to efficiently fetch related ModelResult objects
        pipeline_run_details = db.session.query(PipelineRun).options(joinedload(PipelineRun.results)).filter(PipelineRun.id == run_id).first()

        if not pipeline_run_details:
            flash(f"Details for Run ID {run_id} not found.", "error")
            # Potentially redirect to history page or show a specific error template
            # For now, will render run_detail.html which handles pipeline_run_details being None
    except Exception as e:
        flash(f"Error querying database for run details: {str(e)}", "error")
        print(f"DB Query Error for Run Detail (ID: {run_id}): {e}") # Server log

    return render_template('run_detail.html', run=pipeline_run_details)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
