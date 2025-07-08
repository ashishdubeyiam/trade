from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import json # Added for reading metrics file

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.secret_key = 'super secret key' # Replace in production!

# Define APP_ROOT and PROJECT_ROOT_DIR for use in download_predictions
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(APP_ROOT)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            except Exception as e:
                flash(f"Error saving file '{filename}': {str(e)}", 'error')
                return redirect(url_for('index'))

            # Correctly use APP_ROOT and PROJECT_ROOT_DIR defined globally
            main_script_path = os.path.join(PROJECT_ROOT_DIR, 'fraud_detection_project', 'src', 'main.py')
            absolute_uploaded_file_path = os.path.abspath(file_path)
            ml_project_root_for_cwd = os.path.join(PROJECT_ROOT_DIR, 'fraud_detection_project')

            command = ['python', main_script_path, '--input-file', absolute_uploaded_file_path, '--experiment', 'all_features']

            print(f"Executing command: {' '.join(command)}")
            print(f"Using CWD for subprocess: {ml_project_root_for_cwd}")

            metrics_data = None
            error_message = None
            stderr_details = None
            prediction_files_info = [] # Initialize here to ensure it's always defined

            try:
                process = subprocess.run(command,
                                         capture_output=True,
                                         text=True,
                                         check=False,
                                         timeout=300, # 5 minutes
                                         cwd=ml_project_root_for_cwd)

                print("--- Subprocess STDOUT ---")
                print(process.stdout)
                print("--- Subprocess STDERR ---")
                print(process.stderr)

                if process.returncode == 0:
                    flash(f"Pipeline executed successfully for {filename}!", 'success')
                    metrics_file_path = os.path.join(ml_project_root_for_cwd, 'results', 'evaluation_metrics.json')
                    if os.path.exists(metrics_file_path):
                        with open(metrics_file_path, 'r') as f:
                            metrics_data = json.load(f)

                        if isinstance(metrics_data, dict):
                            predictions_dir_path = os.path.join(ml_project_root_for_cwd, 'results', 'predictions')
                            for model_key_from_metrics in metrics_data.keys():
                                safe_filename_part = model_key_from_metrics.lower().replace(" ", "_")
                                safe_filename_part = "".join(c if c.isalnum() or c == "_" else "" for c in safe_filename_part)
                                csv_filename = f"predictions_{safe_filename_part}.csv"

                                full_prediction_csv_path = os.path.join(predictions_dir_path, csv_filename)
                                if os.path.exists(full_prediction_csv_path):
                                    prediction_files_info.append({
                                        "display_name": model_key_from_metrics,
                                        "csv_filename": csv_filename
                                    })
                                else:
                                    print(f"Prediction CSV not found, won't offer for download: {full_prediction_csv_path}")
                    else:
                        error_message = "Evaluation metrics file not found. Pipeline may have completed but results are missing."
                        flash(error_message, 'error')

                    if process.stderr:
                         flash(f"Pipeline warnings/info (stderr):\n{process.stderr[:200]}...", 'info')
                else:
                    error_message = f"Pipeline execution failed for {filename}. Error code: {process.returncode}"
                    stderr_details = process.stderr
                    flash(error_message, 'error')
                    if stderr_details:
                         flash(f"Error details (see console log for full details):\n{stderr_details[:200]}...", 'error')

            except subprocess.TimeoutExpired:
                error_message = f"Pipeline execution for {filename} timed out after 5 minutes."
                flash(error_message, 'error')
                print("--- Subprocess TIMEOUT ---")
            except Exception as e:
                error_message = f"An error occurred while trying to run the pipeline: {str(e)}"
                flash(error_message, 'error')
                print(f"--- Subprocess EXCEPTION: {e} ---")

            finally: # Ensure cleanup happens
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up uploaded file: {file_path}")
                    except Exception as e_cleanup:
                        print(f"Error cleaning up file {file_path}: {e_cleanup}")

            return render_template('results.html',
                                   metrics_data=metrics_data,
                                   error_message=error_message,
                                   stderr_details=stderr_details,
                                   filename=filename,
                                   prediction_files_info=prediction_files_info)

        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(url_for('index'))

    return redirect(url_for('index')) # Should not be reached if POST and file handling occurs

@app.route('/download_predictions/<path:filename>')
def download_predictions(filename):
    # PROJECT_ROOT_DIR is defined globally
    predictions_directory = os.path.join(PROJECT_ROOT_DIR, 'fraud_detection_project', 'results', 'predictions')

    print(f"Download request for: {filename} from directory: {predictions_directory}")
    try:
        if '..' in filename or filename.startswith('/'): # Basic security check
            flash("Invalid filename.", "error")
            return redirect(url_for('index'))
        return send_from_directory(directory=predictions_directory, path=filename, as_attachment=True)
    except FileNotFoundError:
        flash(f"Error: File '{filename}' not found for download. Ensure the pipeline ran successfully and generated this file.", 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An error occurred while trying to download '{filename}': {str(e)}", 'error')
        return redirect(url_for('index'))

# Placeholder for /history route if it exists or will be added
# @app.route('/history/<int:run_id>')
# def history_detail(run_id):
#     # ... your logic for history ...
#     return render_template('history_detail.html', run_id=run_id)

if __name__ == '__main__':
    app.run(debug=True)
