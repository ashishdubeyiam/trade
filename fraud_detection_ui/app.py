from flask import Flask, render_template, request, redirect, url_for, flash
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

app.secret_key = 'super secret key'

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

                if process.returncode == 0:
                    flash(f"Pipeline executed successfully for {filename}!", 'success')
                    metrics_file_path = os.path.join(ml_project_root_for_cwd, 'results', 'evaluation_metrics.json')
                    if os.path.exists(metrics_file_path):
                        with open(metrics_file_path, 'r') as f:
                            metrics_data = json.load(f)
                    else:
                        error_message = "Evaluation metrics file not found. Pipeline may have completed but results are missing."
                        flash(error_message, 'error')
                    # Capture stderr even on success for potential warnings from scripts
                    if process.stderr:
                         flash(f"Pipeline warnings/info (stderr):\n{process.stderr[:200]}...", 'info')
                else:
                    error_message = f"Pipeline execution failed for {filename}. Error code: {process.returncode}"
                    stderr_details = process.stderr
                    flash(error_message, 'error')
                    if stderr_details:
                         flash(f"Error details (see console log for full details):\n{stderr_details[:200]}...", 'error')

                # Cleanup before rendering results
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up uploaded file: {file_path}")
                    except Exception as e_cleanup:
                        print(f"Error cleaning up file {file_path}: {e_cleanup}")
                return render_template('results.html', metrics_data=metrics_data, error_message=error_message, stderr_details=stderr_details, filename=filename)

            except subprocess.TimeoutExpired:
                error_message = f"Pipeline execution for {filename} timed out after 5 minutes."
                flash(error_message, 'error')
                print("--- Subprocess TIMEOUT ---")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up uploaded file after timeout: {file_path}")
                    except Exception as e_cleanup:
                        print(f"Error cleaning up file {file_path} after timeout: {e_cleanup}")
                return render_template('results.html', error_message=error_message, filename=filename)
            except Exception as e:
                error_message = f"An error occurred while trying to run the pipeline: {str(e)}"
                flash(error_message, 'error')
                print(f"--- Subprocess EXCEPTION: {e} ---")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up uploaded file after exception: {file_path}")
                    except Exception as e_cleanup:
                        print(f"Error cleaning up file {file_path} after exception: {e_cleanup}")
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

if __name__ == '__main__':
    app.run(debug=True)
