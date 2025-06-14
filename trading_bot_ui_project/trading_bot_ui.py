from flask import Flask, render_template, request, url_for
import os
import subprocess
import logging
import requests # For fetching signal from signal_server
import json     # For pretty-printing JSON

# --- Logging Setup ---
# Basic logging for the UI server itself
log_file_ui = 'trading_bot_ui.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.StreamHandler()]) # Add FileHandler if persistent UI logs are needed
logger_ui = logging.getLogger(__name__)


# --- Configuration for Bot Script Execution ---
# Assuming trading_bot_ui.py is in trading_bot_ui_project/
# and oanda_trading_bot/ (the main project with backend scripts) is a sibling directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to 'trading_bot_ui_project'
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Points to the directory containing both 'trading_bot_ui_project' and 'oanda_trading_bot'

BOT_PROJECT_NAME = 'oanda_trading_bot'
BOT_SCRIPT_DIR = os.path.join(PROJECT_ROOT, BOT_PROJECT_NAME)
logger_ui.info(f"Calculated BOT_SCRIPT_DIR: {BOT_SCRIPT_DIR}")


# Path to python executable.
# For Docker/venv, this should ideally point to the venv's python.
# If UI and Bot scripts share the same venv and are run from it, 'python' might suffice.
# For local dev, if you activate venv then run `python trading_bot_ui.py`, 'python' is fine.
PYTHON_EXECUTABLE = 'python'
# Example for a specific venv if UI is outside and needs to call bot's venv python:
# PYTHON_EXECUTABLE = os.path.join(BOT_SCRIPT_DIR, 'venv', 'Scripts', 'python.exe') # Windows example
# PYTHON_EXECUTABLE = os.path.join(BOT_SCRIPT_DIR, 'venv', 'bin', 'python') # Linux/macOS example
logger_ui.info(f"Using PYTHON_EXECUTABLE: {PYTHON_EXECUTABLE}")

SIGNAL_SERVER_URL = os.environ.get('SIGNAL_SERVER_URL', 'http://127.0.0.1:5000/getsignal')
logger_ui.info(f"Signal Server URL configured to: {SIGNAL_SERVER_URL}")


ALLOWED_SCRIPTS = [
    "intraday_data_downloader.py", # Uses OANDA now
    "intraday_data_labeler.py",    # Uses feature engineer
    "intraday_model_trainer.py",   # Trains initial M5 model
    "intraday_model_retrainer.py", # Retrains M5 model
    "forex_data_downloader.py",    # Updates daily master data (AlphaVantage)
    "economic_calendar_scraper.py",# Scrapes ForexFactory
    "news_headlines_fetcher.py",   # Fetches RSS news
    # "signal_server.py" # Usually run separately as a persistent server
]
# Ensure these script names are correct and they exist in BOT_SCRIPT_DIR

# Maps user-friendly key to actual log filename
# Filenames are relative to BOT_SCRIPT_DIR, except for 'ui_server'
ALLOWED_LOG_FILES = {
    "oanda_downloader": "intraday_data_downloader_oanda.log",
    "labeler": "intraday_data_labeler.log",
    "m5_trainer": "intraday_model_trainer.log",
    "m5_retrainer": "intraday_model_retrainer.log",
    "daily_downloader": "forex_data_downloader.log",
    "calendar_scraper": "economic_calendar_scraper.log",
    "news_fetcher": "news_headlines_fetcher.log",
    "signal_server_app": "signal_server.log",
    "signal_server_stdout": "logs/python_stdout.log", # Assumes signal_server is in BOT_SCRIPT_DIR and logs to its 'logs' subdir
    "ui_server": "trading_bot_ui.log" # Log for this UI app itself, relative to BASE_DIR
}


@app.route('/')
def index():
    # Pass the list of allowed scripts and log files to the template
    return render_template('index.html', title='Trading Bot Control Panel',
                           allowed_scripts=ALLOWED_SCRIPTS,
                           allowed_log_files=ALLOWED_LOG_FILES.keys())

@app.route('/run_script')
def run_script():
    script_to_run = request.args.get('name')
    logger_ui.info(f"Received request to run script: {script_to_run}")

    if not script_to_run:
        logger_ui.warning("No script name provided in request.")
        return "Error: No script name specified.", 400

    if script_to_run not in ALLOWED_SCRIPTS:
        logger_ui.error(f"Attempt to run disallowed script: {script_to_run}")
        return "Error: Invalid or disallowed script specified.", 400

    script_path = os.path.join(BOT_SCRIPT_DIR, script_to_run)
    logger_ui.info(f"Full path to script: {script_path}")

    if not os.path.exists(script_path):
        logger_ui.error(f"Script file not found at: {script_path} (Resolved from BOT_SCRIPT_DIR: {BOT_SCRIPT_DIR})")
        return f"Error: Script file '{script_to_run}' not found on server.", 404

    output = ""
    try:
        logger_ui.info(f"Executing script: {PYTHON_EXECUTABLE} {script_path} in cwd: {BOT_SCRIPT_DIR}")
        # Environment variables for the subprocess:
        # By default, subprocess.run inherits the environment of the parent (Flask app).
        # If Flask app's env has OANDA_API_TOKEN etc., scripts should pick them up.
        # For more control, you can construct an 'env' dict:
        # current_env = os.environ.copy()
        # current_env["EXTRA_VAR_FOR_SCRIPT"] = "some_value"

        process_result = subprocess.run(
            [PYTHON_EXECUTABLE, script_path],
            capture_output=True,
            text=True,
            cwd=BOT_SCRIPT_DIR,
            check=False,
            timeout=300  # 5 minutes timeout
        )

        stdout = process_result.stdout if process_result.stdout else ""
        stderr = process_result.stderr if process_result.stderr else ""

        output = f"--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}"

        if process_result.returncode != 0:
            error_message = f"\n\n--- ERROR: Script exited with code {process_result.returncode} ---"
            output += error_message
            logger_ui.error(f"Script {script_to_run} failed. Return code: {process_result.returncode}. Output:\n{output}")
        else:
            logger_ui.info(f"Script {script_to_run} executed successfully. Output:\n{output}")

    except subprocess.TimeoutExpired:
        timeout_message = f"Error: Script {script_to_run} timed out after 300 seconds."
        output = timeout_message
        logger_ui.error(timeout_message)
    except FileNotFoundError as fnf_error: # If PYTHON_EXECUTABLE is wrong
        error_message = f"Error executing script {script_to_run}: Python executable '{PYTHON_EXECUTABLE}' not found. Please check server configuration. Details: {str(fnf_error)}"
        output = error_message
        logger_ui.critical(error_message, exc_info=True)
    except Exception as e:
        error_message = f"An unexpected error occurred while executing script {script_to_run}: {str(e)}"
        output = error_message
        logger_ui.error(error_message, exc_info=True)

    return render_template('script_output.html', title=f"Output of {script_to_run}", script_name=script_to_run, output=output)


@app.route('/view_log/<log_key>')
def view_log(log_key):
    logger_ui.info(f"Received request to view log for key: {log_key}")

    if log_key not in ALLOWED_LOG_FILES:
        logger_ui.error(f"Invalid log key specified: {log_key}")
        return "Error: Invalid log file specified.", 400

    log_filename_or_path = ALLOWED_LOG_FILES[log_key]

    if log_key == "ui_server":
        full_log_path = os.path.join(BASE_DIR, log_filename_or_path)
    else:
        full_log_path = os.path.join(BOT_SCRIPT_DIR, log_filename_or_path)

    logger_ui.info(f"Attempting to read log file: {full_log_path} (Key: {log_key}, Path from dict: {log_filename_or_path})")

    log_content = ""
    file_exists = os.path.exists(full_log_path)

    if file_exists:
        try:
            with open(full_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            if not log_content.strip():
                log_content = f"--- Log file '{log_filename_or_path}' is empty or contains only whitespace. ---"
                logger_ui.info(f"Log file {full_log_path} is empty or whitespace only.")
            else:
                logger_ui.info(f"Successfully read log file {full_log_path}.")
        except Exception as e:
            log_content = f"Error reading log file '{log_filename_or_path}': {str(e)}"
            logger_ui.error(f"Error reading log file {full_log_path}: {e}", exc_info=True)
    else:
        log_content = f"Error: Log file '{log_filename_or_path}' not found at resolved path: {full_log_path}"
        logger_ui.error(log_content)

    return render_template('view_log.html', title=f"Log: {log_filename_or_path}", log_filename=log_filename_or_path, log_content=log_content)

@app.route('/current_signal')
def current_signal():
    logger_ui.info(f"Fetching current signal from: {SIGNAL_SERVER_URL}")
    signal_data_str = "Could not fetch signal due to an internal UI server error."
    try:
        response = requests.get(SIGNAL_SERVER_URL, timeout=10) # 10-second timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        signal_data_json = response.json()
        signal_data_str = json.dumps(signal_data_json, indent=4) # Pretty-print
        logger_ui.info(f"Successfully fetched signal data from signal server.")
    except requests.exceptions.Timeout:
        signal_data_str = f"Error fetching signal: Timeout after 10 seconds from {SIGNAL_SERVER_URL}"
        logger_ui.error(signal_data_str)
    except requests.exceptions.HTTPError as http_err:
        signal_data_str = f"Error fetching signal: HTTP error {http_err.response.status_code} from {SIGNAL_SERVER_URL}. Response: {http_err.response.text}"
        logger_ui.error(signal_data_str)
    except requests.exceptions.RequestException as req_err:
        signal_data_str = f"Error fetching signal: Request exception {str(req_err)} from {SIGNAL_SERVER_URL}"
        logger_ui.error(signal_data_str)
    except json.JSONDecodeError:
        signal_data_str = f"Error: Could not decode JSON response from {SIGNAL_SERVER_URL}. Response text: {response.text if 'response' in locals() else 'N/A'}"
        logger_ui.error(signal_data_str)
    except Exception as e:
        signal_data_str = f"An unexpected error occurred while fetching signal: {str(e)}"
        logger_ui.error(signal_data_str, exc_info=True)

    return render_template('view_signal.html', title="Current Trading Signal", signal_json_str=signal_data_str)


if __name__ == '__main__':
    ui_log_filepath = os.path.join(BASE_DIR, ALLOWED_LOG_FILES.get('ui_server', 'trading_bot_ui.log'))
    found_file_handler_for_ui = False
    for handler in logger_ui.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == os.path.abspath(ui_log_filepath):
            found_file_handler_for_ui = True
            break
    if not found_file_handler_for_ui and not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
        try:
            if not any(isinstance(h, logging.FileHandler) for h in logger_ui.handlers):
                fh_main = logging.FileHandler(ui_log_filepath)
                fh_main.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
                logger_ui.addHandler(fh_main)
                logger_ui.info(f"File handler for UI server log '{ui_log_filepath}' explicitly added in main.")
        except Exception as e_fh_main:
            logger_ui.error(f"Could not add file handler for UI server log in main: {e_fh_main}", exc_info=True)


    ui_port = int(os.environ.get('FLASK_UI_PORT', 5001))
    logger_ui.info(f"Flask UI server starting on http://0.0.0.0:{ui_port}")
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=ui_port, threads=10, _quiet=False)
    except ImportError:
        logger_ui.warning("Waitress not found, falling back to Flask development server.")
        app.run(debug=True, host='0.0.0.0', port=ui_port)
    except Exception as e_serve:
        logger_ui.critical(f"Failed to start server: {e_serve}", exc_info=True)
