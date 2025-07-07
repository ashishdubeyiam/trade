## Explanation of `fraud_detection_project/src/data_preprocessing.py`

The `data_preprocessing.py` script is a cornerstone of the `fraud_detection_project`'s machine learning pipeline. It encapsulates all essential steps required to transform raw input data into a clean, structured, and numerical format suitable for training and evaluating machine learning models.

**1. Overall Purpose:**

This script is responsible for:
*   **Loading** the raw dataset from a CSV file.
*   Performing initial **data inspection** to understand its structure, content, and quality.
*   **Cleaning** the data by handling missing values (if any, though the guide implies this is minimal) and dropping irrelevant columns.
*   Addressing **class imbalance** in the target variable (fraud vs. non-fraud) using undersampling.
*   Conducting **feature engineering**, such as deriving new features from timestamps.
*   Performing **feature transformation**, including encoding categorical features and scaling numerical features.
*   **Splitting** the processed data into training and testing sets.
*   **Saving** these processed datasets for later use in model training and evaluation.

**2. Key Libraries Used:**

*   **`pandas`:** The primary library for all DataFrame manipulations, including data loading, column operations, filtering, and saving CSV files.
*   **`numpy`:** Used for numerical operations, often implicitly through pandas, but can be used directly for array manipulations if needed.
*   **`sklearn.utils.resample`:** Specifically used for the random undersampling of the majority class in the `handle_class_imbalance` function.
*   **`sklearn.model_selection.train_test_split`:** Used to split the data into training and testing sets.
*   **`sklearn.preprocessing.StandardScaler`:** Used to standardize numerical features by removing the mean and scaling to unit variance.
*   **`sklearn.preprocessing.OneHotEncoder`:** Used to convert categorical string features into a numerical format (one-hot vectors).
*   **`sklearn.compose.ColumnTransformer`:** A powerful tool used to apply different transformations to different columns of a dataset in a consistent manner.

**3. Core Functions and their Logic:**

*   **`load_data(file_path: str) -> pd.DataFrame`:**
    *   Takes a `file_path` string (typically provided from `config.py`) as input.
    *   Uses `pd.read_csv(file_path)` to load the data into a pandas DataFrame.
    *   Includes basic error handling, such as a `try-except` block to catch `FileNotFoundError` and print an informative message if the specified file does not exist, then exits the program.
    *   Returns the loaded DataFrame.

*   **`inspect_data(df: pd.DataFrame)`:**
    *   Takes a DataFrame `df` as input.
    *   Prints various pieces of information to the console for initial understanding and debugging:
        *   `df.shape`: Dimensions of the DataFrame (rows, columns).
        *   `df.head()`: First few rows.
        *   `df.info()`: Summary of columns, data types, and non-null counts.
        *   `df.describe()`: Descriptive statistics for numerical columns.
        *   `df.isnull().sum()`: Count of missing values per column.
        *   `df[config.TARGET_COLUMN].value_counts(normalize=True)`: Distribution of the target variable (e.g., percentage of fraud vs. non-fraud).
        *   Information about timestamp columns if `config.TIMESTAMP_COLUMN` is present (min, max, data type).

*   **`handle_class_imbalance(df: pd.DataFrame, target_column: str) -> pd.DataFrame`:**
    *   Aims to balance the dataset due to the typical rarity of fraud instances.
    *   **Method Used (Undersampling):**
        1.  Separates the DataFrame into two DataFrames: one for the majority class and one for the minority class based on the `target_column`.
        2.  Uses `resample` from `sklearn.utils` to downsample the majority class DataFrame.
            *   `n_samples` is set to the number of samples in the minority class, ensuring both classes have equal representation after resampling.
            *   `replace=False` means sampling without replacement.
            *   `random_state=config.RANDOM_SEED` ensures reproducibility.
        3.  Concatenates the undersampled majority class DataFrame with the original minority class DataFrame using `pd.concat`.
        4.  Shuffles the resulting combined DataFrame using `df_balanced.sample(frac=1, random_state=config.RANDOM_SEED)` to ensure the data is not ordered by class.
    *   Returns the balanced (undersampled and shuffled) DataFrame.

*   **`preprocess_features(df: pd.DataFrame, experiment_type: str = 'all_features') -> tuple[pd.DataFrame, pd.Series, object]`:**
    *   This is a central function responsible for most of the feature transformation logic.
    *   **Feature (X) and Target (y) Separation:**
        *   Separates the input DataFrame `df` into features `X` (all columns except the `config.TARGET_COLUMN`) and the target variable `y` (the `config.TARGET_COLUMN` Series).
    *   **Timestamp Feature Engineering:**
        *   Checks if `config.TIMESTAMP_COLUMN` exists in `X`.
        *   If it exists, converts the column to datetime objects using `pd.to_datetime`.
        *   Extracts new time-based features: 'Hour_of_Day' (`.dt.hour`), 'Day_of_Week' (`.dt.dayofweek`), and 'Month_of_Year' (`.dt.month`).
        *   Drops the original `config.TIMESTAMP_COLUMN` from `X`.
    *   **Dropping Specified Columns:**
        *   Removes columns listed in `config.COLUMNS_TO_DROP` from `X` if they exist.
    *   **Experiment-Specific Feature Dropping:**
        *   If `experiment_type == 'selected_features'`, it further drops columns listed in `config.FEATURES_TO_DROP_EXPERIMENT2` from `X` if they exist.
    *   **Automatic Identification of Numerical and Categorical Features:**
        *   After initial transformations and drops, it identifies numerical features (`X.select_dtypes(include=np.number).columns.tolist()`) and uses the predefined list `config.CATEGORICAL_FEATURES` (filtering for those still present in `X`).
    *   **`ColumnTransformer` Setup:**
        *   Initializes a `ColumnTransformer` to apply different transformations to different feature types.
        *   **Categorical Transformer:** Applies `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to the identified categorical features. `handle_unknown='ignore'` ensures that if new categories appear in test data (or during prediction), they don't cause errors but are encoded as all zeros. `sparse_output=False` provides a dense numpy array.
        *   **Numerical Transformer:** Applies `StandardScaler()` to the identified numerical features.
        *   `remainder='passthrough'` ensures that any columns not explicitly selected as numerical or categorical (if any) are passed through without transformation (though typically all relevant features are covered).
    *   **Applying Transformations (`fit_transform`):**
        *   The `ColumnTransformer` is fitted and then transforms the feature set `X` using `preprocessor.fit_transform(X)`. This learns the parameters (e.g., means/stds for scaling, unique categories for OHE) from `X` and applies the transformations.
        *   The output is a NumPy array.
    *   **Reconstructing Feature Names (Attempt and Caveats):**
        *   The script attempts to reconstruct meaningful feature names after `OneHotEncoder` creates new columns (e.g., `Category_A`, `Category_B`). This is done using `preprocessor.get_feature_names_out()`.
        *   The transformed NumPy array is converted back into a pandas DataFrame with these new feature names.
        *   The guide notes potential issues or warnings if this reconstruction is complex, but it's a common practice for better interpretability.
    *   Returns the processed `X` DataFrame, the `y` Series, and the fitted `preprocessor` object (which is important for transforming test data consistently later, though not explicitly shown being passed out in the guide's main workflow description, it's crucial for proper ML practice).

*   **`split_data(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_SEED) -> tuple[...]`:**
    *   Takes the processed features `X` and target `y` as input.
    *   Uses `train_test_split` from `sklearn.model_selection` to divide the data.
    *   `test_size` (from `config.py`) determines the proportion of data for the test set.
    *   `random_state=random_state` (from `config.py`) ensures the split is the same every time the script runs, for reproducibility.
    *   `stratify=y` is used to ensure that the proportion of the target classes (fraud/non-fraud) is approximately the same in both the training and testing sets. This is important for imbalanced datasets.
    *   Returns `X_train, X_test, y_train, y_test`.

*   **`save_processed_data(X_train, X_test, y_train, y_test, train_file_path, test_file_path)`:**
    *   Takes the split data components and file paths (from `config.py`) as input.
    *   Combines features and the target variable back into DataFrames for saving: `train_df = pd.concat([X_train, y_train], axis=1)` and similarly for the test set.
    *   Saves these DataFrames to CSV files using `df.to_csv(file_path, index=False)` at the specified `train_file_path` and `test_file_path`. `index=False` prevents writing the DataFrame index to the CSV.

**4. Workflow (`if __name__ == '__main__':` block):**

The script includes a main execution block that allows it to be run standalone (e.g., `python src/data_preprocessing.py`). This block typically:
1.  Loads data using `load_data(config.RAW_DATA_FILE)`.
2.  Inspects the loaded data using `inspect_data()`.
3.  Handles class imbalance using `handle_class_imbalance()`.
4.  Preprocesses features using `preprocess_features()` (often on the balanced DataFrame). The fitted preprocessor from this step is crucial but its handling for transforming the test set separately is a detail often abstracted in such guides but vital in practice. (Ideally, `preprocess_features` is fit on training data only, then used to transform both train and test data).
5.  Splits the data using `split_data()`.
6.  Saves the processed train and test sets using `save_processed_data()`.
This standalone execution is useful for debugging the preprocessing steps or for generating processed data independently of the full `main.py` pipeline.

**5. Data Flow Summary:**

1.  Raw CSV data is loaded (`load_data`).
2.  Initial inspection is performed (`inspect_data`).
3.  The dataset is balanced via undersampling (`handle_class_imbalance`).
4.  Features are engineered (timestamps) and transformed (OHE for categoricals, scaling for numericals) using `preprocess_features`.
5.  The fully processed dataset is split into training and testing sets (`split_data`).
6.  These final train and test sets are saved as new CSV files (`save_processed_data`).

**6. Dependencies and Interactions:**

*   **`config.py`:** `data_preprocessing.py` heavily relies on `config.py` for numerous settings:
    *   Input data paths (`RAW_DATA_FILE`).
    *   Output data paths (`PROCESSED_DATA_DIR`, `TRAIN_DATA_FILE`, `TEST_DATA_FILE`).
    *   Column names (`TARGET_COLUMN`, `TIMESTAMP_COLUMN`, `COLUMNS_TO_DROP`, `CATEGORICAL_FEATURES`).
    *   Parameters for splitting and reproducibility (`TEST_SIZE`, `RANDOM_SEED`).
*   **Downstream Scripts:**
    *   The primary outputs – the processed `X_train, y_train, X_test, y_test` data (either as in-memory DataFrames/Series if part of a larger script call, or more commonly, the saved `train.csv` and `test.csv` files) – are consumed by:
        *   `train_models.py`: Uses `train.csv` to train the various machine learning models.
        *   `evaluate_models.py`: Uses `test.csv` (and potentially `train.csv` for some aspects like loading the preprocessor if not passed directly) to evaluate the trained models.

In essence, `data_preprocessing.py` is a critical preparatory stage that ensures the data fed into the machine learning models is of high quality, correctly formatted, and appropriately partitioned for robust training and evaluation.
