import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from Common_Utils import CustomException, track_performance, setup_logger, load_yaml

# Setup logger
logger = setup_logger(filename="logs")

# Load config and schema
config = load_yaml("Config_Yaml/config_path.yaml")
schema_config_path: Path = Path(config["DataValidation"]["path"]["schema_config_path"])
raw_df_path: Path = Path(config["DataValidation"]["path"]["raw_df_path"])

schema_config = load_yaml(schema_config_path)
schema = schema_config["columns"]

class DataValidator:
    """
    A class to validate a DataFrame against a defined schema and perform basic quality checks.

    Attributes:
        df (pd.DataFrame): The DataFrame to validate.
        schema (Dict[str, str]): Expected column names and dtypes.
        errors (Dict[str, List[str]]): Collected error messages for each type.
        count_summary (Dict[str, int]): Count of errors for each type.
    """

    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        """
        Initializes the DataValidator.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            schema (Dict[str, str]): Dictionary of expected column names and data types.
        """
        self.df = df.copy()
        self.schema = schema
        self.errors: Dict[str, List[str]] = {
            "missing_columns": [],
            "dtype_mismatches": [],
            "invalid_ranges": [],
            "missing_values": [],
            "duplicates": []
        }
        self.count_summary: Dict[str, int] = {
            "missing_columns": 0,
            "dtype_mismatches": 0,
            "invalid_ranges": 0,
            "missing_values": 0,
            "duplicates": 0
        }

    @track_performance
    def check_columns(self) -> None:
        """Checks if required columns are present in the DataFrame."""
        logger.info("Checking for missing columns...")
        required = set(self.schema.keys())
        present = set(self.df.columns)
        missing = list(required - present)
        if missing:
            self.errors["missing_columns"].extend(missing)
            self.count_summary["missing_columns"] += len(missing)
            logger.warning(f"Missing columns: {missing}")

    @track_performance
    def check_dtypes(self) -> None:
        """Validates the data types of columns based on schema."""
        logger.info("Checking data types...")
        for col, expected_type in self.schema.items():
            if col not in self.df.columns:
                continue

            if expected_type == "category":
                if not pd.api.types.is_categorical_dtype(self.df[col]):
                    if self.df[col].dtype != object:
                        msg = f"{col}: Expected category, got {self.df[col].dtype}"
                        self.errors["dtype_mismatches"].append(msg)
                        self.count_summary["dtype_mismatches"] += 1
                        logger.warning(msg)
            else:
                try:
                    coerced = pd.to_numeric(self.df[col], errors="coerce")
                    invalid_count = coerced.isnull().sum()
                    if invalid_count > 0:
                        msg = f"{col}: {invalid_count} non-numeric or NaN values"
                        self.errors["dtype_mismatches"].append(msg)
                        self.count_summary["dtype_mismatches"] += invalid_count
                        logger.warning(msg)
                except Exception:
                    msg = f"{col}: Failed to convert to {expected_type}"
                    self.errors["dtype_mismatches"].append(msg)
                    self.count_summary["dtype_mismatches"] += 1
                    logger.error(msg)

    @track_performance
    def check_missing_values(self, col_thresh: float = 0.4, row_thresh: float = 0.4) -> None:
        """
        Flags columns/rows with missing data above threshold.

        Args:
            col_thresh (float): Column missing value threshold. Default 0.4.
            row_thresh (float): Row missing value threshold. Default 0.4.
        """
        logger.info("Checking missing values...")
        if self.df.empty:
            msg = "DataFrame is empty."
            self.errors["missing_values"].append(msg)
            self.count_summary["missing_values"] += 1
            logger.error(msg)
            return

        col_missing_ratio = self.df.isnull().mean()
        cols_flagged = col_missing_ratio[col_missing_ratio >= col_thresh].index.tolist()
        if cols_flagged:
            msg = f"Columns with >= {int(col_thresh * 100)}% missing: {cols_flagged}"
            self.errors["missing_values"].append(msg)
            self.count_summary["missing_values"] += len(cols_flagged)
            logger.warning(msg)

        row_missing_ratio = self.df.isnull().mean(axis=1)
        rows_flagged = self.df.index[row_missing_ratio >= row_thresh].tolist()
        if rows_flagged:
            msg = f"Rows with >= {int(row_thresh * 100)}% missing: {rows_flagged[:5]}"
            self.errors["missing_values"].append(msg)
            self.count_summary["missing_values"] += len(rows_flagged)
            logger.warning(msg)

    @track_performance
    def check_value_ranges(self) -> None:
        """Performs validation on known numerical ranges like age, utilization, credit."""
        logger.info("Checking value ranges...")
        if 'Customer_Age' in self.df.columns:
            invalid = self.df[(self.df['Customer_Age'] < 18) | (self.df['Customer_Age'] > 100)]
            if not invalid.empty:
                msg = f"Customer_Age out of [18, 100]: rows {invalid.index.tolist()[:5]}"
                self.errors["invalid_ranges"].append(msg)
                self.count_summary["invalid_ranges"] += len(invalid)
                logger.warning(msg)

        if 'Avg_Utilization_Ratio' in self.df.columns:
            invalid = self.df[(self.df['Avg_Utilization_Ratio'] < 0) |
                              (self.df['Avg_Utilization_Ratio'] > 1)]
            if not invalid.empty:
                msg = f"Avg_Utilization_Ratio out of [0, 1]: rows {invalid.index.tolist()[:5]}"
                self.errors["invalid_ranges"].append(msg)
                self.count_summary["invalid_ranges"] += len(invalid)
                logger.warning(msg)

        if 'Credit_Limit' in self.df.columns:
            invalid = self.df[self.df['Credit_Limit'] < 0]
            if not invalid.empty:
                msg = f"Credit_Limit negative: rows {invalid.index.tolist()[:5]}"
                self.errors["invalid_ranges"].append(msg)
                self.count_summary["invalid_ranges"] += len(invalid)
                logger.warning(msg)

    @track_performance
    def check_duplicates(self) -> None:
        """Detects and logs duplicate rows."""
        logger.info("Checking for duplicates...")
        if self.df.empty:
            return
        dupes = self.df[self.df.duplicated()].index.tolist()
        if dupes:
            self.errors["duplicates"] = dupes[:10]
            self.count_summary["duplicates"] += len(dupes)
            logger.warning(f"Duplicate rows: {dupes[:5]}")

    @track_performance
    def run_all_checks(self) -> Tuple[bool, Dict[str, List[str]], Dict[str, int]]:
        """
        Runs all validation checks and returns result summary.

        Returns:
            Tuple:
                - bool: Validation pass/fail.
                - Dict[str, List[str]]: Detailed errors.
                - Dict[str, int]: Summary count of errors.
        """
        self.check_columns()
        self.check_dtypes()
        self.check_missing_values()
        self.check_value_ranges()
        self.check_duplicates()

        has_errors = any(self.errors[key] for key in self.errors)

        if has_errors:
            logger.warning("Validation failed with issues:")
            for k, v in self.errors.items():
                for issue in v:
                    logger.warning(f"{k.upper()}: {issue}")
            logger.info(f"Validation error counts: {self.count_summary}")
        else:
            logger.info("Data validation passed cleanly.")

        return not has_errors, self.errors, self.count_summary

@track_performance
def execute_data_validation() -> None:
    """
    Loads raw data, initializes validation, runs checks, and logs result.
    """
    try:
        logger.info("Starting data validation pipeline...")
        df = pd.read_csv(raw_df_path)
        validator = DataValidator(df=df, schema=schema)
        valid, errors, counts = validator.run_all_checks()

        if not valid:
            logger.warning("Validation failed.")
            logger.warning(f"Error summary: {counts}")
        else:
            logger.info("Validation passed successfully.")

    except CustomException as ce:
        logger.error(f"Validation failed with custom exception: {ce}")
    except Exception as ex:
        logger.exception(f"Unexpected error in validation: {ex}")

if __name__ == "__main__":
    execute_data_validation()


######sample output in case of errors####

"""
(False,
 {
   'missing_columns': ['Card_Category'],
   'dtype_mismatches': [
     'Credit_Limit: 2 non-numeric or NaN values',
     'Avg_Utilization_Ratio: 1 non-numeric or NaN values'
   ],
   'invalid_ranges': [
     'Customer_Age out of [18, 100]: rows [3, 7]',
     'Avg_Utilization_Ratio out of [0, 1]: rows [9]',
     'Credit_Limit negative: rows [15]'
   ],
   'missing_values': [
     "Columns with >= 40% missing: ['Education_Level']",
     "Rows with >= 40% missing: [2, 8]"
   ],
   'duplicates': [4, 12]
 },
 {
   'missing_columns': 1,
   'dtype_mismatches': 3,
   'invalid_ranges': 4,
   'missing_values': 3,
   'duplicates': 2
 })

"""