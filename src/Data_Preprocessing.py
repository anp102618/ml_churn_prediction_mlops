# bank_churn_pipeline.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml
from Model_Utils.feature_nan_imputation import ImputerFactory
from Model_Utils.feature_outlier_handling import OutlierHandler
from Model_Utils.feature_scaling import ScalerFactory
from Common_Utils.sqlite_handler import SQLiteStrategy

# Setup logger and load config
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")

# Config paths & constants
sqlite_path: Path = Path(config["Data_Preprocessing"]["path"]["sqlite_path"])
table_name: str = config["Data_Preprocessing"]["path"]["sqlite_table"]
output_dir: Path = Path(config["Data_Preprocessing"]["path"]["processed_output_dir"])
output_dir.mkdir(exist_ok=True)
target_column: str = config["Data_Preprocessing"]["const"]["target_column"]
imputation_method: str = config["Data_Preprocessing"]["const"]["imputation_method"]
outlier_method: str = config["Data_Preprocessing"]["const"]["outlier_method"]
iqr_threshold: float = config["Data_Preprocessing"]["const"]["iqr_threshold"]
scaler_type: str = config["Data_Preprocessing"]["const"].get("scaler", "standard")

# Encoding Columns (embedded directly)
ordinal_cols: List[str] = ["Card_Category", "Education_Level", "Income_Category"]
ohe_cols: List[str] = ["Gender", "Marital_Status"]
categories: dict = {
    "Education_Level": ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"],
    "Income_Category": ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
    "Card_Category": ["Blue", "Silver", "Gold", "Platinum"]
}

class PreprocessingStrategy(ABC):
    """
    Abstract base class for preprocessing strategies used in a pipeline.

    Each subclass must implement the `apply` method to define how it transforms a DataFrame.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class RemoveDuplicatesStrategy(PreprocessingStrategy):
    """
    Removes duplicate rows from the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows.
    """
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

class DropHighMissingStrategy(PreprocessingStrategy):
    """
    Drops columns from the DataFrame that have more than 50% missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with high-missing columns dropped.
    """
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, thresh=int(0.5 * len(df)))

    
class ImputeMissingValuesStrategy(PreprocessingStrategy):
    """
        Applies imputation to both numeric and categorical columns.
        Numeric columns: uses self.imputer (assumed to be set externally).
        Categorical columns: uses most frequent value.

        Args:
            df (pd.DataFrame): Input DataFrame with missing values.

        Returns:
            pd.DataFrame: Imputed DataFrame.
    """
    def __init__(self, method: str):
        self.numeric_imputer = ImputerFactory.get_imputer(method)
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.numeric_cols = []
        self.cat_cols = []

    def fit_transform(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(df, pd.Series):
            df_2d = df.to_frame()
            result = self.cat_imputer.fit_transform(df_2d)
            return pd.Series(result.ravel(), name=df.name, index=df.index)

        # DataFrame
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.cat_cols = df.select_dtypes(include='object').columns.tolist()

        if self.numeric_cols:
            df[self.numeric_cols] = self.numeric_imputer.fit_transform(df[self.numeric_cols])
        if self.cat_cols:
            df[self.cat_cols] = self.cat_imputer.fit_transform(df[self.cat_cols])

        return df

    def transform(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(df, pd.Series):
            df_2d = df.to_frame()
            result = self.cat_imputer.transform(df_2d)
            return pd.Series(result.ravel(), name=df.name, index=df.index)

        if self.numeric_cols:
            df[self.numeric_cols] = self.numeric_imputer.transform(df[self.numeric_cols])
        if self.cat_cols:
            df[self.cat_cols] = self.cat_imputer.transform(df[self.cat_cols])

        return df

    def apply(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return self.fit_transform(df)

class OutlierTransformStrategy(PreprocessingStrategy):
    """
    Applies outlier transformation to numerical features only.

    Args:
        method (str): Transformation method (e.g., 'yeo', 'boxcox').
        iqr_threshold (float): IQR threshold to identify outliers.
    
    Returns the full DataFrame with transformed numeric columns and untouched categorical columns.
    """
    def __init__(self, method: str, iqr_threshold: float):
        self.method = method
        self.iqr_threshold = iqr_threshold
        self.handler = None
        self.numeric_cols = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        self.numeric_cols = df_copy.select_dtypes(include='number').columns
        df_numeric = df_copy[self.numeric_cols]

        self.handler = OutlierHandler(df_numeric, iqr_threshold=self.iqr_threshold)
        transformed = self.handler.transform(self.method)

        df_copy[self.numeric_cols] = transformed
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_numeric = df_copy[self.numeric_cols]

        transformed = self.handler.transform(self.method)
        df_copy[self.numeric_cols] = transformed
        return df_copy
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(df)


class PreprocessingContext:
    """
    Manages and executes a sequence of preprocessing steps.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Methods:
        add_step(step): Adds a preprocessing strategy to the pipeline.
        run(): Applies all steps in order.

    Returns:
        pd.DataFrame: The final transformed DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.steps: List[PreprocessingStrategy] = []

    def add_step(self, step: PreprocessingStrategy):
        self.steps.append(step)

    def run(self) -> pd.DataFrame:
        for step in self.steps:
            self.df = step.apply(self.df)
        return self.df


@track_performance
def execute_data_preprocessing() -> None:
    """
    Executes the full preprocessing pipeline:
    - Loads data from SQLite
    - Removes duplicates and high-missing columns
    - Converts target to binary
    - Splits into train/val/test sets
    - Imputes and transforms outliers on training set only
    - Encodes and scales all sets using a fitted pipeline
    - Saves transformed data and pipeline to disk

    Raises:
        CustomException: If preprocessing or saving fails.
    """
    try:
        logger.info(f"Starting Data_preprocessing ..")
        db_context = SQLiteStrategy(db_path=sqlite_path)
        query = f"SELECT * FROM {table_name}"
        df: pd.DataFrame = db_context.read_query(query=query)
        df = df.drop(columns=['CLIENTNUM'])
        print(df.columns)

        context = PreprocessingContext(df)
        context.add_step(RemoveDuplicatesStrategy())
        context.add_step(DropHighMissingStrategy())
        df_clean = context.run()
        print(df_clean.head())

        if df_clean.empty:
            raise CustomException("DataFrame is empty after cleaning. Cannot proceed.")

        # Step 1: Split before any transformation
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        imputer = ImputeMissingValuesStrategy(imputation_method)
        outlier = OutlierTransformStrategy(outlier_method, iqr_threshold)

        # 1. Impute X (train/val/test)
        X_train = imputer.apply(X_train)          # fit + transform
        X_val   = imputer.transform(X_val)        # transform only
        X_test  = imputer.transform(X_test)

        # 2. Outlier Transform X (only numerical cols)
        X_train = outlier.apply(X_train)          # fit + transform
        X_val   = outlier.transform(X_val)        # transform only
        X_test  = outlier.transform(X_test)

        # 3. Impute y (most_frequent), only if needed
        y_train = imputer.apply(y_train)          # fit + transform
        y_val   = imputer.transform(y_val)         #transform only 
        y_test  = imputer.transform(y_test)

        # 4. Convert target to binary (after imputation)
        y_train = y_train.map({"Existing Customer": 0, "Attrited Customer": 1})
        y_val   = y_val.map({"Existing Customer": 0, "Attrited Customer": 1})
        y_test  = y_test.map({"Existing Customer": 0, "Attrited Customer": 1})
        
        encoding_transformer = ColumnTransformer([
            ('ord', OrdinalEncoder(categories=[categories[col] for col in ordinal_cols]), ordinal_cols),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ohe_cols)
        ], remainder='passthrough')

        scaler = ScalerFactory.get_scaler(scaler_type)
        full_pipeline = Pipeline([
            ('encode', encoding_transformer),
            ('scale', scaler)
        ])

        X_train_final = full_pipeline.fit_transform(X_train)
        X_val_final = full_pipeline.transform(X_val)
        X_test_final = full_pipeline.transform(X_test)

        joblib.dump(full_pipeline, output_dir / "encoder_scaler.pkl")
        logger.info("Saved encoder_scaler.pkl successfully.")

        pd.DataFrame(X_train_final).to_csv(output_dir / "X_train.csv", index=False)
        pd.DataFrame(y_train).to_csv(output_dir / "y_train.csv", index=False)
        pd.DataFrame(X_val_final).to_csv(output_dir / "X_val.csv", index=False)
        pd.DataFrame(y_val).to_csv(output_dir / "y_val.csv", index=False)
        pd.DataFrame(X_test_final).to_csv(output_dir / "X_test.csv", index=False)
        pd.DataFrame(y_test).to_csv(output_dir / "y_test.csv", index=False)

        logger.info("Preprocessing and saving completed successfully.")

    except Exception as e:
        logger.exception("Preprocessing pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    execute_data_preprocessing()
