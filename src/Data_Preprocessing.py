# bank_churn_pipeline.py
import os
import sys
import json
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from Common_Utils import setup_logger, track_performance, CustomException, load_yaml
from Model_Utils.feature_nan_imputation import DataImputer
from Model_Utils.feature_outlier_handling import OutlierHandler
from Model_Utils.feature_scaling import ScalerFactory
from Common_Utils.sqlite_handler import SQLiteStrategy

# -------------------- Logger & Config --------------------
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")

# -------------------- Paths --------------------
sqlite_path: Path = Path(config["Data_Preprocessing"]["path"]["sqlite_path"])
table_name: str = config["Data_Preprocessing"]["path"]["sqlite_table"]
output_dir: Path = Path(config["Data_Preprocessing"]["path"]["processed_output_dir"])
tuned_model_dir: Path = Path(config["Data_Preprocessing"]["path"]["tuned_output_dir"])
output_dir.mkdir(exist_ok=True, parents=True)
tuned_model_dir.mkdir(exist_ok=True, parents=True)

# -------------------- Constants --------------------
target_column: str = config["Data_Preprocessing"]["const"]["target_column"]
imputation_method: str = config["Data_Preprocessing"]["const"]["imputation_method"]
outlier_handle_method: str = config["Data_Preprocessing"]["const"]["outlier_method"]
iqr_threshold: float = config["Data_Preprocessing"]["const"]["iqr_threshold"]
scaler_type: str = config["Data_Preprocessing"]["const"].get("scaler", "standard")

ordinal_cols: List[str] = ["Card_Category", "Education_Level", "Income_Category"]
ohe_cols: List[str] = ["Gender", "Marital_Status"]
categories: dict = {
    "Education_Level": ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"],
    "Income_Category": ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
    "Card_Category": ["Blue", "Silver", "Gold", "Platinum"]
}

# -------------------- Strategy Pattern --------------------
class PreprocessingStrategy(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class RemoveDuplicatesStrategy(PreprocessingStrategy):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

class DropHighMissingStrategy(PreprocessingStrategy):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, thresh=int(0.5 * len(df)))

class PreprocessingContext:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.steps: List[PreprocessingStrategy] = []

    def add_step(self, step: PreprocessingStrategy):
        self.steps.append(step)

    def run(self) -> pd.DataFrame:
        for step in self.steps:
            self.df = step.apply(self.df)
        return self.df

# -------------------- Main Pipeline Execution --------------------
@track_performance
def execute_data_preprocessing() -> None:
    try:
        logger.info("Starting Data Preprocessing...")

        # Load Data from SQLite
        db_context = SQLiteStrategy(db_path=sqlite_path)
        df = db_context.read_query(query=f"SELECT * FROM {table_name}")
        df = df.drop(columns=['CLIENTNUM'])

        # Apply Cleaning Strategies
        context = PreprocessingContext(df)
        context.add_step(RemoveDuplicatesStrategy())
        context.add_step(DropHighMissingStrategy())
        df_clean = context.run()

        if df_clean.empty:
            raise CustomException("DataFrame is empty after cleaning. Cannot proceed.")

        # Train-Val-Test Split
        X = df_clean.drop(columns=[target_column])
        y = df_clean[[target_column]]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # Imputation & Outlier Transformation
        x_imp, y_imp = DataImputer(numeric_strategy="mean"), DataImputer()
        X_train_impute = x_imp.fit_transform(X_train)
        y_train_impute = y_imp.fit_transform(y_train)
        X_val_impute = x_imp.transform(X_val)
        y_val_impute = y_imp.transform(y_val)
        X_test_impute = x_imp.transform(X_test)
        y_test_impute = y_imp.transform(y_test)

        outlier_handler = OutlierHandler(iqr_threshold=1.5)
        X_train = outlier_handler.fit_transform(X_train_impute,outlier_handle_method)
        X_val = outlier_handler.transform(X_val_impute, outlier_handle_method)
        X_test = outlier_handler.transform(X_test_impute, outlier_handle_method)


        # Convert target to binary
        y_train = y_train_impute.iloc[:, 0].map({"Existing Customer": 0, "Attrited Customer": 1}).astype(int)
        y_val = y_val_impute.iloc[:, 0].map({"Existing Customer": 0, "Attrited Customer": 1}).astype(int)
        y_test = y_test_impute.iloc[:, 0].map({"Existing Customer": 0, "Attrited Customer": 1}).astype(int)

        # save processed data 
        raw_data_dir = Path("Data/raw_data/preprocessed_data")
        raw_data_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        # Save feature sets
        X_train.to_csv(raw_data_dir / "X_train.csv", index=False)
        X_val.to_csv(raw_data_dir / "X_val.csv", index=False)
        X_test.to_csv(raw_data_dir / "X_test.csv", index=False)

        # Save target sets
        y_train.to_csv(raw_data_dir / "y_train.csv", index=False)
        y_val.to_csv(raw_data_dir / "y_val.csv", index=False)
        y_test.to_csv(raw_data_dir / "y_test.csv", index=False)

        # Encoding + Scaling Pipeline
        encoding_transformer = ColumnTransformer([
            ('ord', OrdinalEncoder(categories=[categories[col] for col in ordinal_cols]), ordinal_cols),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ohe_cols)
        ], remainder='passthrough')

        scaler = ScalerFactory.get_scaler(scaler_type)

        full_pipeline = Pipeline([
            ('encode', encoding_transformer),
            ('scale', scaler)
        ])

        # Fit-transform and get feature names
        X_train_final = full_pipeline.fit_transform(X_train)
        X_val_final = full_pipeline.transform(X_val)
        X_test_final = full_pipeline.transform(X_test)

        feature_names = full_pipeline.get_feature_names_out()

        # Save transformed data
        pd.DataFrame(X_train_final, columns=feature_names).to_csv(output_dir / "X_train.csv", index=False)
        pd.DataFrame(X_val_final, columns=feature_names).to_csv(output_dir / "X_val.csv", index=False)
        pd.DataFrame(X_test_final, columns=feature_names).to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_val.to_csv(output_dir / "y_val.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)

        # Save pipeline & feature names
        joblib.dump(x_imp, tuned_model_dir / "x_imputer.joblib")
        joblib.dump(outlier_handler, tuned_model_dir / "outlier_handler.joblib")
        joblib.dump(full_pipeline, tuned_model_dir / "encoder_scaler.joblib")
        with open(tuned_model_dir / "feature_names.json", "w") as f:
            json.dump(feature_names.tolist(), f)

        logger.info("Preprocessing and saving completed successfully.")

    except Exception as e:
        logger.exception("Preprocessing pipeline failed.")
        raise CustomException(e, sys)

# -------------------- Entry --------------------
if __name__ == "__main__":
    execute_data_preprocessing()
