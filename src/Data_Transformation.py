import pandas as pd
import joblib
import yaml
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.base import TransformerMixin
from Model_Utils.feature_sampling import SamplingFactory
from Model_Utils.feature_selection_extraction import FeatureFactory
from Common_Utils import CustomException, setup_logger, track_performance, load_yaml

logger = setup_logger(filename="logs")

# Load configuration
config: Dict[str, Any] = load_yaml("Config_Yaml/config_path.yaml")
config_path: Dict[str, str] = config["DataTransformation"]["path"]
config_const: Dict[str, Any] = config["DataTransformation"]["const"]

# Resolve paths using Pathlib
X_train_path: Path = Path(config_path["X_train_path"])
X_val_path: Path = Path(config_path["X_val_path"])
X_test_path: Path = Path(config_path["X_test_path"])
y_train_path: Path = Path(config_path["y_train_path"])
y_val_path: Path = Path(config_path["y_val_path"])
y_test_path: Path = Path(config_path["y_test_path"])
processed_data_folder: Path = Path(config_path["processed_data_folder"])

enc_scaler_path: Path = Path(config_path["enc_scaler_path"])
selector_path: Path = Path(config_path["selector_path"])
extractor_path: Path = Path(config_path["extractor_path"])

sampler_method: str = config_const["sampler_method"]
selector_method: str = config_const["selector_method"]
extractor_method: str = config_const["extractor_method"]
k: int = int(config_const["k"])
n_components: int = int(config_const["n_components"])


class DataTransformation:
    """
    A class to handle the complete data preprocessing pipeline for a machine learning workflow.

    This includes loading data, encoding/scaling, sampling, feature selection, and feature extraction,
    all based on paths and parameters defined in a YAML configuration file.
    """
    def __init__(self):
        pass
    
    @track_performance
    def load_data(self) -> Tuple[pd.DataFrame, ...]:
        """
        Loads training, validation, and testing datasets from CSV files.

        Returns:
            Tuple containing:
                - X_train, X_val, X_test (pd.DataFrame): Feature sets
                - y_train, y_val, y_test (pd.DataFrame): Target labels

        Raises:
            CustomException: If any CSV fails to load.
        """
        try:
            X_train = pd.read_csv(X_train_path)
            X_val   = pd.read_csv(X_val_path)
            X_test  = pd.read_csv(X_test_path)
            y_train = pd.read_csv(y_train_path)
            y_val   = pd.read_csv(y_val_path)
            y_test  = pd.read_csv(y_test_path)
            logger.info("Data loaded successfully.")
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logger.error(f"Error loading data from CSVs")
            raise CustomException(e,sys)

    @track_performance
    def encode_and_scale(self,
                         X_train: pd.DataFrame,
                         X_val: pd.DataFrame,
                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Encodes and scales datasets using a pre-trained encoder/scaler.

        Args:
            X_train, X_val, X_test (pd.DataFrame): Raw feature sets.

        Returns:
            Tuple containing encoded and scaled feature sets as DataFrames.

        Raises:
            CustomException: If encoding/scaling fails.
        """
        try:
            encoder_scaler: TransformerMixin = joblib.load(enc_scaler_path)
            feature_names = encoder_scaler.get_feature_names_out()
            
            X_train_enc = encoder_scaler.transform(X_train)
            X_val_enc   = encoder_scaler.transform(X_val)
            X_test_enc  = encoder_scaler.transform(X_test)

            logger.info("Encoding and scaling completed.")

            return (
                pd.DataFrame(X_train_enc, columns=feature_names, index=X_train.index),
                pd.DataFrame(X_val_enc, columns=feature_names, index=X_val.index),
                pd.DataFrame(X_test_enc, columns=feature_names, index=X_test.index)
            )
        except Exception as e:
            logger.error(f"Error during encoding and scaling")
            raise CustomException(e, sys)

    @track_performance
    def sample_data(self,
                    X: pd.DataFrame,
                    y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resamples training data using the configured sampler method (e.g., SMOTE).

        Args:
            X (pd.DataFrame): Feature set
            y (pd.DataFrame): Labels

        Returns:
            Resampled (X, y)

        Raises:
            CustomException: On failure during resampling.
        """
        try:
            if X.isnull().any().any() or y.isnull().any().any():
                raise ValueError("Input data contains missing values before sampling.")

            sampler = SamplingFactory.get_sampler(sampler_method)
            X_res, y_res = sampler.fit_resample(X, y)

            logger.info("Sampling completed successfully.")
            return X_res, y_res
        
        except Exception as e:
            logger(f"Error during sampling")
            raise CustomException(e, sys)

    @track_performance
    def select_features(self,
                        X_train: pd.DataFrame,
                        y_train: pd.DataFrame,
                        X_val: pd.DataFrame,
                        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Applies feature selection to reduce dimensionality using SelectKBest or similar methods.

        Args:
            X_train, y_train (pd.DataFrame): Training data and labels
            X_val, X_test (pd.DataFrame): Validation and test sets

        Returns:
            Tuple containing transformed feature sets.

        Raises:
            CustomException: If selection fails.
        """
        try:
            selector = FeatureFactory.get_processor("selection", selector_method, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train.values.ravel())
            selected_cols = X_train.columns[selector.get_support()]

            X_val_sel = selector.transform(X_val)
            X_test_sel = selector.transform(X_test)

            joblib.dump(selector, selector_path)

            logger.info(f"Feature selection applied. Selected columns: {len(selected_cols)}")
            return (
                pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train.index),
                pd.DataFrame(X_val_sel, columns=selected_cols, index=X_val.index),
                pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test.index)
            )
        except Exception as e:
            logger.error(f"Error during feature selection")
            raise CustomException(e, sys)

    @track_performance
    def extract_features(self,
                         X_train: pd.DataFrame,
                         X_val: pd.DataFrame,
                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Applies dimensionality reduction using KernelPCA or other techniques.

        Args:
            X_train, X_val, X_test (pd.DataFrame): Feature-selected data.

        Returns:
            Tuple of reduced feature sets.

        Raises:
            CustomException: If extraction fails.
        """
        try:
            extractor = FeatureFactory.get_processor("extraction", extractor_method, n_components=n_components)
            X_train_ext = extractor.fit_transform(X_train)
            X_val_ext = extractor.transform(X_val)
            X_test_ext = extractor.transform(X_test)

            joblib.dump(extractor, extractor_path)

            logger.info("Feature extraction completed successfully.")
            return X_train_ext, X_val_ext, X_test_ext
        except Exception as e:
            logger.error(f"Error during feature extraction")
            raise CustomException(e, sys)

@track_performance
def execute_data_transformation() -> Tuple[pd.DataFrame, ...]:
    """
    Executes the full preprocessing pipeline: load, encode, sample, select, extract.

    Returns:
        Tuple containing final processed X_train, X_val, X_test, y_train, y_val, y_test.

    Raises:
        CustomException: If any step fails.
    """
    try:
        logger.info("Data Transformation execution started.")
        context = DataTransformation()
        X_train, X_val, X_test, y_train, y_val, y_test = context.load_data()
        X_train_enc, X_val_enc, X_test_enc = context.encode_and_scale(X_train, X_val, X_test)
        X_train_samp, y_train_samp = context.sample_data(X_train_enc, y_train)
        X_train_sel, X_val_sel, X_test_sel = context.select_features(X_train_samp, y_train_samp, X_val_enc, X_test_enc)
        X_train_final, X_val_final, X_test_final = context.extract_features(X_train_sel, X_val_sel, X_test_sel)

        extracted_feature_names = [f"{extractor_method}_feat_{i}" for i in range(X_train_final.shape[1])]

        # Convert NumPy arrays to DataFrames with column names and original index
        X_train_final = pd.DataFrame(X_train_final, columns=extracted_feature_names, index=X_train_sel.index)
        X_val_final   = pd.DataFrame(X_val_final,   columns=extracted_feature_names, index=X_val_sel.index)
        X_test_final  = pd.DataFrame(X_test_final,  columns=extracted_feature_names, index=X_test_sel.index)

                # Save final outputs
        X_train_final.to_csv(processed_data_folder / "X_train.csv", index=False)
        X_val_final.to_csv(processed_data_folder / "X_val.csv", index=False)
        X_test_final.to_csv(processed_data_folder / "X_test.csv", index=False)
        y_train_samp.to_csv(processed_data_folder / "y_train.csv", index=False)
        y_val.to_csv(processed_data_folder / "y_val.csv", index=False)
        y_test.to_csv(processed_data_folder / "y_test.csv", index=False)

        logger.info("Processed datasets saved successfully.")
        logger.info("Data Transformation execution completed.")

    except Exception as e:
        logger.error(f"Data Transformation execution failed")
        raise CustomException(e, sys)

# -------------------- Entry --------------------
if __name__ == "__main__":
    execute_data_transformation()