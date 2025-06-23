import os
import sys
import yaml
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from Common_Utils import CustomException, setup_logger, track_performance, load_yaml, convert

logger = setup_logger(filename="logs")

# === Load Configuration ===
main_config: Dict[str, Any] = load_yaml("Config_Yaml/config_path.yaml")
config = main_config["Model_Tune_Evaluate"]
config_path = config["path"]
config_const = config["const"]
model_config = load_yaml(config_path["classifiers_yaml"])

# Constants
allowed_models: List[str] = config_const["allowed_models"]
random_state: int = config_const["random_state"]
main_metric: str = config_const["main_metric"]
n_iter: int = config_const["bayes_search_iter"]

# Paths
X_train_path, y_train_path = Path(config_path["X_train_path"]), Path(config_path["y_train_path"])
X_val_path, y_val_path = Path(config_path["X_val_path"]), Path(config_path["y_val_path"])
X_test_path, y_test_path = Path(config_path["X_test_path"]), Path(config_path["y_test_path"])
tuned_model_yaml_path = Path(config_path["tuned_model_yaml"])


class ModelSelector:
    """
    A class to perform model selection, hyperparameter tuning, retraining, and evaluation.

    Attributes:
        results (List[Dict[str, Any]]): List to store results for each tuned model.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelSelector with an empty results list.
        """
        self.results: List[Dict[str, Any]] = []

    @track_performance
    def get_model_class(self, class_name: str):
        """
        Returns an uninitialized model class for a given class name.

        Args:
            class_name (str): The name of the model class (e.g., "LogisticRegression").

        Returns:
            type: The uninitialized class from sklearn or xgboost.

        Raises:
            ImportError: If class_name is not supported.
        """
        model_classes = {
            "XGBClassifier": XGBClassifier,
            "LogisticRegression": LogisticRegression,
            "RidgeClassifier": RidgeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "SVC": SVC,
            "MLPClassifier": MLPClassifier
        }
        if class_name not in model_classes:
            raise ImportError(f"Unsupported model: {class_name}")
        return model_classes[class_name]

    @track_performance
    def evaluate_score(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Optional[float]]:
        """
        Evaluates a trained model on the given dataset.

        Args:
            model (BaseEstimator): The trained model to evaluate.
            X (pd.DataFrame): Feature matrix.
            y (Union[pd.Series, np.ndarray]): True target labels.

        Returns:
            Dict[str, Optional[float]]: Dictionary containing evaluation metrics:
                accuracy, precision, recall, f1, and ROC AUC.
        """
        try:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

            return {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_proba) if y_proba is not None else None
            }
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            raise CustomException(e, sys)

    def run_process(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_yaml_path: Union[str, Path],
        allowed_models: Optional[List[str]] = None
    ) -> None:
        """
        Runs the full tuning, retraining, and evaluation process for each allowed model.

        Steps:
            - Hyperparameter tuning with BayesSearchCV.
            - Retraining on combined train + val.
            - Evaluation on train, val, test.
            - Results saved to YAML.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            output_yaml_path (Union[str, Path]): Path to save final results YAML.
            allowed_models (Optional[List[str]]): List of model keys to tune. Others will be skipped.

        Raises:
            CustomException: If model tuning or evaluation fails.
        """
        try:
            for model_key, cfg in model_config.items():
                if allowed_models and model_key not in allowed_models:
                    logger.info(f"Skipping model '{model_key}' (not allowed)")
                    continue

                logger.info(f"Tuning: {model_key}")
                ModelClass = self.get_model_class(cfg["model"])
                param_grid = cfg["params"]

                search = GridSearchCV(
                    estimator=ModelClass(),
                    param_grid=param_grid,
                    scoring=main_metric,
                    cv=3,
                    n_jobs=-1
                )

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Retrain on train+val
                X_combined = pd.concat([X_train, X_val])
                y_combined = pd.concat([y_train, y_val])
                final_model = clone(best_model).fit(X_combined, y_combined)

                scores = {
                    "train": self.evaluate_score(final_model, X_train, y_train),
                    "val": self.evaluate_score(final_model, X_val, y_val),
                    "test": self.evaluate_score(final_model, X_test, y_test)
                }

                self.results.append({
                    "model_key": model_key,
                    "model_name": type(final_model).__name__,
                    "best_params": search.best_params_,
                    "scores": scores
                })

            self.save_summary(output_yaml_path)

        except Exception as e:
            logger.exception(f"Model tuning failed: {e}")
            raise CustomException(e, sys)

    def save_summary(self, out_path: Union[str, Path]) -> None:
        """
        Saves all model results to a YAML file, sorted by test ROC AUC.

        Args:
            out_path (Union[str, Path]): File path to save the YAML.

        Raises:
            CustomException: If writing to YAML fails.
        """
        try:
            sorted_results = sorted(
                self.results,
                key=lambda x: x["scores"]["test"].get("roc_auc", 0),
                reverse=True
            )

            # Clean the result structure
            cleaned_results = convert(sorted_results)

            with open(out_path, "w") as f:
                yaml.safe_dump(cleaned_results, f, sort_keys=False)

            logger.info(f"Summary saved to: {out_path}")
        except IOError as e:
            logger.exception(f"Failed to save summary: {e}")
            raise CustomException(e, sys)


@track_performance
def execute_model_tune_evaluate() -> None:
    """
    Top-level function to execute model selection and evaluation workflow.

    Loads training/validation/test data from disk, performs tuning, and writes results.

    Raises:
        CustomException: If data loading or process execution fails.
    """
    try:
        logger.info("Starting model_tune_evaluate...")

        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).squeeze("columns")
        X_val = pd.read_csv(X_val_path)
        y_val = pd.read_csv(y_val_path).squeeze("columns")
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        selector = ModelSelector()
        selector.run_process(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            output_yaml_path=tuned_model_yaml_path,
            allowed_models=allowed_models
        )
        logger.info("model_tune_evaluate completed.")

    except Exception as e:
        logger.exception(f"model _tune_evaluate Execution failed: {e}")
        raise CustomException(e, sys)


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    execute_model_tune_evaluate()
