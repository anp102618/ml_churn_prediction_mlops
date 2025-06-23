import os
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import dagshub
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from Common_Utils import (
    setup_logger, track_performance, CustomException,
    load_yaml, delete_joblib_model, copy_yaml_file, convert
)

# ------------------ Global Setup ------------------

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
MLFLOW_TRACKING_URI = "https://dagshub.com/anp102618/ml_churn_prediction_mlops.mlflow"

logger = setup_logger("logs")
config: Dict[str, Any] = load_yaml("Config_Yaml/config_path.yaml")

# ------------------ Load Paths ------------------

config_path = config["Experiment_Tracking_Prediction"]["path"]

tuned_model_yaml: Path = Path(config_path["tuned_model_yaml"])
mlflow_details_yaml: Path = Path(config_path["mlflow_details_yaml"])
processed_data_dir: Path = Path(config_path["processed_data_dir"])
model_save_path: Path = Path(config_path["joblib_model_dir"]) / "model.joblib"



def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary with accuracy, precision, recall, F1 score, and ROC AUC.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }


def safe_log_metrics(metrics: Dict[str, float], prefix: str) -> None:
    """
    Log a dictionary of metrics to MLflow safely.

    Args:
        metrics (Dict[str, float]): Dictionary of metric values.
        prefix (str): Prefix for each metric key in MLflow.
    """
    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}_{k}", float(v))
        except Exception as e:
            logger.warning(f"Failed to log metric {prefix}_{k}: {e}")


@track_performance
def execute_mlflow_steps() -> None:
    """
    Execute the model training, evaluation, and MLflow tracking steps.

    This function:
    - Loads best model info from YAML
    - Loads train/val/test datasets
    - Trains and fine-tunes model
    - Evaluates on all splits
    - Logs to MLflow
    - Saves model and metadata
    """
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
        #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

        # Load best model config
        config_model_list = load_yaml(tuned_model_yaml)
        if not isinstance(config_model_list, list) or not config_model_list:
            raise CustomException("Expected a non-empty list of model configs in tuned_model.yaml")

        # Use the first model entry (xgboost) — assuming sorted by best
        best_model_info = config_model_list[0]

        model_name: str = best_model_info["model_name"]
        params: Dict[str, Any] = best_model_info["best_params"]

        model_name: str = best_model_info["model_name"]
        params: Dict[str, Any] = best_model_info["best_params"]

        logger.info(f"Using model: {model_name} with params: {params}")

        # Load dataset splits
        try:
            X_train = pd.read_csv(processed_data_dir / "X_train.csv")
            X_val = pd.read_csv(processed_data_dir / "X_val.csv")
            X_test = pd.read_csv(processed_data_dir / "X_test.csv")

            y_train = pd.read_csv(processed_data_dir / "y_train.csv").values.ravel()
            y_val = pd.read_csv(processed_data_dir / "y_val.csv").values.ravel()
            y_test = pd.read_csv(processed_data_dir / "y_test.csv").values.ravel()
        except Exception as e:
            raise CustomException(f"Failed to load processed CSVs: {e}")

        # Instantiate model class dynamically
        try:
            model_cls = eval(model_name)
            model = model_cls(**params)
        except NameError:
            raise CustomException(f"Model class '{model_name}' not found.")
        except Exception as e:
            raise CustomException(f"Error instantiating model: {e}")

        # 1. Train on (X_train, y_train)
        model.fit(X_train, y_train)
        train_metrics = evaluate_metrics(y_train, model.predict(X_train))

        # 2. Fine-tune on (X_train + X_val)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        model.fit(X_train_val, y_train_val)
        val_metrics = evaluate_metrics(y_val, model.predict(X_val))

        # 3. Final fit on (X_test, y_test)
        model.predict(X_test)
        test_metrics = evaluate_metrics(y_test, model.predict(X_test))

        # MLflow experiment setup
        experiment_name = "churn-classifier"
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")

            mlflow.log_param("model", model_name)
            mlflow.log_params(params)

            safe_log_metrics(train_metrics, "train")
            safe_log_metrics(val_metrics, "val")
            safe_log_metrics(test_metrics, "test")

            # Save the model
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            delete_joblib_model("Tuned_Model/model.joblib")
            joblib.dump(model, model_save_path)
            
            # Save run metadata
            model_uri = f"runs:/{run_id}/model"
            metadata: Dict[str, Any] = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
                "model": model_name,
                "best_params": params,
                "scores": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics
                },
                "mlflow_run": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "model_uri": model_uri,
                    "artifact_uri": run.info.artifact_uri,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "tracking_uri": MLFLOW_TRACKING_URI
                },
                "saved_model_path": str(model_save_path.resolve())
            }
            
            metadata = convert(metadata)
            with open(mlflow_details_yaml, "w") as f:
                yaml.dump(metadata, f)


    except CustomException as ce:
        logger.error(f"[CustomException] {ce}")
    except Exception as e:
        logger.exception(f"[UnhandledException] {e}")


# ------------------ Entry Point ------------------

if __name__ == "__main__":
    execute_mlflow_steps()
