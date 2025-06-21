import yaml
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Any

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from Model_Utils.classification_models import ModelFactory
from Model_Utils.feature_sampling import SamplingFactory
from Model_Utils.feature_selection_extraction import FeatureFactory
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml

# ------------------ Logger Setup ------------------ #
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")

# ----------------- Config Paths ------------------ #
processed_data_dir: Path = Path(config["Model_Tune_Evaluate"]["path"]["processed_data_dir"])
tuned_model_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["tuned_model_yaml"])
classifiers_yaml: Path = Path(config["Model_Tune_Evaluate"]["path"]["classifiers_yaml"])
selector_path: Path = Path(config["Model_Tune_Evaluate"]["path"]["selector_path"])
extractor_path: Path = Path(config["Model_Tune_Evaluate"]["path"]["extractor_path"])

# ---------------- Config Constants ---------------- #
allowed_models: List[str] = config["Model_Tune_Evaluate"]["const"]["allowed_models"]
sampling_method: str = config["Model_Tune_Evaluate"]["const"]["sampling"]
selection_method: str = config["Model_Tune_Evaluate"]["const"]["selection"]
extraction_method: str = config["Model_Tune_Evaluate"]["const"]["extraction"]
k: int = config["Model_Tune_Evaluate"]["const"]["k"]
n_components: int = config["Model_Tune_Evaluate"]["const"]["n_components"]


def get_search_space(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    search_space = {}
    for key, values in param_grid.items():
        first = values[0]
        if isinstance(first, float):
            search_space[key] = Real(min(values), max(values))
        elif isinstance(first, int):
            search_space[key] = Integer(min(values), max(values))
        else:
            search_space[key] = Categorical(values)
    return search_space


class ClassificationModelTuner:
    def __init__(self, config_path: Path, allowed_models: Optional[List[str]] = None):
        self.factory = ModelFactory(config_path)
        self.config_path = config_path
        self.allowed_models = allowed_models

        self.X_train = pd.read_csv(processed_data_dir / "X_train.csv")
        self.y_train = pd.read_csv(processed_data_dir / "y_train.csv").iloc[:, 0]
        self.X_val = pd.read_csv(processed_data_dir / "X_val.csv")
        self.y_val = pd.read_csv(processed_data_dir / "y_val.csv").iloc[:, 0]
        self.X_test = pd.read_csv(processed_data_dir / "X_test.csv")
        self.y_test = pd.read_csv(processed_data_dir / "y_test.csv").iloc[:, 0]
        self.results = []

    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else preds
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "roc_auc": roc_auc_score(y, proba)
        }

    @track_performance
    def run_grid_search(self) -> None:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            for model_name, model_info in config.items():
                if self.allowed_models and model_name not in self.allowed_models:
                    logger.info(f"Skipping {model_name}...")
                    continue

                logger.info(f"Running Bayesian search for: {model_name}")
                search_space = get_search_space(model_info['params'])
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                logger.info(f"Starting Feature Engineering..")
                sampler = SamplingFactory.get_sampler(sampling_method)
                selector = FeatureFactory.get_processor(kind='selection', method=selection_method, k=k)
                extractor = FeatureFactory.get_processor(kind='extraction', method=extraction_method, n_components=n_components)
                logger.info(f"Feature selector and extractor initialized..")

                X_sampled, y_sampled = sampler.sample(self.X_train, self.y_train)
                X_selected = selector.process(X_sampled, y_sampled)
                selected_columns = selector.get_selected_columns()
                X_extracted = extractor.process(X_selected, y_sampled)
                X_extracted_df = pd.DataFrame(X_extracted)

                logger.info(f"Feature Engineering completed..")

                logger.info(f"Starting model tuning..")
                model = self.factory.get_model(model_name)
                opt = BayesSearchCV(
                    estimator=model,
                    search_spaces=search_space,
                    n_iter=20,
                    cv=kf,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
                opt.fit(X_extracted, y_sampled)
                best_model = opt.best_estimator_
                best_params = opt.best_params_

                logger.info(f"initial model tuning finished..")

                logger.info(f"Starting model fine tuning on (train+val)..")
                X_val_resampled = self.X_val.copy()  
                X_val_selected = selector.transform(X_val_resampled[selected_columns])
                X_val_extracted = extractor.transform(X_val_selected)

                X_finetune = np.concatenate([X_extracted_df, X_val_extracted], axis=0)
                y_finetune = np.concatenate([y_sampled, self.y_val], axis=0)
                best_model.fit(X_finetune, y_finetune)
                logger.info(f" model fine tuning on (train+val) completed..")

                selector_path.parent.mkdir(parents=True, exist_ok=True)
                extractor_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(selector, selector_path)
                joblib.dump(extractor, extractor_path)
                logger.info(f"saved processor artifacts successfully..")

                X_test_selected = selector.transform(self.X_test[selected_columns])
                X_test_extracted = extractor.transform(X_test_selected)

                test_scores = self.evaluate_model(best_model, X_test_extracted, self.y_test)

                logger.info(f"Best ROC AUC for {model_name}: {round(test_scores['roc_auc'], 4)}")
                logger.info(f"Best Params: {best_params}")

                self.results.append({
                    "model": model_name,
                    "params": best_params,
                    "roc_auc": round(test_scores['roc_auc'], 4),
                    "scores": {k: round(v, 4) for k, v in test_scores.items()}
                })
                logger.info(f"final evaluation performed and score metrics saved successfully ..")

        except Exception as e:
            logger.exception("Bayesian search failed.")
            raise CustomException(e, sys)

    def save_results(self, output_path: Path) -> None:
        try:
            sorted_results = sorted(self.results, key=lambda x: x['roc_auc'], reverse=True)
            with open(output_path, 'w') as f:
                yaml.dump(sorted_results, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Best model: {sorted_results[0]['model']} with ROC AUC: {sorted_results[0]['roc_auc']}")
        except Exception as e:
            logger.exception("Failed to save results.")
            raise CustomException(e, sys)


@track_performance
def execute_model_tune_evaluate() -> None:
    try:
        logger.info("Starting Classification Model Tuning...")
        tuner = ClassificationModelTuner(config_path=classifiers_yaml, allowed_models=allowed_models)
        tuner.run_grid_search()
        tuner.save_results(tuned_model_yaml)
        logger.info("Tuning and evaluation complete.")
    except Exception as e:
        logger.critical("Fatal error in classifier tuning pipeline.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    execute_model_tune_evaluate()
