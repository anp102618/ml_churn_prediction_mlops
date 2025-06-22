import yaml
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from Model_Utils.classification_models import ModelFactory
from Model_Utils.feature_sampling import SamplingFactory
from Model_Utils.feature_selection_extraction import FeatureFactory
from Common_Utils import setup_logger, track_performance, CustomException, load_yaml

logger = setup_logger("logs")
config = load_yaml("Config_Yaml/config_path.yaml")

# Paths
paths = config["Model_Tune_Evaluate"]["path"]
processed_data_dir = Path(paths["processed_data_dir"])
tuned_model_yaml = Path(paths["tuned_model_yaml"])
classifiers_yaml = Path(paths["classifiers_yaml"])
selector_path = Path(paths["selector_path"])
extractor_path = Path(paths["extractor_path"])

# Constants
const = config["Model_Tune_Evaluate"]["const"]
allowed_models = const["allowed_models"]
sampling_method = const["sampling"]
selection_method = const["selection"]
extraction_method = const["extraction"]
k = const["k"]
n_components = const["n_components"]

def get_search_space(param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {
        key: (Real(min(vals), max(vals)) if isinstance(vals[0], float)
              else Integer(min(vals), max(vals)) if isinstance(vals[0], int)
              else Categorical(vals))
        for key, vals in param_grid.items()
    }

def prepare_features(X: pd.DataFrame, encoder_scaler, selector, extractor, expected_cols: list) -> pd.DataFrame:
    X_enc = encoder_scaler.transform(X)
    X_enc = pd.DataFrame(X_enc, columns=encoder_scaler.get_feature_names_out(), index=X.index)
    X_enc = X_enc.reindex(columns=expected_cols, fill_value=0)
    return extractor.transform(selector.transform(X_enc))

class ClassificationModelTuner:
    def __init__(self, config_path: Path):
        self.factory = ModelFactory(config_path)
        self.X_train = pd.read_csv(processed_data_dir / "X_train.csv")
        self.y_train = pd.read_csv(processed_data_dir / "y_train.csv").iloc[:, 0]
        self.X_val = pd.read_csv(processed_data_dir / "X_val.csv")
        self.y_val = pd.read_csv(processed_data_dir / "y_val.csv").iloc[:, 0]
        self.X_test = pd.read_csv(processed_data_dir / "X_test.csv")
        self.y_test = pd.read_csv(processed_data_dir / "y_test.csv").iloc[:, 0]
        self.results = []

    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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
    def run_grid_search(self):
        try:
            model_config = load_yaml(classifiers_yaml)

            for model_name, model_info in model_config.items():
                if allowed_models and model_name not in allowed_models:
                    continue

                sampler = SamplingFactory.get_sampler(sampling_method)
                selector = FeatureFactory.get_processor("selection", selection_method, k=k)
                extractor = FeatureFactory.get_processor("extraction", extraction_method, n_components=n_components)

                X_sampled, y_sampled = sampler.fit_resample(self.X_train, self.y_train)
                X_selected = selector.fit_transform(X_sampled, y_sampled)
                X_extracted = extractor.fit_transform(X_selected, y_sampled)

                model = self.factory.get_model(model_name)
                search_space = get_search_space(model_info["params"])
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                opt = BayesSearchCV(model, search_space, n_iter=20, cv=kf, scoring='roc_auc', n_jobs=-1, random_state=42)
                opt.fit(X_extracted, y_sampled)
                best_model = opt.best_estimator_

                # Fine-tuning
                encoder_scaler = joblib.load("Tuned_Model/encoder_scaler.joblib")
                X_val_extracted = prepare_features(self.X_val, encoder_scaler, selector, extractor, self.X_train.columns)
                X_finetune = np.concatenate([X_extracted, X_val_extracted])
                y_finetune = np.concatenate([y_sampled, self.y_val])
                best_model.fit(X_finetune, y_finetune)

                # Save artifacts
                selector_path.parent.mkdir(parents=True, exist_ok=True)
                extractor_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(selector, selector_path)
                joblib.dump(extractor, extractor_path)

                # Evaluate on test set
                X_test_extracted = prepare_features(self.X_test, encoder_scaler, selector, extractor, self.X_train.columns)
                test_scores = self.evaluate(best_model, X_test_extracted, self.y_test)

                self.results.append({
                    "model": model_name,
                    "params": opt.best_params_,
                    "roc_auc": round(test_scores['roc_auc'], 4),
                    "scores": {k: round(v, 4) for k, v in test_scores.items()}
                })

        except Exception as e:
            logger.exception("Bayesian search failed.")
            raise CustomException(e, sys)

    def save_results(self, output_path: Path):
        try:
            sorted_results = sorted(self.results, key=lambda x: x['roc_auc'], reverse=True)
            with open(output_path, 'w') as f:
                yaml.dump(sorted_results, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Best model: {sorted_results[0]['model']} | ROC AUC: {sorted_results[0]['roc_auc']}")
        except Exception as e:
            logger.exception("Failed to save results.")
            raise CustomException(e, sys)

@track_performance
def execute_model_tune_evaluate():
    try:
        logger.info("Starting Classification Model Tuning...")
        tuner = ClassificationModelTuner(config_path=classifiers_yaml)
        tuner.run_grid_search()
        tuner.save_results(tuned_model_yaml)
        logger.info("Tuning and evaluation complete.")
    except Exception as e:
        logger.critical("Fatal error in tuning pipeline.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    execute_model_tune_evaluate()
