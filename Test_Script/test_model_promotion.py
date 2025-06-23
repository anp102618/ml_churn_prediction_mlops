import unittest
import yaml
import logging
from typing import Dict, Any

from Common_Utils import load_yaml, setup_logger, copy_selected_files, copy_csv_file

logger = setup_logger("logs")


class SimpleModelTest(unittest.TestCase):
    """
    Unit test class for validating XGBClassifier model performance.
    If performance metrics meet specified thresholds on all splits (train, val, test),
    the model is marked as 'production' in the YAML metadata.
    """

    def setUp(self) -> None:
        """
        Initializes test configuration by loading metrics from YAML.
        """
        self.curr_yaml_path: str = "Tuned_Model/mlflow_details.yaml"

        try:
            self.metrics: Dict[str, Any] = load_yaml(self.curr_yaml_path).get("scores", {})
            if not self.metrics:
                logger.warning("No 'scores' section found in YAML file.")
        except Exception as e:
            logger.exception(f"Error loading YAML file at {self.curr_yaml_path}")
            raise e

    def check_metrics(self) -> bool:
        """
        Verifies that accuracy > 0.90 and ROC AUC > 0.80 for train, val, and test splits.

        Returns:
            bool: True if all metrics pass the threshold, False otherwise.
        """
        required_splits = ["train", "val", "test"]
        threshold_accuracy = 0.90
        threshold_auc = 0.80

        for split in required_splits:
            split_metrics = self.metrics.get(split)
            if not split_metrics:
                logger.warning(f"Missing metrics for split: {split}")
                return False

            acc = split_metrics.get("accuracy")
            auc = split_metrics.get("roc_auc")

            if acc is None or auc is None:
                logger.warning(f"Incomplete metrics for split: {split} - accuracy: {acc}, roc_auc: {auc}")
                return False

            logger.info(f"{split.upper()} - Accuracy: {acc}, ROC AUC: {auc}")

            if acc < threshold_accuracy or auc < threshold_auc:
                logger.warning(f"{split.upper()} metrics did not meet threshold: acc < {threshold_accuracy} or auc < {threshold_auc}")
                return False

        return True

    def test_model_passes(self) -> None:
        """
        Test case to check model promotion based on performance metrics.
        If passed, updates the YAML with stage='production'.
        """
        try:
            if self.check_metrics():
                logger.info("All metrics passed. Promoting model to production.")

                with open(self.curr_yaml_path, "r") as f:
                    meta = yaml.safe_load(f)

                meta.setdefault("mlflow_run", {})["stage"] = "production"

                with open(self.curr_yaml_path, "w") as f:
                    yaml.safe_dump(meta, f)

                print("Test Passed: Model promoted to production.")
            else:
                logger.warning("Test Failed: Model metrics did not meet the threshold.")
                print("Test Failed: Accuracy or ROC AUC below threshold.")

        except Exception as e:
            logger.exception("An error occurred during the model promotion test.")
            self.fail(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    unittest.main()
    copy_selected_files(source_dir="Tuned_Model", destination_dir="Data/ref_data", file_types=[".joblib", ".yaml",".json", ".csv" ])
    copy_selected_files(source_dir="Data/processed_data", destination_dir="Data/ref_data", file_types=[".csv"])
    copy_csv_file(source_file="Data/raw_data/extracted_files/bank_churners.csv", destination_folder="Data/ref_data")
