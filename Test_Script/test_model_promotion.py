import unittest
import yaml
import logging
from Common_Utils import load_yaml, setup_logger

logger = setup_logger("logs")


class SimpleModelTest(unittest.TestCase):
    """
    Minimal unit test for XGBClassifier promotion check.
    """

    def setUp(self):
        self.curr_yaml_path = "Tuned_Model/mlflow_details.yaml"
        self.metrics = load_yaml(self.curr_yaml_path).get("scores", {})

    def check_metrics(self) -> bool:
        """
        Verifies that accuracy > 0.90 and ROC AUC > 0.80 for train, val, test.

        Returns:
            bool: True if all thresholds are satisfied.
        """
        for split in ["train", "val", "test"]:
            acc = self.metrics.get(split, {}).get("accuracy", 0)
            auc = self.metrics.get(split, {}).get("roc_auc", 0)
            logger.info(f"{split.upper()} - Accuracy: {acc}, ROC AUC: {auc}")
            if acc < 0.90 or auc < 0.80:
                return False
        return True

    def test_model_passes(self):
        """
        Runs basic threshold checks and promotes model if successful.
        """
        if self.check_metrics():
            logger.info("All metrics passed. Promoting model.")
            with open(self.curr_yaml_path, "r") as f:
                meta = yaml.safe_load(f)
            meta.setdefault("mlflow_run", {})["stage"] = "production"
            with open(self.curr_yaml_path, "w") as f:
                yaml.safe_dump(meta, f)
            print("Test Passed: Model promoted to production.")
        else:
            logger.warning("Model metrics did not meet the threshold.")
            print("Test Failed: Accuracy or ROC AUC below threshold.")


if __name__ == "__main__":
    unittest.main()
