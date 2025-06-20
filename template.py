import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "__init__.py",
    ".github/workflows/.gitkeep",
    ".github/workflows/ci-cd.yaml",
    ".gitignore",
    f"Common_Utils/__init__.py",
    f"Common_Utils/file_reader.py",
    f"Common_Utils/sqlite_handler.py",
    f"Common_Utils/dataframe_methods.py",
    

    f"Data/__init__.py",
    f"Data/raw_data/zip_files/__init__.py",
    f"Data/raw_data/extracted_files/__init__.py",
    f"Data/data.db",
    f"Data/raw_data/__init__.py",
    f"Data/processed_data/__init__.py",

    f"Tuned_Model/__init__.py",
    
    f"Config_Yaml/__init__.py",
    f"Config_Yaml/config_path.yaml",
    f"Config_Yaml/classifiers.yaml",

    f"Model_Utils/__init__.py",
    f"Model_Utils/feature_nan_imputation.py",
    f"Model_Utils/feature_outlier_handling.py",
    f"Model_Utils/classsification_models.py",
    f"Model_Utils/feature_scaling.py",
    f"Model_Utils/feature_encoding.py",
    f"Model_Utils/feature_sampling.py",
    f"Model_Utils/feature_selection_extraction.py",
  

    f"src/__init__.py",
    f"src/Data_Ingestion.py",
    f"src/Data_Validation.py",
    f"src/Data_Preprocessing.py",
    f"src/Data_Transformation.py",
    f"src/Model_tune_evaluate.py",
    f"src/Experiment_Tracking_Prediction.py",

    f"Tuned_Model/__init__.py",

    f"Test_Script/__init__.py",
    f"Test_Script/test_model_promotion.py",
    
    f"k8s/fastapi-deployment.yaml",
    f"k8s/fastapi-service.yaml",
    f"k8s/prometheus-configmap.yaml",
    f"k8s/prometheus-deployment.yaml",
    f"k8s/prometheus-service.yaml",
    f"k8s/grafana-deployment.yaml",
    f"k8s/grafana-service.yaml",

    "app.py",
    "EDA.ipynb",
    "requirements.txt",
    "app_requirements.txt",
    "setup.py",
    "main.py",
    "streamlit_app.py",
    "Dockerfile",
    ".dockerignore",
    
 ]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")