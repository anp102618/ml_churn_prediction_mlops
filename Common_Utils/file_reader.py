import os
import sys
import zipfile
import logging
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type
from Common_Utils import CustomException, track_performance, setup_logger,load_yaml

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/config_path.yaml")


# ----------------------------
# Strategy Pattern Interfaces
# ----------------------------

class FileReaderStrategy(ABC):
    """Abstract base class for all file reading strategies."""

    @abstractmethod
    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read the file and return a DataFrame.

        Args:
            file_path (str): Path to the file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        pass


class CSVReader(FileReaderStrategy):
    """Concrete strategy to read CSV files."""

    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, index_col=0)
        except Exception as e:
            logger.error(f"Failed to read CSV: {file_path} | {e}")
            raise CustomException(e, sys)


class JSONReader(FileReaderStrategy):
    """Concrete strategy to read JSON files."""

    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_json(file_path)
        except Exception as e:
            logger.error(f"Failed to read JSON: {file_path} | {e}")
            raise CustomException(e, sys)


class ExcelReader(FileReaderStrategy):
    """Concrete strategy to read Excel files."""

    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Failed to read Excel: {file_path} | {e}")
            raise CustomException(e, sys)


# ----------------------------
# Factory Pattern
# ----------------------------

class FileReaderFactory:
    """
    Factory class to return appropriate FileReaderStrategy
    based on file extension.
    """

    registry: dict[str, Type[FileReaderStrategy]] = {
        ".csv": CSVReader,
        ".json": JSONReader,
        ".xls": ExcelReader,
        ".xlsx": ExcelReader
    }

    @classmethod
    def get_reader(cls, file_path: str) -> FileReaderStrategy:
        """
        Selects a reader strategy based on file extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            FileReaderStrategy: Strategy instance.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.registry:
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(f"Using reader for extension: {ext}")
        return cls.registry[ext]()


# ----------------------------
# Context Class
# ----------------------------

MAX_FILE_SIZE_MB = 100


class FileReaderContext:
    """
    Context class that uses a reader strategy to read files.
    """

    def __init__(self, strategy: FileReaderStrategy):
        self._strategy = strategy

    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads a file using the strategy.

        Args:
            file_path (str): File to read.

        Returns:
            pd.DataFrame: Loaded data.

        Raises:
            CustomException: On read failure or edge case violations.
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(file_path)

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                logger.error(f"File too large: {file_size_mb:.2f} MB")
                raise ValueError(f"File exceeds max size of {MAX_FILE_SIZE_MB} MB.")

            logger.info(f"Reading file: {file_path}")
            df = self._strategy.read(file_path)
            logger.info(f"Successfully read file. Shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


# ----------------------------
# Manager Class
# ----------------------------

class FileReaderManager:
    """
    Orchestrates the process of extracting, and reading data files.
    """

    @staticmethod
    def extract_zip(zip_path: str, extract_dir: str) -> str:
        """
        Extracts a zip file and returns the path to the first supported file.

        Args:
            zip_path (str): Path to zip.
            extract_dir (str): Where to extract.

        Returns:
            str: Path to first supported file.

        Raises:
            FileNotFoundError: If no readable file is found.
        """
        try:
            os.makedirs(extract_dir, exist_ok=True)
            logger.info(f"Extracting zip file: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            for root, _, files in os.walk(extract_dir):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in FileReaderFactory.registry:
                        file_path = os.path.join(root, file)
                        logger.info(f"Found readable file: {file_path}")
                        return file_path

            raise FileNotFoundError("No supported file found in extracted contents.")

        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    @staticmethod
    def load_dataset(zip_path: str, extract_dir: str) -> pd.DataFrame:
        """
        Downloads, extracts, and loads a Kaggle dataset.

        Args:
            dataset_name (str): Kaggle dataset slug.
            zip_dir (str): Directory to store zip file.
            extract_dir (str): Directory to extract contents.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            CustomException: On failure at any stage.
        """
        try:
            file_path = FileReaderManager.extract_zip(zip_path, extract_dir)
            strategy = FileReaderFactory.get_reader(file_path)
            context = FileReaderContext(strategy)
            return context.read_file(file_path)

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise CustomException(e, sys)
