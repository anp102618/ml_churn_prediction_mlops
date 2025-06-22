from abc import ABC, abstractmethod
from typing import Union, List, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import TransformerMixin


# ------------------------ Strategy Pattern ------------------------ #

class ImputerStrategy(ABC):
    """
    Abstract base class for imputation strategies.
    All derived strategies must implement `get_imputer()`.
    """
    @abstractmethod
    def get_imputer(self) -> TransformerMixin:
        """
        Returns:
            TransformerMixin: A scikit-learn compatible imputer.
        """
        pass


class MeanImputerStrategy(ImputerStrategy):
    def get_imputer(self) -> SimpleImputer:
        """Imputer using mean for numeric columns."""
        return SimpleImputer(strategy="mean")


class MedianImputerStrategy(ImputerStrategy):
    def get_imputer(self) -> SimpleImputer:
        """Imputer using median for numeric columns."""
        return SimpleImputer(strategy="median")


class MostFrequentImputerStrategy(ImputerStrategy):
    def get_imputer(self) -> SimpleImputer:
        """Imputer using the most frequent value for categorical columns."""
        return SimpleImputer(strategy="most_frequent")


class KNNImputerStrategy(ImputerStrategy):
    def get_imputer(self) -> KNNImputer:
        """Imputer using K-Nearest Neighbors for numeric columns."""
        return KNNImputer(n_neighbors=5)


class IterativeImputerStrategy(ImputerStrategy):
    def get_imputer(self) -> IterativeImputer:
        """Imputer using multivariate iterative modeling (Bayesian Ridge by default)."""
        return IterativeImputer(random_state=42)


# ------------------------ Factory ------------------------ #

class ImputerFactory:
    """
    Factory class to get an imputer instance based on strategy name.
    Supported strategies:
        - "mean"
        - "median"
        - "mode" / "most_frequent"
        - "knn"
        - "iterative"
    """
    @staticmethod
    def get_imputer(strategy: str) -> TransformerMixin:
        """
        Args:
            strategy (str): The imputation strategy name.

        Returns:
            TransformerMixin: Corresponding sklearn-compatible imputer.

        Raises:
            ValueError: If strategy is not supported.
        """
        strategy = strategy.strip().lower()
        if strategy == "mean":
            return MeanImputerStrategy().get_imputer()
        elif strategy == "median":
            return MedianImputerStrategy().get_imputer()
        elif strategy in ("mode", "most_frequent"):
            return MostFrequentImputerStrategy().get_imputer()
        elif strategy == "knn":
            return KNNImputerStrategy().get_imputer()
        elif strategy == "iterative":
            return IterativeImputerStrategy().get_imputer()
        else:
            raise ValueError(f"Unknown imputation strategy: '{strategy}'. "
                             f"Supported: ['mean', 'median', 'mode', 'knn', 'iterative'].")


# ------------------------ Data Imputer Wrapper ------------------------ #

class DataImputer:
    """
    Unified imputer class that automatically applies numeric or categorical
    imputation based on column types.
    
    Attributes:
        numeric_imputer (TransformerMixin): Imputer for numeric columns.
        categorical_imputer (TransformerMixin): Imputer for categorical columns.
        numeric_cols (List[str]): List of numeric columns detected in fit().
        categorical_cols (List[str]): List of categorical columns detected in fit().

    - Uses one imputer object internally for numeric and one for categorical.
    - Can be used on both X (multiple columns) and y (single-column) DataFrames.
    """

    def __init__(self,
                 numeric_strategy: str = "mean",
                 categorical_strategy: str = "most_frequent") -> None:
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_imputer: TransformerMixin = self._get_imputer(numeric_strategy)
        self.categorical_imputer: TransformerMixin = self._get_imputer(categorical_strategy)
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._fitted = False

    def _get_imputer(self, strategy: str) -> TransformerMixin:
        strategy = strategy.lower()
        if strategy == "mean":
            return SimpleImputer(strategy="mean")
        elif strategy == "median":
            return SimpleImputer(strategy="median")
        elif strategy in ("mode", "most_frequent"):
            return SimpleImputer(strategy="most_frequent")
        elif strategy == "knn":
            return KNNImputer(n_neighbors=5)
        elif strategy == "iterative":
            return IterativeImputer(random_state=42)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if self.numeric_cols:
            self.numeric_imputer.fit(df[self.numeric_cols])
        if self.categorical_cols:
            self.categorical_imputer.fit(df[self.categorical_cols])

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("You must call fit() before transform().")

        df_copy = df.copy()
        if self.numeric_cols:
            df_copy[self.numeric_cols] = self.numeric_imputer.transform(df_copy[self.numeric_cols])
        if self.categorical_cols:
            df_copy[self.categorical_cols] = self.categorical_imputer.transform(df_copy[self.categorical_cols])
        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

