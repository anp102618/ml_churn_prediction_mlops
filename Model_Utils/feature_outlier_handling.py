import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer
from typing import List


def find_outlier_columns(df: pd.DataFrame, threshold: float = 1.5) -> List[str]:
    """
    Identify numeric columns in the DataFrame that contain outliers
    using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): IQR threshold to detect outliers.

    Returns:
        List[str]: Column names with detected outliers.
    """
    if df.empty:
        return []

    outlier_cols = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        if ((df[col] < lower) | (df[col] > upper)).any():
            outlier_cols.append(col)

    return outlier_cols


# ---------------- Strategy Pattern ---------------- #

class OutlierHandlerStrategy(ABC):
    """
    Abstract base class for outlier handling strategies.
    """
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply transformation to handle outliers on specified columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        pass


class LogTransformStrategy(OutlierHandlerStrategy):
    """
    Applies logarithmic transformation to positive-valued columns.
    """
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        applicable_cols = []

        for col in columns:
            if (df_copy[col] <= 0).any():
                continue  # log undefined for non-positive values
            applicable_cols.append(col)
            df_copy[col] = np.log(df_copy[col])

        if not applicable_cols:
            print("⚠️ No columns suitable for log transform (non-positive values present).")

        return df_copy


class YeoJohnsonTransformStrategy(OutlierHandlerStrategy):
    """
    Applies Yeo-Johnson transformation to handle both positive and negative values.
    """
    def handle(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        if not columns:
            print("⚠️ No columns provided for Yeo-Johnson transform.")
            return df_copy

        try:
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            df_copy[columns] = transformer.fit_transform(df_copy[columns])
        except Exception as e:
            print(f"Yeo-Johnson transformation failed: {e}")

        return df_copy


# ---------------- Factory ---------------- #

class OutlierHandlerFactory:
    """
    Factory to create appropriate outlier transformation strategy.
    """
    @staticmethod
    def get_handler(strategy: str) -> OutlierHandlerStrategy:
        strategy = strategy.strip().lower()
        if strategy == 'log':
            return LogTransformStrategy()
        elif strategy == 'yeo':
            return YeoJohnsonTransformStrategy()
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Supported: ['log', 'yeo'].")


# ---------------- OutlierHandler Interface ---------------- #

class OutlierHandler:
    """
    Handles outlier detection and transformation using a chosen strategy.

    Attributes:
        iqr_threshold (float): Threshold multiplier for IQR method.
        outlier_columns (List[str]): Columns with detected outliers.
    """
    def __init__(self, iqr_threshold: float = 1.5):
        self.iqr_threshold: float = iqr_threshold
        self.outlier_columns: List[str] = []
        self._strategy: OutlierHandlerStrategy = None
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Detects outlier columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        self.outlier_columns = find_outlier_columns(df, self.iqr_threshold)
        self._fitted = True

    def transform(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted outlier columns and strategy.

        Args:
            df (pd.DataFrame): DataFrame to transform.
            strategy (str): Strategy to apply ("log", "yeo").

        Returns:
            pd.DataFrame: Transformed DataFrame.

        Raises:
            ValueError: If `fit()` was not called before transform.
        """
        if not self._fitted:
            raise ValueError("You must call `fit()` before `transform()`.")

        self._strategy = OutlierHandlerFactory.get_handler(strategy)
        return self._strategy.handle(df, self.outlier_columns)

    def fit_transform(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Convenience method to detect outliers and transform in one step.

        Args:
            df (pd.DataFrame): Input DataFrame.
            strategy (str): Strategy to use ('log', 'yeo').

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df, strategy)
