from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    All derived classes must implement `fit_resample`.
    """

    @abstractmethod
    def fit_resample(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset to handle class imbalance.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features.
            y (Union[pd.Series, np.ndarray]): Target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Resampled X and y.
        """
        pass


class NoSamplingStrategy(SamplingStrategy):
    """
    No resampling applied. Returns X and y unchanged.
    """
    def fit_resample(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return X, y


class RandomUnderSamplerStrategy(SamplingStrategy):
    """
    Random undersampling of the majority class.
    """
    def __init__(self) -> None:
        self.sampler = RandomUnderSampler()

    def fit_resample(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return self.sampler.fit_resample(X, y)


class RandomOverSamplerStrategy(SamplingStrategy):
    """
    Random oversampling of the minority class.
    """
    def __init__(self) -> None:
        self.sampler = RandomOverSampler()

    def fit_resample(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return self.sampler.fit_resample(X, y)


class SMOTESamplerStrategy(SamplingStrategy):
    """
    Synthetic Minority Over-sampling Technique (SMOTE).
    """
    def __init__(self) -> None:
        self.sampler = SMOTE()

    def fit_resample(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return self.sampler.fit_resample(X, y)


class SamplingFactory:
    """
    Factory for creating sampling strategy instances.

    Supported strategies:
        - "none"
        - "undersample"
        - "oversample"
        - "smote"
    """

    @staticmethod
    def get_sampler(strategy: str) -> SamplingStrategy:
        """
        Create a SamplingStrategy instance based on the given strategy name.

        Args:
            strategy (str): Name of the sampling strategy.

        Returns:
            SamplingStrategy: A strategy instance implementing `fit_resample`.

        Raises:
            ValueError: If strategy is not supported.
        """
        strategy = strategy.strip().lower()

        if strategy == "none":
            return NoSamplingStrategy()
        elif strategy == "undersample":
            return RandomUnderSamplerStrategy()
        elif strategy == "oversample":
            return RandomOverSamplerStrategy()
        elif strategy == "smote":
            return SMOTESamplerStrategy()
        else:
            raise ValueError(
                f"Unknown sampling strategy: '{strategy}'. "
                f"Supported: ['none', 'undersample', 'oversample', 'smote']"
            )
