from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    Each strategy must implement the `sample` method.
    """
    @abstractmethod
    def sample(self, X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pass


class NoSamplingStrategy(SamplingStrategy):
    """
    No sampling is applied; returns X and y unchanged.
    """
    def sample(self, X, y):
        return X, y


class RandomUnderSamplerStrategy(SamplingStrategy):
    """
    Undersamples the majority class using random sampling.
    """
    def __init__(self):
        self.sampler = RandomUnderSampler()

    def sample(self, X, y):
        return self.sampler.fit_resample(X, y)


class RandomOverSamplerStrategy(SamplingStrategy):
    """
    Oversamples the minority class using random sampling.
    """
    def __init__(self):
        self.sampler = RandomOverSampler()

    def sample(self, X, y):
        return self.sampler.fit_resample(X, y)


class SMOTESamplerStrategy(SamplingStrategy):
    """
    Oversamples the minority class using SMOTE.
    """
    def __init__(self):
        self.sampler = SMOTE()

    def sample(self, X, y):
        return self.sampler.fit_resample(X, y)


class SamplingFactory:
    """
    Factory class for creating a sampling strategy instance based on a given name.

    Supported strategies:
        - "none"
        - "undersample"
        - "oversample"
        - "smote"
    """

    @staticmethod
    def get_sampler(strategy: str) -> SamplingStrategy:
        """
        Returns a sampling strategy instance based on the given strategy name.

        Args:
            strategy (str): Name of the sampling strategy.

        Returns:
            SamplingStrategy: An instance implementing the sampling logic.

        Raises:
            ValueError: If an unsupported sampling strategy is provided.
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
            raise ValueError(f"Unknown sampling strategy: '{strategy}'. "
                             f"Supported: ['none', 'undersample', 'oversample', 'smote']")
