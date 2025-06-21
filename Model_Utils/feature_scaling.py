from abc import ABC, abstractmethod
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.base import TransformerMixin


class ScalerStrategy(ABC):
    """
    Abstract base class for scaler strategies.
    All subclasses must implement the `get_scaler` method.
    """
    @abstractmethod
    def get_scaler(self) -> TransformerMixin:
        pass


class StandardScalerStrategy(ScalerStrategy):
    """
    Standard Scaler: Removes the mean and scales to unit variance.
    """
    def get_scaler(self) -> StandardScaler:
        return StandardScaler()


class MinMaxScalerStrategy(ScalerStrategy):
    """
    Min-Max Scaler: Scales features to a given range (default 0 to 1).
    """
    def get_scaler(self) -> MinMaxScaler:
        return MinMaxScaler()


class RobustScalerStrategy(ScalerStrategy):
    """
    Robust Scaler: Scales using statistics that are robust to outliers.
    """
    def get_scaler(self) -> RobustScaler:
        return RobustScaler()


class MaxAbsScalerStrategy(ScalerStrategy):
    """
    MaxAbs Scaler: Scales each feature by its maximum absolute value.
    Suitable for data that is already centered at zero without outliers.
    """
    def get_scaler(self) -> MaxAbsScaler:
        return MaxAbsScaler()


class ScalerFactory:
    """
    Factory class for creating scaler instances based on the given strategy name.

    Supported strategies:
        - "standard"
        - "minmax"
        - "robust"
        - "maxabs"
    """

    @staticmethod
    def get_scaler(strategy: str) -> Union[
        StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    ]:
        """
        Returns a scaler object based on the selected strategy.

        Args:
            strategy (str): The name of the scaling strategy.

        Returns:
            TransformerMixin: A scikit-learn scaler instance.

        Raises:
            ValueError: If an unsupported strategy is provided.
        """
        strategy = strategy.strip().lower()

        if strategy == "standard":
            return StandardScalerStrategy().get_scaler()
        elif strategy == "minmax":
            return MinMaxScalerStrategy().get_scaler()
        elif strategy == "robust":
            return RobustScalerStrategy().get_scaler()
        elif strategy == "maxabs":
            return MaxAbsScalerStrategy().get_scaler()
        else:
            raise ValueError(f"Unknown scaling strategy: '{strategy}'. "
                             f"Supported: ['standard', 'minmax', 'robust', 'maxabs']")
