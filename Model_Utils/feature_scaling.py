from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.base import TransformerMixin
from typing import Union


class ScalerFactory:
    """
    Factory class for returning scikit-learn scaler instances directly.

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
        Return a scaler object based on the strategy name.

        Args:
            strategy (str): The name of the scaling strategy.

        Returns:
            TransformerMixin: A scikit-learn scaler instance.
        """
        strategy = strategy.strip().lower()

        if strategy == "standard":
            return StandardScaler()
        elif strategy == "minmax":
            return MinMaxScaler()
        elif strategy == "robust":
            return RobustScaler()
        elif strategy == "maxabs":
            return MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: '{strategy}'. "
                             f"Supported: ['standard', 'minmax', 'robust', 'maxabs']")
