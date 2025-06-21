import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from Common_Utils import CustomException, track_performance, setup_logger,load_yaml

# Setup logger
logger = setup_logger(filename="logs")
config = load_yaml("Config_Yaml/classifiers.yaml")


class ModelFactory:
    """
    A factory class to build, train, and predict using various classification models,
    including both sklearn and Keras models.
    """

    def __init__(self, config_path: str):
        """
        Initializes the factory with model configuration from a YAML file.

        Args:
            config_path (str): Path to the model configuration YAML file.

        Raises:
            ValueError: If the config file cannot be loaded.
        """
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logger.error("Failed to load configuration file.", exc_info=True)
            raise ValueError(f"Error loading config: {e}")

    def get_model(self, model_name: str, param_override: Optional[Dict] = None) -> Union[BaseEstimator, tf.keras.Model]:
        """
        Instantiates a model based on the name and parameters.

        Args:
            model_name (str): Name of the model (e.g., 'mlp', 'svc').
            param_override (dict, optional): Parameters to override defaults.

        Returns:
            Union[BaseEstimator, tf.keras.Model]: Configured model instance.

        Raises:
            ValueError: If the model is not supported or config is missing.
        """
        try:
            if model_name not in self.config:
                raise ValueError(f"Model '{model_name}' not found in configuration")

            params = self.config[model_name].get("params", {}).copy()
            if param_override:
                params.update(param_override)

            if model_name == "logistic_regression":
                return LogisticRegression(**params)
            elif model_name == "random_forest":
                return RandomForestClassifier(**params)
            elif model_name == "svc":
                return SVC(**params)
            elif model_name == "xgboost":
                return XGBClassifier(**params)
            elif model_name == "mlp":
                return self._build_keras_mlp(params)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        except Exception as e:
            logger.error("Error initializing model.", exc_info=True)
            raise e

    def _build_keras_mlp(self, params: Dict) -> tf.keras.Model:
        """
        Builds a Keras MLP classifier model.

        Args:
            params (Dict): Dictionary of model hyperparameters.

        Returns:
            tf.keras.Model: Compiled Keras model.

        Raises:
            ValueError: If required parameters are missing.
        """
        try:
            input_dim = params.get("input_dim")
            if input_dim is None:
                raise ValueError("'input_dim' must be specified for MLP model")

            model = Sequential()
            for i, units in enumerate(params.get("hidden_layers", [64, 32])):
                if i == 0:
                    model.add(Dense(units, activation=params.get("activation", "relu"), input_dim=input_dim))
                else:
                    model.add(Dense(units, activation=params.get("activation", "relu")))
                model.add(Dropout(params.get("dropout", 0.2)))

            model.add(Dense(1, activation="sigmoid"))

            optimizer_name = params.get("optimizer", "adam").lower()
            learning_rate = params.get("learning_rate", 0.001)
            if optimizer_name == "adam":
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == "rmsprop":
                optimizer = RMSprop(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            return model

        except Exception as e:
            logger.error("Error building Keras MLP model.", exc_info=True)
            raise e

    def train(
        self,
        model: Union[BaseEstimator, tf.keras.Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Union[BaseEstimator, tf.keras.callbacks.History]:
        """
        Trains the given model.

        Args:
            model: Model instance.
            X_train: Training features.
            y_train: Training labels.
            model_name: Model name.
            X_val: Optional validation features (only for MLP).
            y_val: Optional validation labels (only for MLP).

        Returns:
            Trained model or Keras training history.
        """
        try:
            if model_name == "mlp":
                params = self.config[model_name].get("params", {})
                return model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                    epochs=params.get("epochs", 20),
                    batch_size=params.get("batch_size", 32),
                    verbose=1
                )
            else:
                model.fit(X_train, y_train)
                return model
        except Exception as e:
            logger.error("Training failed.", exc_info=True)
            raise e

    def predict(
        self,
        model: Union[BaseEstimator, tf.keras.Model],
        X_test: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """
        Predicts using the trained model.

        Args:
            model: Trained model.
            X_test: Test features.
            model_name: Model name.

        Returns:
            np.ndarray: Model predictions.
        """
        try:
            if model_name == "mlp":
                return (model.predict(X_test) > 0.5).astype("int").flatten()
            else:
                return model.predict(X_test)
        except Exception as e:
            logger.error("Prediction failed.", exc_info=True)
            raise e
