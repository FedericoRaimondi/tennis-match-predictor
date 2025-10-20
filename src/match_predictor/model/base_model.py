"""Base model class for machine learning models."""

from abc import ABC, abstractmethod
from typing import Union

import mlflow
import numpy as np
import pandas as pd


class BaseModel(ABC, mlflow.pyfunc.PythonModel):
    """Abstract base class for machine learning models."""

    def __init__(self, config_path: str = None):
        """Initialize the model with optional configuration.

        Args:
            config_path: Path to the configuration file (optional)
        """
        self.config_path = config_path
        self.model = None
        if config_path:
            # safe load yaml config
            import yaml

            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}

    @abstractmethod
    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series]):
        """Train the model on the given data.

        Args:
            X: Features (array-like or DataFrame)
            y: Target values (array-like)
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.Series]:
        """Predict target values for given features.

        Args:
            X: Features (array-like or DataFrame)

        Returns:
            Predicted values (array-like or Series)
        """
        pass

    @abstractmethod
    def evaluate(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series]) -> float:
        """Evaluate the model on the given data.

        Args:
            X: Features (array-like or DataFrame)
            y: True target values (array-like)

        Returns:
            Evaluation metric(s)
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the model to a file.

        Args:
            filepath: Path to the file where the model will be saved
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseModel":
        """Load the model from a file.

        Args:
            filepath: Path to the file from which the model will be loaded

        Returns:
            model instance
        """
        pass
