"""Generic model class for scikit-learn compatible estimators."""

import importlib
from typing import Union

import numpy as np
import pandas as pd

from match_predictor.model.base_model import BaseModel


class EstimatorModel(BaseModel):
    """Model class compatible with scikit-learn like estimators.

    Loads the estimator dynamically based on the configuration file or ModelConfig object.
    """

    def __init__(self, config_path: str = None, config=None):
        """Initialize the EstimatorModel with optional configuration.

        Args:
            config_path: Path to the configuration file (optional)
            config: ModelConfig object (optional, takes precedence over config_path)
        """
        super().__init__(config_path)

        # Handle ModelConfig object from match_predictor.config
        if config is not None:
            if hasattr(config, "estimator"):
                # It's a ModelConfig object
                estimator_config = config.estimator
                module_path = estimator_config.module
                class_name = estimator_config.class_name
                params = estimator_config.params
            else:  # pragma: no cover
                raise ValueError("Config object must have an 'estimator' attribute")
        else:
            # Use YAML-based config
            estimator_config = self.config.get("estimator", {})
            module_path = estimator_config.get("module")
            class_name = estimator_config.get("class_name")
            params = estimator_config.get("params", {})

        if not module_path or not class_name:
            raise ValueError("Estimator 'module' and 'class_name' must be specified in the config.")

        module = importlib.import_module(module_path)
        estimator_class = getattr(module, class_name)
        self.model = estimator_class(**params)

    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series]):
        """Train the model on the given data.

        Args:
            X: Features (array-like or DataFrame)
            y: Target values (array-like)
        """
        self.model.fit(X, y)

    def predict(self, model_input: Union[np.array, pd.DataFrame]):
        """Predict target values for given features.

        Args:
            model_input: Features (array-like or DataFrame)

        Returns:
            Predicted values (array-like or Series)
        """
        return self.model.predict(model_input)

    def evaluate(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series], **kwargs) -> float:
        """Evaluate the model on the given data.

        Args:
            X: Features (array-like or DataFrame)
            y: True target values (array-like)
            **kwargs: Additional keyword arguments for the scoring method

        Returns:
            Evaluation metric(s)
        """
        # Check if the model has a 'score' method
        if not hasattr(self.model, "score"):
            raise NotImplementedError("The underlying model does not implement a 'score' method.")
        else:
            return self.model.score(X, y, **kwargs)

    def save(self, filepath: str):
        """Save the model to a file.

        Args:
            filepath: Path to the file where the model will be saved
        """
        import joblib

        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> "EstimatorModel":
        """Load the model from a file.

        Args:
            filepath: Path to the file from which the model will be loaded

        Returns:
            model instance
        """
        import joblib

        self.model = joblib.load(filepath)


# if __name__ == "__main__":
#     # Example usage
#     model = EstimatorModel(config_path="./config/model_config.yaml")
#     print(model.model)

#     # Example data
#     X_example = pd.DataFrame({"feature1": [0.1, 0.2, 0.3], "feature2": [1.0, 0.9, 0.8]})
#     y_example = pd.Series([0, 1, 0])

#     model.fit(X_example, y_example)
#     predictions = model.predict(X_example)
#     print("Predictions:", predictions)
#     score = model.evaluate(X_example, y_example)
#     print("Evaluation Score:", score)
