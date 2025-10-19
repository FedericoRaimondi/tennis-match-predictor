"""Model training module."""

import importlib
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from match_predictor.config import ModelConfig
from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer
from match_predictor.ml_pipeline.hyperparameter_tuning import HyperparameterTuner


class ModelTrainer:
    """Class for training machine learning models."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model trainer.

        Args:
            config: Model configuration (optional, defaults to ModelConfig())
        """
        self.config = config or ModelConfig()
        self.logger = logger
        self.model = None
        self.feature_names = None
        self.training_metrics = {}

    def load_estimator_class(self):
        """Load the estimator class from configuration."""
        module_path = self.config.estimator.module
        class_name = self.config.estimator.class_name

        try:
            module = importlib.import_module(module_path)
            estimator_class = getattr(module, class_name)
            self.logger.info(f"Loaded estimator: {module_path}.{class_name}")
            return estimator_class
        except Exception as e:
            self.logger.error(f"Failed to load estimator: {e}")
            raise

    def train(self, df: pd.DataFrame, tune_hyperparameters: bool = False, log_to_mlflow: bool = True) -> dict:
        """Train the model.

        Args:
            df: Training dataframe with features and target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting model training...")

        # Engineer features
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.engineer_features(df)

        # Prepare features and target
        X, y = feature_engineer.prepare_features_for_training(df_engineered)
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.training.test_size, random_state=self.config.training.random_state, stratify=y
        )

        # Further split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.config.training.validation_size,
            random_state=self.config.training.random_state,
            stratify=y_train,
        )

        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Validation set size: {len(X_val)}")
        self.logger.info(f"Test set size: {len(X_test)}")

        # Load estimator class
        estimator_class = self.load_estimator_class()

        # Hyperparameter tuning
        if tune_hyperparameters:
            self.logger.info("Performing hyperparameter tuning...")
            tuner = HyperparameterTuner(self.config)
            best_params = tuner.tune(estimator_class, X_train, y_train)
        else:
            best_params = self.config.estimator.params

        # Train final model
        self.logger.info("Training final model with best parameters...")
        self.model = estimator_class(**best_params)
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Evaluate on test set
        test_predictions = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Store metrics
        self.training_metrics = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "hyperparameters": best_params,
            "feature_count": len(self.feature_names),
        }

        self.logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Log to MLflow if enabled
        if log_to_mlflow:
            self._log_to_mlflow(X_train, y_train, X_test, y_test, val_accuracy, test_accuracy, best_params)

        return self.training_metrics

    def _log_to_mlflow(self, X_train, y_train, X_test, y_test, val_accuracy, test_accuracy, params):
        """Log training run to MLflow."""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))

            # Log model
            mlflow.xgboost.log_model(
                self.model, "model", registered_model_name=self.config.mlflow.model_name, input_example=X_train.iloc[:5]
            )

            # Log confusion matrix and classification report
            y_pred = self.model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")

            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            self.logger.info("Logged training run to MLflow")

    def save_model(self, path: str | Path):
        """Save the trained model.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "config": self.config,
                    "metrics": self.training_metrics,
                },
                f,
            )

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path):
        """Load a trained model.

        Args:
            path: Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            saved_data = pickle.load(f)

        self.model = saved_data["model"]
        self.feature_names = saved_data["feature_names"]
        self.training_metrics = saved_data.get("metrics", {})

        self.logger.info(f"Model loaded from {path}")

    def should_promote_to_champion(self, champion_accuracy: float | None = None) -> bool:
        """Determine if the current model should be promoted to champion.

        Args:
            champion_accuracy: Accuracy of the current champion model (if exists)

        Returns:
            Boolean indicating if model should be promoted
        """
        if "test_accuracy" not in self.training_metrics:
            self.logger.warning("No test accuracy found in training metrics")
            return False

        current_accuracy = self.training_metrics["test_accuracy"]

        # Check minimum threshold
        if current_accuracy < self.config.training.min_accuracy_threshold:
            self.logger.info(
                f"Model accuracy {current_accuracy:.4f} below threshold "
                f"{self.config.training.min_accuracy_threshold:.4f}"
            )
            return False

        # If no champion exists, promote
        if champion_accuracy is None:
            self.logger.info("No champion model exists, promoting current model")
            return True

        # Compare with champion
        if current_accuracy > champion_accuracy:
            self.logger.info(f"Model accuracy {current_accuracy:.4f} better than champion {champion_accuracy:.4f}")
            return True

        self.logger.info(f"Model accuracy {current_accuracy:.4f} not better than champion {champion_accuracy:.4f}")
        return False
