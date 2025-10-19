"""Model configuration using Pydantic."""

from typing import Any

from pydantic import BaseModel, Field


class EstimatorConfig(BaseModel):
    """Configuration for the ML estimator."""

    module: str = Field(
        default="xgboost",
        description="Python module containing the estimator class"
    )
    class_name: str = Field(
        default="XGBClassifier",
        description="Name of the estimator class"
    )
    params: dict[str, Any] = Field(
        default={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "enable_categorical": True,
            "seed": 42,
            "n_estimators": 1000,
            "max_depth": 3,
            "learning_rate": 0.01,
            "colsample_bytree": 0.5,
        },
        description="Parameters for the estimator"
    )


class HyperparameterTuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    n_trials: int = Field(
        default=50,
        description="Number of Optuna trials for hyperparameter tuning"
    )
    cv_folds: int = Field(
        default=5,
        description="Number of cross-validation folds"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    timeout: int | None = Field(
        default=3600,
        description="Timeout for hyperparameter tuning in seconds (None for no timeout)"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(
        default=0.2,
        description="Proportion of data to use for testing"
    )
    validation_size: float = Field(
        default=0.2,
        description="Proportion of training data to use for validation"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    min_accuracy_threshold: float = Field(
        default=0.60,
        description="Minimum accuracy required to deploy a new model as champion"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""

    experiment_name: str = Field(
        default="tennis-match-predictor",
        description="MLflow experiment name"
    )
    tracking_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking URI"
    )
    model_name: str = Field(
        default="tennis_predictor_model",
        description="Registered model name in MLflow"
    )


class ModelConfig(BaseModel):
    """Main model configuration."""

    model_name: str = Field(
        default="atp_match_predictor",
        description="Name of the model"
    )
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = Field(default_factory=HyperparameterTuningConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    champion_model_path: str = Field(
        default="models/",
        description="Path to store the champion model"
    )
