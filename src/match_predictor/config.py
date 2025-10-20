"""Configuration management using Pydantic with YAML support."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""

    github_repo: str = Field(default="JeffSackmann/tennis_atp", description="GitHub repository for tennis data")
    selected_year: int = Field(default=1991, description="Starting year for match data filtering")
    tourney_levels: list[str] = Field(
        default=["G", "F", "M", "A"],
        description="Tournament levels to include (G=Grand Slam, F=Tour Finals, M=Masters, A=ATP Tour)",
    )


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    rolling_windows: list[int] = Field(default=[3, 5, 10], description="Window sizes for rolling statistics")
    stats_columns_mean: list[str] = Field(
        default=[
            "ace",
            "df",
            "svpt",
            "1stIn",
            "1stWon",
            "2ndWon",
            "SvGms",
            "bpSaved",
            "bpFaced",
            "rank_points",
            "minutes",
        ],
        description="Columns for which to calculate rolling mean",
    )
    stats_columns_sum: list[str] = Field(
        default=["minutes", "results"], description="Columns for which to calculate rolling sum"
    )
    elo_k_factor: float = Field(default=32.0, description="K-factor for ELO rating calculation")
    elo_initial_rating: float = Field(default=1500.0, description="Initial ELO rating for new players")


class DataConfig(BaseModel):
    """Main data configuration."""

    source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    inference_data_path: str = Field(default="data/", description="Path to store inference data")
    matches_results_file: str = Field(default="matches_results.pkl", description="Filename for matches results data")
    tournament_info_file: str = Field(default="tournament_info.pkl", description="Filename for tournament info data")

    player_stats_file: str = Field(
        default="player_stats_latest.pkl", description="Filename for latest player stats data"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config/data_config.yaml") -> "DataConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            DataConfig instance with values from YAML file
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict) if config_dict else cls()


class EstimatorConfig(BaseModel):
    """Configuration for the ML estimator."""

    module: str = Field(default="xgboost", description="Python module containing the estimator class")
    class_name: str = Field(default="XGBClassifier", description="Name of the estimator class")
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
        description="Parameters for the estimator",
    )


class HyperparameterTuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    n_trials: int = Field(default=50, description="Number of Optuna trials for hyperparameter tuning")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    timeout: int | None = Field(
        default=3600, description="Timeout for hyperparameter tuning in seconds (None for no timeout)"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(default=0.2, description="Proportion of data to use for testing")
    validation_size: float = Field(default=0.2, description="Proportion of training data to use for validation")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    min_accuracy_threshold: float = Field(
        default=0.60, description="Minimum accuracy required to deploy a new model as champion"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""

    experiment_name: str = Field(default="tennis-match-predictor", description="MLflow experiment name")
    tracking_uri: str = Field(default="file:./mlruns", description="MLflow tracking URI")
    model_name: str = Field(default="tennis_predictor_model", description="Registered model name in MLflow")


class ModelConfig(BaseModel):
    """Main model configuration."""

    model_name: str = Field(default="atp_match_predictor", description="Name of the model")
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = Field(default_factory=HyperparameterTuningConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    champion_model_path: str = Field(default="models/", description="Path to store the champion model")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config/model_config.yaml") -> "ModelConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ModelConfig instance with values from YAML file
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict) if config_dict else cls()
