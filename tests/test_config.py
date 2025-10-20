"""Tests for Pydantic configuration."""

import pytest
from match_predictor.config import (
    DataConfig,
    DataSourceConfig,
    EstimatorConfig,
    FeatureConfig,
    HyperparameterTuningConfig,
    ModelConfig,
    TrainingConfig,
)


def test_data_source_config_defaults():
    """Test DataSourceConfig with default values."""
    config = DataSourceConfig()
    assert config.github_repo == "JeffSackmann/tennis_atp"
    assert config.selected_year == 1991
    assert config.tourney_levels == ["G", "F", "M", "A"]


def test_feature_config_defaults():
    """Test FeatureConfig with default values."""
    config = FeatureConfig()
    assert config.rolling_windows == [3, 5, 10]
    assert len(config.stats_columns_mean) > 0
    assert "ace" in config.stats_columns_mean
    assert config.elo_k_factor == 32.0


def test_data_config_initialization():
    """Test DataConfig initialization."""
    config = DataConfig()
    assert isinstance(config.source, DataSourceConfig)
    assert isinstance(config.features, FeatureConfig)
    assert config.inference_data_path == "data/"


def test_data_config_custom_values():
    """Test DataConfig with custom values."""
    config = DataConfig(
        source=DataSourceConfig(selected_year=2000),
        inference_data_path="custom_data/"
    )
    assert config.source.selected_year == 2000
    assert config.inference_data_path == "custom_data/"


def test_estimator_config_defaults():
    """Test EstimatorConfig with default values."""
    config = EstimatorConfig()
    assert config.module == "xgboost"
    assert config.class_name == "XGBClassifier"
    assert "objective" in config.params
    assert config.params["objective"] == "binary:logistic"


def test_hyperparameter_tuning_config():
    """Test HyperparameterTuningConfig."""
    config = HyperparameterTuningConfig()
    assert config.n_trials == 50
    assert config.cv_folds == 5
    assert config.random_state == 42


def test_training_config():
    """Test TrainingConfig."""
    config = TrainingConfig()
    assert config.test_size == 0.2
    assert config.validation_size == 0.2
    assert config.min_accuracy_threshold == 0.60


def test_model_config_initialization():
    """Test ModelConfig initialization."""
    config = ModelConfig()
    assert isinstance(config.estimator, EstimatorConfig)
    assert isinstance(config.hyperparameter_tuning, HyperparameterTuningConfig)
    assert isinstance(config.training, TrainingConfig)
    assert config.model_name == "atp_match_predictor"


def test_model_config_custom_values():
    """Test ModelConfig with custom values."""
    config = ModelConfig(
        model_name="custom_model",
        estimator=EstimatorConfig(module="sklearn.ensemble", class_name="RandomForestClassifier")
    )
    assert config.model_name == "custom_model"
    assert config.estimator.class_name == "RandomForestClassifier"


def test_data_config_from_yaml():
    """Test DataConfig loading from YAML file."""
    # Test with default path
    config = DataConfig.from_yaml("config/data_config.yaml")
    assert config.source.github_repo == "JeffSackmann/tennis_atp"
    assert config.source.selected_year == 1991
    assert config.features.rolling_windows == [3, 5, 10]


def test_model_config_from_yaml():
    """Test ModelConfig loading from YAML file."""
    # Test with default path
    config = ModelConfig.from_yaml("config/model_config.yaml")
    assert config.model_name == "atp_match_predictor"
    assert config.estimator.module == "xgboost"
    assert config.estimator.class_name == "XGBClassifier"
    assert config.training.min_accuracy_threshold == 0.60


def test_config_from_yaml_nonexistent():
    """Test config loading with nonexistent file returns defaults."""
    config = DataConfig.from_yaml("nonexistent_file.yaml")
    assert config.source.github_repo == "JeffSackmann/tennis_atp"
    assert config.features.elo_k_factor == 32.0


def test_feature_config_elo_initial_rating():
    """Test FeatureConfig elo_initial_rating."""
    config = FeatureConfig()
    assert config.elo_initial_rating == 1500.0


def test_feature_config_stats_columns_sum():
    """Test FeatureConfig stats_columns_sum."""
    config = FeatureConfig()
    assert "minutes" in config.stats_columns_sum
    assert "results" in config.stats_columns_sum


def test_data_config_matches_results_file():
    """Test DataConfig matches_results_file default."""
    config = DataConfig()
    assert config.matches_results_file == "matches_results.csv"


def test_data_config_tournament_info_file():
    """Test DataConfig tournament_info_file default."""
    config = DataConfig()
    assert config.tournament_info_file == "tournament_info.csv"


def test_estimator_config_custom_params():
    """Test EstimatorConfig with custom parameters."""
    config = EstimatorConfig(
        module="sklearn.tree",
        class_name="DecisionTreeClassifier",
        params={"max_depth": 5, "random_state": 123},
    )
    assert config.module == "sklearn.tree"
    assert config.class_name == "DecisionTreeClassifier"
    assert config.params["max_depth"] == 5
    assert config.params["random_state"] == 123


def test_hyperparameter_tuning_config_timeout():
    """Test HyperparameterTuningConfig timeout."""
    config = HyperparameterTuningConfig()
    assert config.timeout == 3600


def test_training_config_random_state():
    """Test TrainingConfig random_state."""
    config = TrainingConfig()
    assert config.random_state == 42


def test_mlflow_config_defaults():
    """Test MLflowConfig with default values."""
    from match_predictor.config import MLflowConfig

    config = MLflowConfig()
    assert config.experiment_name == "tennis-match-predictor"
    assert config.tracking_uri == "file:./mlruns"
    assert config.model_name == "tennis_predictor_model"


def test_mlflow_config_custom_values():
    """Test MLflowConfig with custom values."""
    from match_predictor.config import MLflowConfig

    config = MLflowConfig(
        experiment_name="custom_experiment", tracking_uri="http://localhost:5000", model_name="custom_model"
    )
    assert config.experiment_name == "custom_experiment"
    assert config.tracking_uri == "http://localhost:5000"
    assert config.model_name == "custom_model"


def test_model_config_mlflow():
    """Test ModelConfig includes MLflowConfig."""
    config = ModelConfig()
    assert hasattr(config, "mlflow")
    assert config.mlflow.experiment_name == "tennis-match-predictor"


def test_model_config_champion_model_path():
    """Test ModelConfig champion_model_path."""
    config = ModelConfig()
    assert config.champion_model_path == "models/"


def test_data_config_from_yaml_empty_file(tmp_path):
    """Test DataConfig loading from empty YAML file."""
    import yaml

    empty_yaml = tmp_path / "empty.yaml"
    with open(empty_yaml, "w") as f:
        yaml.dump(None, f)

    config = DataConfig.from_yaml(empty_yaml)
    # Should return default config for empty file
    assert config.source.github_repo == "JeffSackmann/tennis_atp"


def test_model_config_from_yaml_empty_file(tmp_path):
    """Test ModelConfig loading from empty YAML file."""
    import yaml

    empty_yaml = tmp_path / "empty.yaml"
    with open(empty_yaml, "w") as f:
        yaml.dump(None, f)

    config = ModelConfig.from_yaml(empty_yaml)
    # Should return default config for empty file
    assert config.model_name == "atp_match_predictor"


def test_data_source_config_custom_tourney_levels():
    """Test DataSourceConfig with custom tourney levels."""
    config = DataSourceConfig(tourney_levels=["G", "M"])
    assert config.tourney_levels == ["G", "M"]


def test_feature_config_custom_rolling_windows():
    """Test FeatureConfig with custom rolling windows."""
    config = FeatureConfig(rolling_windows=[5, 10, 15])
    assert config.rolling_windows == [5, 10, 15]


def test_hyperparameter_tuning_config_none_timeout():
    """Test HyperparameterTuningConfig with None timeout."""
    config = HyperparameterTuningConfig(timeout=None)
    assert config.timeout is None


def test_training_config_custom_test_size():
    """Test TrainingConfig with custom test_size."""
    config = TrainingConfig(test_size=0.3, validation_size=0.15)
    assert config.test_size == 0.3
    assert config.validation_size == 0.15


def test_data_source_config_fields():
    """Test DataSourceConfig all fields."""
    config = DataSourceConfig(
        github_repo="test/repo",
        selected_year=2000,
        tourney_levels=["G"]
    )
    assert config.github_repo == "test/repo"
    assert config.selected_year == 2000
    assert config.tourney_levels == ["G"]


def test_feature_config_all_stats_columns():
    """Test FeatureConfig stats columns."""
    config = FeatureConfig()
    assert len(config.stats_columns_mean) > 10
    assert "ace" in config.stats_columns_mean
    assert "minutes" in config.stats_columns_sum
    assert "results" in config.stats_columns_sum


def test_estimator_config_all_params():
    """Test EstimatorConfig default params."""
    config = EstimatorConfig()
    assert "objective" in config.params
    assert "eval_metric" in config.params
    assert "enable_categorical" in config.params
    assert "seed" in config.params
    assert "n_estimators" in config.params
    assert "max_depth" in config.params
    assert "learning_rate" in config.params
    assert "colsample_bytree" in config.params


def test_data_config_all_fields():
    """Test DataConfig all fields."""
    config = DataConfig(
        source=DataSourceConfig(github_repo="custom/repo"),
        features=FeatureConfig(elo_k_factor=40.0),
        inference_data_path="custom_data/",
        matches_results_file="custom_matches.pkl",
        tournament_info_file="custom_tourney.pkl"
    )
    assert config.source.github_repo == "custom/repo"
    assert config.features.elo_k_factor == 40.0
    assert config.inference_data_path == "custom_data/"
    assert config.matches_results_file == "custom_matches.pkl"
    assert config.tournament_info_file == "custom_tourney.pkl"
