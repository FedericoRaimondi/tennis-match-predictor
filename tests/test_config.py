"""Tests for Pydantic configuration."""

import pytest
from config.data_config import DataConfig, DataSourceConfig, FeatureConfig
from config.model_config import (
    EstimatorConfig,
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
    assert "p_ace" in config.stats_columns_mean
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
