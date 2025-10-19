"""Tests for model training module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from match_predictor.config import ModelConfig
from match_predictor.ml_pipeline.training import ModelTrainer


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    data = {
        "tourney_id": ["T1"] * 100,
        "tourney_date": pd.date_range("2021-01-01", periods=100),
        "match_num": list(range(100)),
        "player_id": np.random.randint(1, 20, 100),
        "opponent_id": np.random.randint(1, 20, 100),
        "player_rank": np.random.randint(1, 100, 100),
        "opponent_rank": np.random.randint(1, 100, 100),
        "results": np.random.randint(0, 2, 100),
        "p_ace": np.random.rand(100),
        "p_df": np.random.rand(100),
        "o_ace": np.random.rand(100),
        "o_df": np.random.rand(100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def quick_config():
    """Create config for quick training."""
    config = ModelConfig()
    config.estimator.params["n_estimators"] = 10
    config.estimator.params["max_depth"] = 2
    config.training.test_size = 0.3
    config.training.validation_size = 0.3
    return config


def test_model_trainer_init_default():
    """Test ModelTrainer initialization with defaults."""
    trainer = ModelTrainer()
    assert trainer.config is not None
    assert trainer.model is None
    assert trainer.feature_names is None
    assert trainer.training_metrics == {}


def test_model_trainer_init_with_config(quick_config):
    """Test ModelTrainer initialization with custom config."""
    trainer = ModelTrainer(config=quick_config)
    assert trainer.config == quick_config
    assert trainer.config.estimator.params["n_estimators"] == 10


def test_model_trainer_load_estimator_class(quick_config):
    """Test loading estimator class from config."""
    trainer = ModelTrainer(config=quick_config)
    estimator_class = trainer.load_estimator_class()

    from xgboost import XGBClassifier

    assert estimator_class == XGBClassifier


def test_model_trainer_load_estimator_class_invalid():
    """Test loading invalid estimator class raises error."""
    config = ModelConfig()
    config.estimator.module = "invalid_module"
    config.estimator.class_name = "InvalidClass"

    trainer = ModelTrainer(config=config)

    with pytest.raises(Exception):
        trainer.load_estimator_class()


@patch("match_predictor.ml_pipeline.training.FeatureEngineer")
def test_model_trainer_train_without_tuning(mock_feature_engineer, quick_config, sample_training_data):
    """Test model training without hyperparameter tuning."""
    # Mock feature engineer
    mock_fe_instance = MagicMock()
    mock_fe_instance.engineer_features.return_value = sample_training_data
    X = sample_training_data.drop(columns=["results"])
    y = sample_training_data["results"]
    mock_fe_instance.prepare_features_for_training.return_value = (X, y)
    mock_feature_engineer.return_value = mock_fe_instance

    trainer = ModelTrainer(config=quick_config)
    metrics = trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=False)

    assert trainer.model is not None
    assert "val_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert "train_size" in metrics
    assert "val_size" in metrics
    assert "test_size" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


@patch("match_predictor.ml_pipeline.training.FeatureEngineer")
@patch("match_predictor.ml_pipeline.training.HyperparameterTuner")
def test_model_trainer_train_with_tuning(
    mock_tuner, mock_feature_engineer, quick_config, sample_training_data
):
    """Test model training with hyperparameter tuning."""
    # Mock feature engineer
    mock_fe_instance = MagicMock()
    mock_fe_instance.engineer_features.return_value = sample_training_data
    X = sample_training_data.drop(columns=["results"])
    y = sample_training_data["results"]
    mock_fe_instance.prepare_features_for_training.return_value = (X, y)
    mock_feature_engineer.return_value = mock_fe_instance

    # Mock tuner
    mock_tuner_instance = MagicMock()
    mock_tuner_instance.tune.return_value = quick_config.estimator.params
    mock_tuner.return_value = mock_tuner_instance

    trainer = ModelTrainer(config=quick_config)
    metrics = trainer.train(sample_training_data, tune_hyperparameters=True, log_to_mlflow=False)

    assert trainer.model is not None
    assert mock_tuner_instance.tune.called
    assert "val_accuracy" in metrics
    assert "test_accuracy" in metrics


def test_model_trainer_save_model(quick_config, sample_training_data):
    """Test saving trained model."""
    with patch("match_predictor.ml_pipeline.training.FeatureEngineer") as mock_fe:
        # Mock feature engineer
        mock_fe_instance = MagicMock()
        mock_fe_instance.engineer_features.return_value = sample_training_data
        X = sample_training_data.drop(columns=["results"])
        y = sample_training_data["results"]
        mock_fe_instance.prepare_features_for_training.return_value = (X, y)
        mock_fe.return_value = mock_fe_instance

        trainer = ModelTrainer(config=quick_config)
        trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model_path)

            assert model_path.exists()


def test_model_trainer_save_model_without_training(quick_config):
    """Test saving model without training raises error."""
    trainer = ModelTrainer(config=quick_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"

        with pytest.raises(ValueError, match="No trained model to save"):
            trainer.save_model(model_path)


def test_model_trainer_load_model(quick_config, sample_training_data):
    """Test loading trained model."""
    with patch("match_predictor.ml_pipeline.training.FeatureEngineer") as mock_fe:
        # Mock feature engineer
        mock_fe_instance = MagicMock()
        mock_fe_instance.engineer_features.return_value = sample_training_data
        X = sample_training_data.drop(columns=["results"])
        y = sample_training_data["results"]
        mock_fe_instance.prepare_features_for_training.return_value = (X, y)
        mock_fe.return_value = mock_fe_instance

        # Train and save model
        trainer = ModelTrainer(config=quick_config)
        trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model_path)

            # Load into new trainer
            new_trainer = ModelTrainer(config=quick_config)
            new_trainer.load_model(model_path)

            assert new_trainer.model is not None
            assert new_trainer.feature_names is not None


def test_model_trainer_load_model_file_not_found(quick_config):
    """Test loading model from non-existent file raises error."""
    trainer = ModelTrainer(config=quick_config)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        trainer.load_model("nonexistent_model.pkl")


def test_model_trainer_should_promote_to_champion_no_champion(quick_config):
    """Test promotion decision when no champion exists."""
    trainer = ModelTrainer(config=quick_config)
    trainer.training_metrics = {"test_accuracy": 0.70}

    should_promote = trainer.should_promote_to_champion(champion_accuracy=None)
    assert should_promote is True


def test_model_trainer_should_promote_to_champion_better_accuracy(quick_config):
    """Test promotion decision when new model is better."""
    trainer = ModelTrainer(config=quick_config)
    trainer.training_metrics = {"test_accuracy": 0.75}

    should_promote = trainer.should_promote_to_champion(champion_accuracy=0.70)
    assert should_promote is True


def test_model_trainer_should_promote_to_champion_worse_accuracy(quick_config):
    """Test promotion decision when new model is worse."""
    trainer = ModelTrainer(config=quick_config)
    trainer.training_metrics = {"test_accuracy": 0.65}

    should_promote = trainer.should_promote_to_champion(champion_accuracy=0.70)
    assert should_promote is False


def test_model_trainer_should_promote_to_champion_below_threshold(quick_config):
    """Test promotion decision when below minimum threshold."""
    trainer = ModelTrainer(config=quick_config)
    trainer.training_metrics = {"test_accuracy": 0.55}
    trainer.config.training.min_accuracy_threshold = 0.60

    should_promote = trainer.should_promote_to_champion(champion_accuracy=None)
    assert should_promote is False


def test_model_trainer_should_promote_to_champion_no_metrics(quick_config):
    """Test promotion decision when no metrics available."""
    trainer = ModelTrainer(config=quick_config)
    trainer.training_metrics = {}

    should_promote = trainer.should_promote_to_champion(champion_accuracy=0.70)
    assert should_promote is False


@patch("match_predictor.ml_pipeline.training.mlflow")
@patch("match_predictor.ml_pipeline.training.FeatureEngineer")
def test_model_trainer_log_to_mlflow(mock_feature_engineer, mock_mlflow, quick_config, sample_training_data):
    """Test logging to MLflow."""
    # Mock feature engineer
    mock_fe_instance = MagicMock()
    mock_fe_instance.engineer_features.return_value = sample_training_data
    X = sample_training_data.drop(columns=["results"])
    y = sample_training_data["results"]
    mock_fe_instance.prepare_features_for_training.return_value = (X, y)
    mock_feature_engineer.return_value = mock_fe_instance

    trainer = ModelTrainer(config=quick_config)
    trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=True)

    # Verify MLflow methods were called
    assert mock_mlflow.set_tracking_uri.called
    assert mock_mlflow.set_experiment.called
    assert mock_mlflow.start_run.called


def test_model_trainer_feature_names_stored(quick_config, sample_training_data):
    """Test that feature names are stored after training."""
    with patch("match_predictor.ml_pipeline.training.FeatureEngineer") as mock_fe:
        # Mock feature engineer
        mock_fe_instance = MagicMock()
        mock_fe_instance.engineer_features.return_value = sample_training_data
        X = sample_training_data.drop(columns=["results"])
        y = sample_training_data["results"]
        mock_fe_instance.prepare_features_for_training.return_value = (X, y)
        mock_fe.return_value = mock_fe_instance

        trainer = ModelTrainer(config=quick_config)
        trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=False)

        assert trainer.feature_names is not None
        assert len(trainer.feature_names) > 0
