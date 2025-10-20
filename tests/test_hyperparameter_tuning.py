"""Tests for hyperparameter tuning module."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from match_predictor.config import HyperparameterTuningConfig, ModelConfig
from match_predictor.ml_pipeline.hyperparameter_tuning import HyperparameterTuner


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


@pytest.fixture
def quick_config():
    """Create config with quick hyperparameter tuning."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=2, cv_folds=2, timeout=10, random_state=42)
    return config


def test_hyperparameter_tuner_init_default():
    """Test HyperparameterTuner initialization with defaults."""
    tuner = HyperparameterTuner()
    assert tuner.config is not None
    assert tuner.tuning_config is not None
    assert tuner.best_params is None
    assert tuner.study is None


def test_hyperparameter_tuner_init_with_config(quick_config):
    """Test HyperparameterTuner initialization with custom config."""
    tuner = HyperparameterTuner(config=quick_config)
    assert tuner.config == quick_config
    assert tuner.tuning_config.n_trials == 2
    assert tuner.tuning_config.cv_folds == 2


def test_hyperparameter_tuner_objective(quick_config, sample_data):
    """Test objective function."""
    import optuna

    tuner = HyperparameterTuner(config=quick_config)
    X, y = sample_data

    # Create a trial
    study = optuna.create_study()
    trial = study.ask()

    score = tuner.objective(trial, XGBClassifier, X, y)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_hyperparameter_tuner_tune(quick_config, sample_data):
    """Test hyperparameter tuning."""
    tuner = HyperparameterTuner(config=quick_config)
    X, y = sample_data

    best_params = tuner.tune(XGBClassifier, X, y)

    assert tuner.best_params is not None
    assert tuner.study is not None
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert "learning_rate" in best_params
    assert "n_estimators" in best_params
    assert "objective" in best_params
    assert best_params["objective"] == "binary:logistic"


def test_hyperparameter_tuner_get_optimization_history(quick_config, sample_data):
    """Test getting optimization history."""
    tuner = HyperparameterTuner(config=quick_config)
    X, y = sample_data

    # Before tuning
    history = tuner.get_optimization_history()
    assert history == []

    # After tuning
    tuner.tune(XGBClassifier, X, y)
    history = tuner.get_optimization_history()

    assert len(history) == 2  # n_trials = 2
    assert all("trial_number" in trial for trial in history)
    assert all("value" in trial for trial in history)
    assert all("params" in trial for trial in history)
    assert all("state" in trial for trial in history)


def test_hyperparameter_tuner_get_best_trial_info(quick_config, sample_data):
    """Test getting best trial info."""
    tuner = HyperparameterTuner(config=quick_config)
    X, y = sample_data

    # Before tuning
    best_trial = tuner.get_best_trial_info()
    assert best_trial == {}

    # After tuning
    tuner.tune(XGBClassifier, X, y)
    best_trial = tuner.get_best_trial_info()

    assert "trial_number" in best_trial
    assert "value" in best_trial
    assert "params" in best_trial
    assert isinstance(best_trial["value"], float)
    assert isinstance(best_trial["params"], dict)


def test_hyperparameter_tuner_with_different_random_state():
    """Test tuner with different random state."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=2, cv_folds=2, random_state=123)

    tuner = HyperparameterTuner(config=config)
    assert tuner.tuning_config.random_state == 123


def test_hyperparameter_tuner_search_space():
    """Test that objective creates proper search space."""
    import optuna

    # Create config with fewer cv_folds to match small dataset
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(cv_folds=2)
    tuner = HyperparameterTuner(config=config)
    study = optuna.create_study()
    trial = study.ask()

    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 0, 1, 1, 1])

    # Call objective to populate trial params
    score = tuner.objective(trial, XGBClassifier, X, y)

    # Check that all expected parameters are in the trial
    expected_params = [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]

    for param in expected_params:
        assert param in trial.params


def test_hyperparameter_tuner_parameter_ranges():
    """Test that parameters are within expected ranges."""
    import optuna

    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=1, cv_folds=2)
    tuner = HyperparameterTuner(config=config)

    X = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
    y = pd.Series(np.random.randint(0, 2, 50))

    tuner.tune(XGBClassifier, X, y)

    params = tuner.best_params

    # Check parameter ranges
    assert 2 <= params["max_depth"] <= 10
    assert 0.001 <= params["learning_rate"] <= 0.3
    assert 100 <= params["n_estimators"] <= 2000
    assert 1 <= params["min_child_weight"] <= 10
    assert 0.5 <= params["subsample"] <= 1.0
    assert 0.3 <= params["colsample_bytree"] <= 1.0
    assert 0.0 <= params["gamma"] <= 5.0
    assert 0.0 <= params["reg_alpha"] <= 1.0
    assert 0.0 <= params["reg_lambda"] <= 1.0


def test_hyperparameter_tuner_with_timeout():
    """Test hyperparameter tuning with timeout."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=1000, timeout=1)  # 1 second timeout

    tuner = HyperparameterTuner(config=config)
    X = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
    y = pd.Series(np.random.randint(0, 2, 50))

    best_params = tuner.tune(XGBClassifier, X, y)

    # Should stop before completing all 1000 trials due to timeout
    assert len(tuner.study.trials) < 1000
    assert best_params is not None


def test_hyperparameter_tuner_base_parameters():
    """Test that base parameters are correctly added."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=1, cv_folds=2, random_state=42)

    tuner = HyperparameterTuner(config=config)
    X = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
    y = pd.Series(np.random.randint(0, 2, 50))

    best_params = tuner.tune(XGBClassifier, X, y)

    # Check base parameters
    assert best_params["objective"] == "binary:logistic"
    assert best_params["eval_metric"] == "logloss"
    assert best_params["enable_categorical"] is True
    assert best_params["seed"] == 42


def test_hyperparameter_tuner_cv_folds():
    """Test hyperparameter tuning with different cv_folds."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=1, cv_folds=3)

    tuner = HyperparameterTuner(config=config)
    assert tuner.tuning_config.cv_folds == 3


def test_hyperparameter_tuner_none_timeout():
    """Test hyperparameter tuning with None timeout."""
    config = ModelConfig()
    config.hyperparameter_tuning = HyperparameterTuningConfig(n_trials=1, timeout=None)

    tuner = HyperparameterTuner(config=config)
    X = pd.DataFrame({"feature1": np.random.randn(30), "feature2": np.random.randn(30)})
    y = pd.Series(np.random.randint(0, 2, 30))

    best_params = tuner.tune(XGBClassifier, X, y)
    assert best_params is not None
