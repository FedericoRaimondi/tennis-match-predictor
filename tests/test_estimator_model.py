"""Tests for the estimator model class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from match_predictor.model.estimator_model import EstimatorModel


@pytest.fixture
def sample_config_file():
    """Create a temporary config file for testing."""
    config_content = """
estimator:
  module: sklearn.ensemble
  class_name: RandomForestClassifier
  params:
    n_estimators: 10
    max_depth: 3
    random_state: 42
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    yield config_path
    Path(config_path).unlink()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 0, 1, 1, 1])
    return X, y


def test_estimator_model_init_with_config(sample_config_file):
    """Test EstimatorModel initialization with config file."""
    model = EstimatorModel(config_path=sample_config_file)
    assert model.model is not None
    assert hasattr(model.model, "fit")
    assert hasattr(model.model, "predict")


def test_estimator_model_init_without_module_raises_error():
    """Test EstimatorModel raises error when module is not specified."""
    config_content = """
estimator:
  class_name: RandomForestClassifier
  params:
    n_estimators: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        with pytest.raises(ValueError, match="Estimator 'module' and 'class_name' must be specified"):
            EstimatorModel(config_path=config_path)
    finally:
        Path(config_path).unlink()


def test_estimator_model_init_without_class_name_raises_error():
    """Test EstimatorModel raises error when class_name is not specified."""
    config_content = """
estimator:
  module: sklearn.ensemble
  params:
    n_estimators: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        with pytest.raises(ValueError, match="Estimator 'module' and 'class_name' must be specified"):
            EstimatorModel(config_path=config_path)
    finally:
        Path(config_path).unlink()


def test_estimator_model_fit(sample_config_file, sample_data):
    """Test model fit method."""
    model = EstimatorModel(config_path=sample_config_file)
    X, y = sample_data

    model.fit(X, y)
    # Check that model has been fitted
    assert hasattr(model.model, "n_features_in_")


def test_estimator_model_predict(sample_config_file, sample_data):
    """Test model predict method."""
    model = EstimatorModel(config_path=sample_config_file)
    X, y = sample_data

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_estimator_model_evaluate(sample_config_file, sample_data):
    """Test model evaluate method."""
    model = EstimatorModel(config_path=sample_config_file)
    X, y = sample_data

    model.fit(X, y)
    score = model.evaluate(X, y)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_estimator_model_save_and_load(sample_config_file, sample_data):
    """Test model save and load methods."""
    model = EstimatorModel(config_path=sample_config_file)
    X, y = sample_data

    model.fit(X, y)
    original_predictions = model.predict(X)

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        assert Path(temp_path).exists()

        # Load into new model
        new_model = EstimatorModel(config_path=sample_config_file)
        new_model.load(temp_path)

        # Verify loaded model makes same predictions
        new_predictions = new_model.predict(X)
        assert np.array_equal(original_predictions, new_predictions)
    finally:
        Path(temp_path).unlink()


def test_estimator_model_with_numpy_arrays(sample_config_file):
    """Test model with numpy arrays."""
    model = EstimatorModel(config_path=sample_config_file)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_estimator_model_with_xgboost():
    """Test EstimatorModel with XGBoost classifier."""
    config_content = """
estimator:
  module: xgboost
  class_name: XGBClassifier
  params:
    n_estimators: 10
    max_depth: 2
    random_state: 42
    objective: binary:logistic
    eval_metric: logloss
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        model = EstimatorModel(config_path=config_path)
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 0, 1, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)
        score = model.evaluate(X, y)

        assert len(predictions) == len(y)
        assert isinstance(score, float)
    finally:
        Path(config_path).unlink()


def test_estimator_model_with_logistic_regression():
    """Test EstimatorModel with LogisticRegression."""
    config_content = """
estimator:
  module: sklearn.linear_model
  class_name: LogisticRegression
  params:
    random_state: 42
    max_iter: 100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        model = EstimatorModel(config_path=config_path)
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 0, 1, 1, 1])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
    finally:
        Path(config_path).unlink()


def test_estimator_model_evaluate_without_score_method():
    """Test evaluate raises error for models without score method."""
    # Create a mock estimator without score method
    config_content = """
estimator:
  module: sklearn.ensemble
  class_name: RandomForestClassifier
  params:
    n_estimators: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        model = EstimatorModel(config_path=config_path)
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])

        model.fit(X, y)

        # Remove the score method to simulate a model without it
        delattr(model.model, "score")

        with pytest.raises(NotImplementedError, match="does not implement a 'score' method"):
            model.evaluate(X, y)
    finally:
        Path(config_path).unlink()


def test_estimator_model_with_empty_params():
    """Test EstimatorModel with empty params."""
    config_content = """
estimator:
  module: sklearn.tree
  class_name: DecisionTreeClassifier
  params: {}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        model = EstimatorModel(config_path=config_path)
        assert model.model is not None
    finally:
        Path(config_path).unlink()
