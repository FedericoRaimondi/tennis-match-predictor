"""Tests for the base model class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from match_predictor.model.base_model import BaseModel


class MockModel(BaseModel):
    """Mock implementation of BaseModel for testing."""

    def fit(self, X, y):
        """Mock fit method."""
        self.model = {"fitted": True, "shape": X.shape}
        return self

    def predict(self, X):
        """Mock predict method."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return np.zeros(len(X))

    def evaluate(self, X, y):
        """Mock evaluate method."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return 0.85

    def save(self, filepath: str):
        """Mock save method."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str):
        """Mock load method."""
        import pickle

        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        return self


def test_base_model_init_without_config():
    """Test BaseModel initialization without config."""
    model = MockModel()
    assert model.config_path is None
    assert model.model is None
    assert model.config == {}


def test_base_model_init_with_config():
    """Test BaseModel initialization with config file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("test_param: 42\n")
        f.write("another_param: test\n")
        config_path = f.name

    try:
        model = MockModel(config_path=config_path)
        assert model.config_path == config_path
        assert model.config["test_param"] == 42
        assert model.config["another_param"] == "test"
    finally:
        Path(config_path).unlink()


def test_base_model_fit():
    """Test model fit method."""
    model = MockModel()
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    model.fit(X, y)
    assert model.model is not None
    assert model.model["fitted"] is True
    assert model.model["shape"] == (3, 2)


def test_base_model_predict():
    """Test model predict method."""
    model = MockModel()
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    # Should raise error before fitting
    with pytest.raises(ValueError, match="Model not fitted"):
        model.predict(X)

    # Should work after fitting
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 3


def test_base_model_evaluate():
    """Test model evaluate method."""
    model = MockModel()
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    # Should raise error before fitting
    with pytest.raises(ValueError, match="Model not fitted"):
        model.evaluate(X, y)

    # Should work after fitting
    model.fit(X, y)
    score = model.evaluate(X, y)
    assert score == 0.85


def test_base_model_save_and_load():
    """Test model save and load methods."""
    model = MockModel()
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    # Fit the model
    model.fit(X, y)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        assert Path(temp_path).exists()

        # Load into new model
        new_model = MockModel()
        new_model.load(temp_path)
        assert new_model.model is not None
        assert new_model.model["fitted"] is True
    finally:
        Path(temp_path).unlink()


def test_base_model_with_numpy_arrays():
    """Test model with numpy arrays instead of DataFrames."""
    model = MockModel()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 3

    score = model.evaluate(X, y)
    assert score == 0.85


def test_base_model_abstract_methods():
    """Test that BaseModel cannot be instantiated directly."""
    # BaseModel is abstract and requires all abstract methods to be implemented
    # This is enforced by ABC
    assert hasattr(BaseModel, "__abstractmethods__")
    assert "fit" in BaseModel.__abstractmethods__
    assert "predict" in BaseModel.__abstractmethods__
    assert "evaluate" in BaseModel.__abstractmethods__
    assert "save" in BaseModel.__abstractmethods__
    assert "load" in BaseModel.__abstractmethods__
