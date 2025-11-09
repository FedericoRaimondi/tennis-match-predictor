"""Integration test for feature-aware monitoring system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from match_predictor.config import DataConfig, ModelConfig
from match_predictor.ml_pipeline.drift_detection import FeatureAwareDriftDetector
from match_predictor.ml_pipeline.monitoring import ModelMonitor
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


def test_feature_aware_monitoring_integration(sample_training_data):
    """Test complete feature-aware monitoring workflow."""
    # 1. Train a model
    config = ModelConfig()
    config.estimator.params["n_estimators"] = 10
    config.estimator.params["max_depth"] = 2
    config.training.test_size = 0.3
    config.training.validation_size = 0.3

    with patch("match_predictor.ml_pipeline.training.FeatureEngineer") as mock_fe:
        # Mock feature engineer
        mock_fe_instance = MagicMock()
        X = sample_training_data.drop(
            columns=["results", "tourney_id", "tourney_date", "match_num", "player_id", "opponent_id"]
        )
        y = sample_training_data["results"]
        mock_fe_instance.prepare_features_for_training.return_value = (X, y)
        mock_fe.return_value = mock_fe_instance

        trainer = ModelTrainer(config=config)
        trainer.train(sample_training_data, tune_hyperparameters=False, log_to_mlflow=False)

    # 2. Extract model and features
    trained_model = trainer.model
    feature_names = trainer.feature_names

    assert trained_model is not None
    assert feature_names is not None
    assert len(feature_names) > 0

    # 3. Create drift detector with trained model
    detector = FeatureAwareDriftDetector(model=trained_model, feature_names=feature_names)

    # Verify features were extracted
    assert detector.feature_names == feature_names

    # 4. Create reference and current data
    np.random.seed(42)
    reference_data = pd.DataFrame({f: np.random.randn(100) for f in feature_names})

    np.random.seed(123)
    current_data = pd.DataFrame({f: np.random.randn(50) for f in feature_names})

    # 5. Test drift detection
    detector.set_reference_data(reference_data)
    detector.set_current_data(current_data)

    drift_results = detector.detect_drift()

    # Verify drift results
    assert "drift_detected" in drift_results
    assert "drift_share" in drift_results
    assert "drifted_features" in drift_results
    assert "total_features" in drift_results
    assert drift_results["total_features"] == len(feature_names)

    # 6. Test monitoring with feature-aware detector
    monitor = ModelMonitor(model=trained_model, feature_names=feature_names)
    monitor.set_reference_data(reference_data.assign(winner=np.random.randint(0, 2, 100)))
    monitor.set_current_data(current_data.assign(winner=np.random.randint(0, 2, 50)))

    drift_results = monitor.check_data_drift()

    assert "drift_detected" in drift_results
    assert "feature_drift_details" in drift_results
    assert len(drift_results["feature_drift_details"]) == len(feature_names)

    # 7. Generate monitoring report
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "monitoring_report.html"
        monitor.generate_monitoring_report(output_path=report_path)

        assert report_path.exists()
        # Verify report contains feature details
        with open(report_path, "r") as f:
            content = f.read()
            assert "Feature-Level Drift Details" in content
            assert "DRIFT" in content or "OK" in content


def test_monitoring_only_uses_model_features():
    """Test that monitoring only uses features from the model."""
    # Create model with specific features
    feature_names = ["feat1", "feat2", "feat3"]

    class MockModel:
        def __init__(self):
            self.feature_names_in_ = np.array(feature_names)

    model = MockModel()

    # Create data with extra features
    np.random.seed(42)
    reference_data = pd.DataFrame(
        {
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "feat3": np.random.randn(50),
            "extra_feat": np.random.randn(50),  # Not in model
            "another_extra": np.random.randn(50),  # Not in model
        }
    )

    current_data = pd.DataFrame(
        {
            "feat1": np.random.randn(30),
            "feat2": np.random.randn(30),
            "feat3": np.random.randn(30),
            "extra_feat": np.random.randn(30),
            "another_extra": np.random.randn(30),
        }
    )

    # Initialize monitor with model
    monitor = ModelMonitor(model=model)
    monitor.set_reference_data(reference_data)
    monitor.set_current_data(current_data)

    # Verify that drift detector only uses model features
    assert list(monitor.drift_detector.reference_data.columns) == feature_names
    assert list(monitor.drift_detector.current_data.columns) == feature_names

    # Run drift detection
    drift_results = monitor.check_data_drift()

    # Should only analyze model features
    assert drift_results["total_features"] == 3
    assert set(drift_results["feature_drift_details"].keys()) == set(feature_names)


def test_drift_detection_with_categorical_and_numerical():
    """Test drift detection correctly handles mixed feature types."""
    np.random.seed(42)

    # Create mixed data
    reference_data = pd.DataFrame(
        {
            "numerical1": np.random.randn(100),
            "numerical2": np.random.randn(100),
            "categorical1": np.random.choice(["A", "B", "C"], 100),
            "categorical2": np.random.randint(0, 3, 100),  # Few unique values = categorical
        }
    )

    current_data = pd.DataFrame(
        {
            "numerical1": np.random.randn(50),
            "numerical2": np.random.randn(50),
            "categorical1": np.random.choice(["A", "B", "C"], 50),
            "categorical2": np.random.randint(0, 3, 50),
        }
    )

    detector = FeatureAwareDriftDetector()
    detector.set_reference_data(reference_data)
    detector.set_current_data(current_data)

    drift_results = detector.detect_drift()

    details = drift_results["feature_drift_details"]

    # Check that numerical features use KS test
    assert details["numerical1"]["type"] == "numerical"
    assert details["numerical1"]["test"] == "ks"
    assert details["numerical2"]["type"] == "numerical"
    assert details["numerical2"]["test"] == "ks"

    # Check that categorical features use JS divergence
    assert details["categorical1"]["type"] == "categorical"
    assert details["categorical1"]["test"] == "js"
    assert details["categorical2"]["type"] == "categorical"
    assert details["categorical2"]["test"] == "js"


def test_performance_degradation_triggers_retraining():
    """Test that performance degradation triggers retraining recommendation."""
    monitor = ModelMonitor()

    # Simulate poor performance
    predictions = pd.Series([0] * 100)  # All zeros
    actual = pd.Series([1] * 100)  # All ones

    performance_results = monitor.check_model_performance(predictions, actual)

    # Should detect degradation
    assert performance_results["performance_degradation"] is True
    assert performance_results["accuracy"] == 0.0

    # Should trigger retraining
    should_retrain, reason = monitor.should_trigger_retraining(performance_results=performance_results)
    assert should_retrain is True
    assert "performance degradation" in reason.lower()


def test_high_drift_share_triggers_retraining():
    """Test that high drift share triggers retraining recommendation."""
    monitor = ModelMonitor()

    # Simulate high drift
    drift_results = {
        "drift_detected": True,
        "drift_share": 0.5,  # 50% drift
        "requires_retraining": True,
    }

    should_retrain, reason = monitor.should_trigger_retraining(drift_results=drift_results)

    assert should_retrain is True
    assert "data drift" in reason.lower()
