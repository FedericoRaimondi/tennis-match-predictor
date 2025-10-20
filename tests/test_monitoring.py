"""Tests for model monitoring module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from match_predictor.ml_pipeline.monitoring import ModelMonitor


@pytest.fixture
def sample_reference_data():
    """Create sample reference data."""
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randint(0, 5, 100),
        "winner": np.random.randint(0, 2, 100),
        "prediction": np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data():
    """Create sample current data (similar distribution)."""
    np.random.seed(123)
    data = {
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "feature3": np.random.randint(0, 5, 50),
        "winner": np.random.randint(0, 2, 50),
        "prediction": np.random.randint(0, 2, 50),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_drifted_data():
    """Create sample data with drift."""
    np.random.seed(456)
    data = {
        "feature1": np.random.randn(50) + 5,  # Shifted distribution
        "feature2": np.random.randn(50) * 3,  # Different variance
        "feature3": np.random.randint(5, 10, 50),  # Different range
        "winner": np.random.randint(0, 2, 50),
        "prediction": np.random.randint(0, 2, 50),
    }
    return pd.DataFrame(data)


def test_model_monitor_init():
    """Test ModelMonitor initialization."""
    monitor = ModelMonitor()
    assert monitor.reference_data is None
    assert monitor.current_data is None
    assert monitor.target_column is None
    assert monitor.prediction_column is None


def test_model_monitor_set_reference_data(sample_reference_data):
    """Test setting reference data."""
    monitor = ModelMonitor()
    monitor.set_reference_data(sample_reference_data, target_column="winner", prediction_column="prediction")

    assert monitor.reference_data is not None
    assert len(monitor.reference_data) == 100
    assert monitor.target_column == "winner"
    assert monitor.prediction_column == "prediction"


def test_model_monitor_set_current_data(sample_current_data):
    """Test setting current data."""
    monitor = ModelMonitor()
    monitor.set_current_data(sample_current_data)

    assert monitor.current_data is not None
    assert len(monitor.current_data) == 50


def test_model_monitor_check_data_drift_no_data():
    """Test data drift check without setting data raises error."""
    monitor = ModelMonitor()

    with pytest.raises(ValueError, match="Both reference and current data must be set"):
        monitor.check_data_drift()


def test_model_monitor_check_data_drift(sample_reference_data, sample_current_data):
    """Test data drift check."""
    monitor = ModelMonitor()
    monitor.set_reference_data(sample_reference_data)
    monitor.set_current_data(sample_current_data)

    drift_results = monitor.check_data_drift()

    assert "drift_detected" in drift_results
    assert "drift_share" in drift_results
    assert "drifted_features" in drift_results
    assert "requires_retraining" in drift_results
    assert isinstance(drift_results["drift_detected"], bool)


def test_model_monitor_check_data_quality_no_data():
    """Test data quality check without setting data raises error."""
    monitor = ModelMonitor()

    with pytest.raises(ValueError, match="Current data must be set"):
        monitor.check_data_quality()


def test_model_monitor_check_data_quality(sample_reference_data, sample_current_data):
    """Test data quality check."""
    import numpy as np
    
    monitor = ModelMonitor()
    monitor.set_reference_data(sample_reference_data)
    monitor.set_current_data(sample_current_data)

    quality_results = monitor.check_data_quality()

    assert "missing_values" in quality_results
    assert "data_quality_score" in quality_results
    assert isinstance(quality_results["missing_values"], (int, float, np.integer))
    assert 0.0 <= quality_results["data_quality_score"] <= 1.0


def test_model_monitor_check_data_quality_with_missing_values():
    """Test data quality check with missing values."""
    monitor = ModelMonitor()

    # Create data with missing values
    data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feature2": [np.nan, 2.0, 3.0, np.nan, 5.0],
            "winner": [0, 1, 0, 1, 0],
        }
    )

    monitor.set_reference_data(data.copy())
    monitor.set_current_data(data.copy())

    quality_results = monitor.check_data_quality()

    assert quality_results["missing_values"] > 0


def test_model_monitor_check_model_performance():
    """Test model performance check."""
    monitor = ModelMonitor()

    predictions = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    actual = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])

    performance_results = monitor.check_model_performance(predictions, actual)

    assert "accuracy" in performance_results
    assert "precision" in performance_results
    assert "recall" in performance_results
    assert "performance_degradation" in performance_results
    assert 0.0 <= performance_results["accuracy"] <= 1.0
    assert isinstance(performance_results["performance_degradation"], bool)


def test_model_monitor_check_model_performance_perfect():
    """Test model performance with perfect predictions."""
    monitor = ModelMonitor()

    predictions = pd.Series([0, 1, 0, 1, 0, 1])
    actual = pd.Series([0, 1, 0, 1, 0, 1])

    performance_results = monitor.check_model_performance(predictions, actual)

    assert performance_results["accuracy"] == 1.0
    assert performance_results["precision"] == 1.0
    assert performance_results["recall"] == 1.0
    assert performance_results["performance_degradation"] is False


def test_model_monitor_check_model_performance_poor():
    """Test model performance with poor predictions."""
    monitor = ModelMonitor()

    predictions = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
    actual = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])

    performance_results = monitor.check_model_performance(predictions, actual)

    assert performance_results["accuracy"] == 0.0
    assert performance_results["performance_degradation"] is True


def test_model_monitor_generate_monitoring_report_no_data():
    """Test generating report without setting data raises error."""
    monitor = ModelMonitor()

    with pytest.raises(ValueError, match="Both reference and current data must be set"):
        monitor.generate_monitoring_report()


def test_model_monitor_generate_monitoring_report(sample_reference_data, sample_current_data):
    """Test generating monitoring report."""
    monitor = ModelMonitor()
    monitor.set_reference_data(sample_reference_data)
    monitor.set_current_data(sample_current_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "monitoring_report.html"
        monitor.generate_monitoring_report(output_path=report_path)

        assert report_path.exists()


def test_model_monitor_should_trigger_retraining_no_issues():
    """Test retraining trigger with no issues."""
    monitor = ModelMonitor()

    drift_results = {"drift_detected": False, "requires_retraining": False}
    performance_results = {"performance_degradation": False, "accuracy": 0.75}

    should_retrain, reason = monitor.should_trigger_retraining(drift_results, performance_results)

    assert should_retrain is False
    assert reason == "No issues detected"


def test_model_monitor_should_trigger_retraining_drift():
    """Test retraining trigger with drift detected."""
    monitor = ModelMonitor()

    drift_results = {"drift_detected": True, "requires_retraining": True, "drift_share": 0.4}
    performance_results = {"performance_degradation": False, "accuracy": 0.75}

    should_retrain, reason = monitor.should_trigger_retraining(drift_results, performance_results)

    assert should_retrain is True
    assert "data drift" in reason.lower()


def test_model_monitor_should_trigger_retraining_performance():
    """Test retraining trigger with performance degradation."""
    monitor = ModelMonitor()

    drift_results = {"drift_detected": False, "requires_retraining": False}
    performance_results = {"performance_degradation": True, "accuracy": 0.55}

    should_retrain, reason = monitor.should_trigger_retraining(drift_results, performance_results)

    assert should_retrain is True
    assert "performance degradation" in reason.lower()


def test_model_monitor_should_trigger_retraining_both():
    """Test retraining trigger with both drift and performance issues."""
    monitor = ModelMonitor()

    drift_results = {"drift_detected": True, "requires_retraining": True, "drift_share": 0.5}
    performance_results = {"performance_degradation": True, "accuracy": 0.50}

    should_retrain, reason = monitor.should_trigger_retraining(drift_results, performance_results)

    assert should_retrain is True
    assert "data drift" in reason.lower()
    assert "performance degradation" in reason.lower()


def test_model_monitor_should_trigger_retraining_none_results():
    """Test retraining trigger with None results."""
    monitor = ModelMonitor()

    should_retrain, reason = monitor.should_trigger_retraining(drift_results=None, performance_results=None)

    assert should_retrain is False
    assert reason == "No issues detected"


def test_model_monitor_data_immutability(sample_reference_data):
    """Test that setting data creates a copy."""
    monitor = ModelMonitor()
    original_data = sample_reference_data.copy()

    monitor.set_reference_data(sample_reference_data)

    # Modify original data
    sample_reference_data["new_column"] = 1

    # Monitor's data should not be affected
    assert "new_column" not in monitor.reference_data.columns


def test_model_monitor_check_performance_with_all_zeros():
    """Test performance check with all zero predictions."""
    monitor = ModelMonitor()

    predictions = pd.Series([0, 0, 0, 0])
    actual = pd.Series([0, 0, 0, 0])

    performance_results = monitor.check_model_performance(predictions, actual)

    # Should handle edge case gracefully
    assert performance_results["accuracy"] == 1.0
    assert performance_results["precision"] >= 0.0
    assert performance_results["recall"] >= 0.0


def test_model_monitor_check_performance_with_single_class():
    """Test performance check with single class predictions."""
    monitor = ModelMonitor()

    predictions = pd.Series([0, 0, 0, 0])
    actual = pd.Series([1, 1, 1, 1])

    performance_results = monitor.check_model_performance(predictions, actual)

    # Should handle edge case without errors
    assert 0.0 <= performance_results["accuracy"] <= 1.0
    assert 0.0 <= performance_results["precision"] <= 1.0
    assert 0.0 <= performance_results["recall"] <= 1.0
