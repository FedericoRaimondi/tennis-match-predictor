"""Tests for feature-aware drift detection module."""

import numpy as np
import pandas as pd
import pytest

from match_predictor.ml_pipeline.drift_detection import FeatureAwareDriftDetector


@pytest.fixture
def sample_model():
    """Create a mock model with feature_names_in_ attribute."""

    class MockModel:
        def __init__(self, feature_names):
            self.feature_names_in_ = np.array(feature_names)

    return MockModel(["feature1", "feature2", "feature3"])


@pytest.fixture
def sample_reference_data():
    """Create sample reference data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),  # numerical
            "feature2": np.random.randn(100),  # numerical
            "feature3": np.random.randint(0, 5, 100),  # categorical
            "feature4": np.random.randn(100),  # extra feature not in model
        }
    )


@pytest.fixture
def sample_current_data():
    """Create sample current data with similar distribution."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randint(0, 5, 50),
            "feature4": np.random.randn(50),
        }
    )


@pytest.fixture
def sample_drifted_data():
    """Create sample data with significant drift."""
    np.random.seed(456)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(50) + 5,  # shifted distribution
            "feature2": np.random.randn(50) * 3,  # different variance
            "feature3": np.random.randint(5, 10, 50),  # different range
            "feature4": np.random.randn(50),
        }
    )


def test_drift_detector_init_with_model(sample_model):
    """Test drift detector initialization with model."""
    detector = FeatureAwareDriftDetector(model=sample_model)

    assert detector.feature_names == ["feature1", "feature2", "feature3"]
    assert detector.reference_data is None
    assert detector.current_data is None


def test_drift_detector_init_with_explicit_features():
    """Test drift detector initialization with explicit feature names."""
    feature_names = ["feat1", "feat2"]
    detector = FeatureAwareDriftDetector(feature_names=feature_names)

    assert detector.feature_names == feature_names
    assert detector.model is None


def test_drift_detector_init_without_features():
    """Test drift detector initialization without model or features."""
    detector = FeatureAwareDriftDetector()

    assert detector.feature_names is None
    assert detector.model is None


def test_drift_detector_set_reference_data(sample_model, sample_reference_data):
    """Test setting reference data."""
    detector = FeatureAwareDriftDetector(model=sample_model)
    detector.set_reference_data(sample_reference_data)

    assert detector.reference_data is not None
    assert len(detector.reference_data) == 100
    # Should only have features from model
    assert list(detector.reference_data.columns) == ["feature1", "feature2", "feature3"]


def test_drift_detector_set_current_data(sample_model, sample_current_data):
    """Test setting current data."""
    detector = FeatureAwareDriftDetector(model=sample_model)
    detector.set_current_data(sample_current_data)

    assert detector.current_data is not None
    assert len(detector.current_data) == 50
    # Should only have features from model
    assert list(detector.current_data.columns) == ["feature1", "feature2", "feature3"]


def test_drift_detector_detect_no_drift(sample_model, sample_reference_data, sample_current_data):
    """Test drift detection with no drift."""
    detector = FeatureAwareDriftDetector(model=sample_model)
    detector.set_reference_data(sample_reference_data)
    detector.set_current_data(sample_current_data)

    drift_results = detector.detect_drift()

    assert "drift_detected" in drift_results
    assert "drift_share" in drift_results
    assert "drifted_features" in drift_results
    assert "total_features" in drift_results
    assert "feature_drift_details" in drift_results
    assert drift_results["total_features"] == 3  # Only model features


def test_drift_detector_detect_with_drift(sample_model, sample_reference_data, sample_drifted_data):
    """Test drift detection with significant drift."""
    detector = FeatureAwareDriftDetector(model=sample_model)
    detector.set_reference_data(sample_reference_data)
    detector.set_current_data(sample_drifted_data)

    drift_results = detector.detect_drift()

    assert drift_results["drift_detected"] is True
    assert drift_results["drifted_features"] > 0
    assert drift_results["drift_share"] > 0
    # Should require retraining if drift share > 30%
    if drift_results["drift_share"] > 0.3:
        assert drift_results["requires_retraining"] is True


def test_drift_detector_without_data():
    """Test drift detection without setting data raises error."""
    detector = FeatureAwareDriftDetector()

    with pytest.raises(ValueError, match="Both reference and current data must be set"):
        detector.detect_drift()


def test_drift_detector_is_numerical():
    """Test numerical feature detection."""
    detector = FeatureAwareDriftDetector()

    numerical_series = pd.Series([1.0, 2.0, 3.0, 4.0])
    categorical_series = pd.Series(["a", "b", "c", "d"])

    assert detector._is_numerical(numerical_series) is True
    assert detector._is_numerical(categorical_series) is False


def test_drift_detector_is_categorical():
    """Test categorical feature detection."""
    detector = FeatureAwareDriftDetector()

    categorical_series = pd.Series(["a", "b", "c", "d"])
    numerical_series = pd.Series([1.0, 2.0, 3.0, 4.0])
    integer_categorical = pd.Series([0, 1, 0, 1, 0, 1])  # Few unique values

    assert detector._is_categorical(categorical_series) is True
    assert detector._is_categorical(numerical_series) is False
    assert detector._is_categorical(integer_categorical) is True


def test_drift_detector_ks_test():
    """Test Kolmogorov-Smirnov test."""
    detector = FeatureAwareDriftDetector()

    # Same distribution
    ref_series = pd.Series(np.random.randn(100))
    cur_series = pd.Series(np.random.randn(100))

    result = detector._kolmogorov_smirnov_test(ref_series, cur_series)

    assert "drift_detected" in result
    assert "statistic" in result
    assert "p_value" in result
    assert result["test"] == "ks"


def test_drift_detector_js_divergence():
    """Test Jensen-Shannon divergence."""
    detector = FeatureAwareDriftDetector()

    # Same distribution
    ref_series = pd.Series(["a", "b", "c"] * 30)
    cur_series = pd.Series(["a", "b", "c"] * 20)

    result = detector._jensen_shannon_divergence(ref_series, cur_series)

    assert "drift_detected" in result
    assert "divergence" in result
    assert result["test"] == "js"


def test_drift_detector_performance_metrics():
    """Test performance metrics calculation."""
    detector = FeatureAwareDriftDetector()

    predictions = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    actual = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])

    metrics = detector.calculate_performance_metrics(predictions, actual)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "performance_degradation" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_drift_detector_performance_metrics_perfect():
    """Test performance metrics with perfect predictions."""
    detector = FeatureAwareDriftDetector()

    predictions = pd.Series([0, 1, 0, 1, 0, 1])
    actual = pd.Series([0, 1, 0, 1, 0, 1])

    metrics = detector.calculate_performance_metrics(predictions, actual)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["performance_degradation"] is False


def test_drift_detector_feature_drift_details(sample_model, sample_reference_data, sample_current_data):
    """Test that feature drift details are populated correctly."""
    detector = FeatureAwareDriftDetector(model=sample_model)
    detector.set_reference_data(sample_reference_data)
    detector.set_current_data(sample_current_data)

    drift_results = detector.detect_drift()
    details = drift_results["feature_drift_details"]

    # Check that all model features have drift details
    assert "feature1" in details
    assert "feature2" in details
    assert "feature3" in details

    # Check structure of drift details
    for _feature, detail in details.items():
        assert "type" in detail
        assert "test" in detail
        assert "drift_detected" in detail


def test_drift_detector_without_model_features(sample_reference_data, sample_current_data):
    """Test drift detection without model features (uses all columns)."""
    detector = FeatureAwareDriftDetector()
    detector.set_reference_data(sample_reference_data)
    detector.set_current_data(sample_current_data)

    drift_results = detector.detect_drift()

    # Should include all features when no model is specified
    assert drift_results["total_features"] == 4


def test_drift_detector_handles_missing_values():
    """Test that drift detector handles missing values gracefully."""
    detector = FeatureAwareDriftDetector()

    ref_series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    cur_series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])

    result = detector._kolmogorov_smirnov_test(ref_series, cur_series)

    # Should not crash and should return valid result
    assert "drift_detected" in result
    assert "statistic" in result
    assert "p_value" in result


def test_drift_detector_xgboost_model_features():
    """Test feature extraction from XGBoost-like model."""

    class MockXGBModel:
        def get_booster(self):
            class Booster:
                feature_names = ["feat1", "feat2", "feat3"]

            return Booster()

    detector = FeatureAwareDriftDetector(model=MockXGBModel())

    assert detector.feature_names == ["feat1", "feat2", "feat3"]
