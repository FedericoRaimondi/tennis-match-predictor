"""Tests for feature engineering module."""

import pandas as pd
import pytest

from match_predictor.config import DataConfig
from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_ml_data():
    """Create sample ML data for testing."""
    data = {
        "tourney_id": ["2021-001", "2021-001"],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "player_1": [100, 101],
        "player_2": [101, 100],
        "winner": [0, 1],
        "player_rank": [10, 20],
        "opponent_rank": [20, 10],
        "p_1stIn": [30, 28],
        "p_svpt": [50, 48],
        "p_1stWon": [20, 18],
        "p_bpSaved": [2, 3],
        "p_bpFaced": [3, 4],
    }
    return pd.DataFrame(data)


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.config is not None


def test_prepare_features_missing_target():
    """Test that prepare_features raises error when target is missing."""
    engineer = FeatureEngineer()
    df = pd.DataFrame({"feature1": [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Target column 'winner' not found"):
        engineer.prepare_features_for_training(df)


def test_feature_engineer_with_custom_config():
    """Test FeatureEngineer with custom config."""
    config = DataConfig()
    config.features.rolling_windows = [5, 10]
    engineer = FeatureEngineer(config)
    
    assert engineer.config.features.rolling_windows == [5, 10]


def test_prepare_features_excludes_datetime():
    """Test that datetime columns are excluded from features."""
    engineer = FeatureEngineer()
    data = {
        "winner": [0, 1],
        "feature1": [10, 20],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
    }
    df = pd.DataFrame(data)
    X, y = engineer.prepare_features_for_training(df)
    
    # Datetime column should not be in features
    assert "tourney_date" not in X.columns
    assert "feature1" in X.columns


def test_get_feature_names_excludes_all_metadata():
    """Test that all metadata columns are excluded from feature names."""
    engineer = FeatureEngineer()
    data = {
        "winner": [0, 1],
        "player_1": [100, 101],
        "player_2": [101, 100],
        "p_id": [1, 2],
        "o_id": [2, 1],
        "tourney_id": ["T1", "T1"],
        "tourney_name": ["Tournament", "Tournament"],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "match_num": [1, 2],
        "p_name": ["A", "B"],
        "o_name": ["B", "A"],
        "player_rank": [10, 20],
    }
    df = pd.DataFrame(data)
    feature_names = engineer.get_feature_names(df)

    # All metadata should be excluded
    metadata_columns = [
        "winner",
        "player_1",
        "player_2",
        "p_id",
        "o_id",
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "match_num",
        "p_name",
        "o_name",
    ]
    for col in metadata_columns:
        assert col not in feature_names

    # Only player_rank should be in features
    assert "player_rank" in feature_names
