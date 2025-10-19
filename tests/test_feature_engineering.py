"""Tests for feature engineering module."""

import pandas as pd
import pytest

from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_ml_data():
    """Create sample ML data for testing."""
    data = {
        "tourney_id": ["2021-001", "2021-001"],
        "tourney_date": ["2021-01-01", "2021-01-02"],
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


def test_engineer_features(sample_ml_data):
    """Test feature engineering."""
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(sample_ml_data)

    # Check that new features are added
    assert "p_1st_serve_pct" in df_engineered.columns
    assert "p_1st_serve_win_pct" in df_engineered.columns
    assert "p_bp_saved_pct" in df_engineered.columns


def test_add_match_statistics(sample_ml_data):
    """Test match statistics calculation."""
    engineer = FeatureEngineer()
    df_with_stats = engineer._add_match_statistics(sample_ml_data)

    # Verify calculated features
    assert df_with_stats["p_1st_serve_pct"].iloc[0] == 30 / 50
    assert df_with_stats["p_1st_serve_win_pct"].iloc[0] == 20 / 30


def test_add_player_rankings(sample_ml_data):
    """Test player ranking features."""
    engineer = FeatureEngineer()
    df_with_ranks = engineer._add_player_rankings(sample_ml_data)

    # Verify rank difference
    assert df_with_ranks["rank_difference"].iloc[0] == 10 - 20
    assert df_with_ranks["rank_difference"].iloc[1] == 20 - 10


def test_add_temporal_features(sample_ml_data):
    """Test temporal feature extraction."""
    engineer = FeatureEngineer()
    df_with_time = engineer._add_temporal_features(sample_ml_data)

    # Verify temporal features
    assert "month" in df_with_time.columns
    assert "quarter" in df_with_time.columns
    assert "day_of_year" in df_with_time.columns
    assert df_with_time["month"].iloc[0] == 1


def test_prepare_features_for_training(sample_ml_data):
    """Test feature preparation for training."""
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(sample_ml_data)
    X, y = engineer.prepare_features_for_training(df_engineered)

    # Verify X and y shapes
    assert len(X) == len(sample_ml_data)
    assert len(y) == len(sample_ml_data)

    # Verify target values
    assert list(y) == [0, 1]

    # Verify metadata columns are excluded
    assert "player_1" not in X.columns
    assert "player_2" not in X.columns
    assert "winner" not in X.columns


def test_prepare_features_missing_target():
    """Test feature preparation with missing target."""
    engineer = FeatureEngineer()
    df = pd.DataFrame({"feature1": [1, 2, 3]})

    with pytest.raises(ValueError, match="Target column 'winner' not found"):
        engineer.prepare_features_for_training(df)


def test_get_feature_names(sample_ml_data):
    """Test getting feature names."""
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(sample_ml_data)
    feature_names = engineer.get_feature_names(df_engineered)

    # Verify feature names don't include metadata
    assert "winner" not in feature_names
    assert "player_1" not in feature_names
    assert "tourney_id" not in feature_names

    # Verify feature names include actual features
    assert any("rank" in name for name in feature_names)
