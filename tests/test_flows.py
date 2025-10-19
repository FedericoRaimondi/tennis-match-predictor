"""Tests for Prefect flows."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from match_predictor.ml_pipeline.flows import (
    check_new_data_task,
    evaluate_and_promote_task,
    load_data_task,
    prepare_ml_data_task,
    save_data_task,
    train_model_task,
)


@pytest.fixture
def sample_matches():
    """Create sample match data."""
    return pd.DataFrame({
        "tourney_date": ["2021-01-01", "2021-01-02"],
        "winner_name": ["Player A", "Player B"],
        "loser_name": ["Player B", "Player A"],
        "tourney_name": ["Tournament 1", "Tournament 1"],
    })


@pytest.fixture
def sample_ml_data():
    """Create sample ML data."""
    return pd.DataFrame({
        "tourney_id": ["2021-001", "2021-001"],
        "tourney_date": ["2021-01-01", "2021-01-02"],
        "player_1": [100, 101],
        "player_2": [101, 100],
        "winner": [0, 1],
        "player_rank": [10, 20],
        "opponent_rank": [20, 10],
    })


@pytest.fixture
def mock_data_config():
    """Create mock data configuration."""
    from match_predictor.config import DataConfig
    return DataConfig()


@pytest.fixture
def mock_model_config():
    """Create mock model configuration."""
    from match_predictor.config import ModelConfig
    return ModelConfig()


def test_load_data_task(mock_data_config, sample_matches, monkeypatch):
    """Test load_data_task."""
    mock_loader = MagicMock()
    mock_loader.load_matches.return_value = sample_matches
    
    def mock_init(self, repo):
        pass
    
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.__init__", mock_init)
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.load_matches", 
                       lambda self: sample_matches)
    
    result = load_data_task(mock_data_config)
    assert len(result) == 2


def test_prepare_ml_data_task(sample_matches, monkeypatch):
    """Test prepare_ml_data_task."""
    mock_ml_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    def mock_init(self, repo):
        pass
    
    def mock_get_ml_data(self, df):
        return mock_ml_data
    
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.__init__", mock_init)
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.get_ml_data", mock_get_ml_data)
    
    result = prepare_ml_data_task(sample_matches)
    assert len(result) == 2


def test_save_data_task(sample_matches, mock_data_config, tmp_path, monkeypatch):
    """Test save_data_task."""
    # Override the data path to use temp directory
    mock_data_config.inference_data_path = str(tmp_path)
    
    save_data_task(sample_matches, mock_data_config)
    
    # Verify file was created
    expected_file = tmp_path / mock_data_config.matches_results_file
    assert expected_file.exists()
    
    # Verify data can be loaded
    loaded = pd.read_pickle(expected_file)
    assert len(loaded) == len(sample_matches)


def test_evaluate_and_promote_task_no_champion(mock_model_config, tmp_path):
    """Test evaluate_and_promote_task when no champion exists."""
    # Override model path to use temp directory
    mock_model_config.champion_model_path = str(tmp_path)
    
    # Create a mock latest model
    latest_path = tmp_path / "latest_model.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump({
            "model": "mock_model",
            "metrics": {"test_accuracy": 0.75}
        }, f)
    
    metrics = {"test_accuracy": 0.75}
    promoted = evaluate_and_promote_task(metrics, mock_model_config)
    
    assert promoted is True
    assert (tmp_path / "champion_model.pkl").exists()


def test_evaluate_and_promote_task_better_model(mock_model_config, tmp_path):
    """Test evaluate_and_promote_task when new model is better."""
    # Override model path to use temp directory
    mock_model_config.champion_model_path = str(tmp_path)
    
    # Create mock champion model
    champion_path = tmp_path / "champion_model.pkl"
    with open(champion_path, "wb") as f:
        pickle.dump({
            "model": "champion_model",
            "metrics": {"test_accuracy": 0.65}
        }, f)
    
    # Create mock latest model
    latest_path = tmp_path / "latest_model.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump({
            "model": "new_model",
            "metrics": {"test_accuracy": 0.75}
        }, f)
    
    metrics = {"test_accuracy": 0.75}
    promoted = evaluate_and_promote_task(metrics, mock_model_config)
    
    assert promoted is True


def test_evaluate_and_promote_task_worse_model(mock_model_config, tmp_path):
    """Test evaluate_and_promote_task when new model is worse."""
    # Override model path to use temp directory
    mock_model_config.champion_model_path = str(tmp_path)
    
    # Create mock champion model
    champion_path = tmp_path / "champion_model.pkl"
    with open(champion_path, "wb") as f:
        pickle.dump({
            "model": "champion_model",
            "metrics": {"test_accuracy": 0.75}
        }, f)
    
    # Create mock latest model
    latest_path = tmp_path / "latest_model.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump({
            "model": "new_model",
            "metrics": {"test_accuracy": 0.65}
        }, f)
    
    metrics = {"test_accuracy": 0.65}
    promoted = evaluate_and_promote_task(metrics, mock_model_config)
    
    assert promoted is False


def test_check_new_data_task_no_existing(mock_data_config, sample_matches, tmp_path, monkeypatch):
    """Test check_new_data_task when no existing data."""
    # Override data path to use temp directory
    mock_data_config.inference_data_path = str(tmp_path)
    
    def mock_init(self, repo):
        pass
    
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.__init__", mock_init)
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.load_matches",
                       lambda self: sample_matches)
    
    has_new_data = check_new_data_task(mock_data_config)
    
    assert has_new_data is True


def test_check_new_data_task_with_new_data(mock_data_config, sample_matches, tmp_path, monkeypatch):
    """Test check_new_data_task when new data is available."""
    # Override data path to use temp directory
    mock_data_config.inference_data_path = str(tmp_path)
    
    # Create existing data with less rows
    existing_data = sample_matches.head(1)
    matches_file = tmp_path / mock_data_config.matches_results_file
    existing_data.to_pickle(matches_file)
    
    def mock_init(self, repo):
        pass
    
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.__init__", mock_init)
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.load_matches",
                       lambda self: sample_matches)
    
    has_new_data = check_new_data_task(mock_data_config)
    
    assert has_new_data is True


def test_check_new_data_task_no_new_data(mock_data_config, sample_matches, tmp_path, monkeypatch):
    """Test check_new_data_task when no new data is available."""
    # Override data path to use temp directory
    mock_data_config.inference_data_path = str(tmp_path)
    
    # Create existing data with same rows
    matches_file = tmp_path / mock_data_config.matches_results_file
    sample_matches.to_pickle(matches_file)
    
    def mock_init(self, repo):
        pass
    
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.__init__", mock_init)
    monkeypatch.setattr("match_predictor.ml_pipeline.flows.DataLoader.load_matches",
                       lambda self: sample_matches)
    
    has_new_data = check_new_data_task(mock_data_config)
    
    assert has_new_data is False
