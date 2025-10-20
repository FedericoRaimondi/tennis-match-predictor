"""Integration tests for Prefect flows."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from match_predictor.config import DataConfig, ModelConfig
from match_predictor.ml_pipeline.flows import (
    evaluate_and_promote_task,
    load_data_task,
    prepare_ml_data_task,
    save_data_task,
    train_model_task,
    training_flow,
)


@pytest.fixture
def sample_matches():
    """Create sample match data."""
    return pd.DataFrame({
        "tourney_date": pd.to_datetime(["20210101", "20210102"]),
        "tourney_id": ["2021-001", "2021-001"],
        "tourney_name": ["Test Open", "Test Open"],
        "tourney_level": ["A", "A"],
        "tourney_year": [2021, 2021],
        "surface": ["Hard", "Hard"],
        "draw_size": [32, 32],
        "match_num": [1, 2],
        "winner_id": [100, 101],
        "winner_seed": [1, 2],
        "winner_entry": ["", ""],
        "winner_name": ["Player A", "Player B"],
        "winner_hand": ["R", "L"],
        "winner_ht": [180, 185],
        "winner_ioc": ["USA", "ESP"],
        "winner_age": [25, 27],
        "winner_rank": [10, 20],
        "winner_rank_points": [2000, 1500],
        "w_ace": [5, 6],
        "w_df": [1, 2],
        "w_svpt": [50, 48],
        "w_1stIn": [30, 28],
        "w_1stWon": [20, 18],
        "w_2ndWon": [15, 14],
        "w_SvGms": [10, 9],
        "w_bpSaved": [2, 3],
        "w_bpFaced": [3, 4],
        "loser_id": [101, 100],
        "loser_seed": [2, 1],
        "loser_entry": ["", ""],
        "loser_name": ["Player B", "Player A"],
        "loser_hand": ["L", "R"],
        "loser_ht": [185, 180],
        "loser_ioc": ["ESP", "USA"],
        "loser_age": [27, 25],
        "loser_rank": [20, 10],
        "loser_rank_points": [1500, 2000],
        "l_ace": [6, 5],
        "l_df": [2, 1],
        "l_svpt": [48, 50],
        "l_1stIn": [28, 30],
        "l_1stWon": [18, 20],
        "l_2ndWon": [14, 15],
        "l_SvGms": [9, 10],
        "l_bpSaved": [3, 2],
        "l_bpFaced": [4, 3],
        "score": ["6-3 6-4", "7-5 6-2"],
        "best_of": [3, 3],
        "round": ["R32", "R16"],
        "minutes": [90, 95],
    })


@pytest.fixture
def mock_data_config(tmp_path):
    """Create mock data configuration."""
    config = DataConfig()
    config.inference_data_path = str(tmp_path / "data")
    config.matches_results_file = "matches_results.csv"
    config.tournament_info_file = "tournament_info.csv"
    config.player_stats_file = "player_stats.csv"
    return config


@pytest.fixture
def mock_model_config(tmp_path):
    """Create mock model configuration."""
    config = ModelConfig()
    config.champion_model_path = str(tmp_path / "models")
    config.training.min_accuracy_threshold = 0.6
    return config


def test_load_data_task_integration(sample_matches, mock_data_config, monkeypatch):
    """Test load_data_task integration."""
    # Mock the DataLoader to return sample data
    def mock_load_matches(self):
        return sample_matches
    
    from match_predictor.data import data_loader
    monkeypatch.setattr(data_loader.DataLoader, "load_matches", mock_load_matches)
    
    # Execute task
    result = load_data_task(mock_data_config)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "winner_name" in result.columns


def test_save_data_task_integration(sample_matches, mock_data_config, tmp_path):
    """Test save_data_task integration."""
    # Create data directory
    data_path = Path(mock_data_config.inference_data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Execute task
    save_data_task(sample_matches, mock_data_config)
    
    # Verify files were created
    matches_file = data_path / mock_data_config.matches_results_file
    tournament_file = data_path / mock_data_config.tournament_info_file
    player_stats_file = data_path / mock_data_config.player_stats_file
    
    assert matches_file.exists()
    assert tournament_file.exists()
    assert player_stats_file.exists()
    
    # Verify content
    saved_matches = pd.read_csv(matches_file)
    assert len(saved_matches) == 2


def test_prepare_ml_data_task_integration(sample_matches, mock_data_config, monkeypatch):
    """Test prepare_ml_data_task integration."""
    # Mock random choice to make test deterministic
    import numpy as np
    monkeypatch.setattr(np.random, "randint", lambda a, b, size: np.zeros(size, dtype=int))
    
    # Execute task
    result = prepare_ml_data_task(mock_data_config, sample_matches)
    
    assert isinstance(result, pd.DataFrame)
    assert "winner" in result.columns
    assert "player_1" in result.columns
    assert "player_2" in result.columns


def test_evaluate_and_promote_task_promotes_good_model(mock_model_config, tmp_path):
    """Test evaluate_and_promote_task promotes model with good metrics."""
    metrics = {"test_accuracy": 0.75, "train_accuracy": 0.80}
    
    # Create mock model file
    models_path = Path(mock_model_config.champion_model_path)
    models_path.mkdir(parents=True, exist_ok=True)
    latest_model = models_path / "latest_model.pkl"
    latest_model.write_text("mock model")
    
    # Execute task
    promoted = evaluate_and_promote_task(metrics, mock_model_config)
    
    assert promoted is True
    # Check champion model was created
    champion_model = models_path / "champion_model.pkl"
    assert champion_model.exists()


def test_evaluate_and_promote_task_rejects_poor_model(mock_model_config, tmp_path):
    """Test evaluate_and_promote_task rejects model with poor metrics when champion is better."""
    metrics = {"test_accuracy": 0.70, "train_accuracy": 0.72}
    
    # Create mock model files
    models_path = Path(mock_model_config.champion_model_path)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Create an existing champion with better accuracy
    champion_model = models_path / "champion_model.pkl"
    champion_data = {"metrics": {"test_accuracy": 0.80}}  # Better than new model
    with open(champion_model, "wb") as f:
        import pickle
        pickle.dump(champion_data, f)
    
    latest_model = models_path / "latest_model.pkl"
    latest_model.write_text("mock model")
    
    # Execute task
    promoted = evaluate_and_promote_task(metrics, mock_model_config)
    
    assert promoted is False
    # Champion model should still exist with original data
    with open(champion_model, "rb") as f:
        import pickle
        saved_data = pickle.load(f)
    assert saved_data["metrics"]["test_accuracy"] == 0.80


def test_train_model_task_integration(mock_model_config, tmp_path, monkeypatch):
    """Test train_model_task integration with mocked training."""
    # Create simple ML data
    ml_data = pd.DataFrame({
        "winner": [0, 1, 0, 1],
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
    })
    
    # Mock the ModelTrainer to avoid actual training
    mock_metrics = {"test_accuracy": 0.75, "train_accuracy": 0.80}
    
    def mock_train(self, data, tune_hyperparameters=False, log_to_mlflow=False):
        return mock_metrics
    
    def mock_save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("mock model")
    
    from match_predictor.ml_pipeline import training
    monkeypatch.setattr(training.ModelTrainer, "train", mock_train)
    monkeypatch.setattr(training.ModelTrainer, "save_model", mock_save_model)
    
    # Execute task
    result = train_model_task(ml_data, mock_model_config, tune_hyperparameters=False)
    
    assert result == mock_metrics
    assert result["test_accuracy"] == 0.75


def test_training_flow_integration(sample_matches, tmp_path, monkeypatch):
    """Test complete training flow integration."""
    # Create config files
    config_path = tmp_path / "config"
    config_path.mkdir()
    
    data_config_path = config_path / "data_config.yaml"
    model_config_path = config_path / "model_config.yaml"
    
    # Write minimal config files
    data_config_path.write_text(f"""
inference_data_path: {tmp_path / "data"}
matches_results_file: matches_results.csv
tournament_info_file: tournament_info.csv
player_stats_file: player_stats.csv
""")
    
    model_config_path.write_text(f"""
champion_model_path: {tmp_path / "models"}
training:
  min_accuracy_threshold: 0.6
""")
    
    # Mock all the heavy operations
    def mock_load_matches(self):
        return sample_matches
    
    mock_metrics = {"test_accuracy": 0.75, "train_accuracy": 0.80}
    
    def mock_train(self, data, tune_hyperparameters=False, log_to_mlflow=False):
        return mock_metrics
    
    def mock_save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("mock model")
    
    import numpy as np
    from match_predictor.data import data_loader
    from match_predictor.ml_pipeline import training
    
    monkeypatch.setattr(data_loader.DataLoader, "load_matches", mock_load_matches)
    monkeypatch.setattr(training.ModelTrainer, "train", mock_train)
    monkeypatch.setattr(training.ModelTrainer, "save_model", mock_save_model)
    monkeypatch.setattr(np.random, "randint", lambda a, b, size: np.zeros(size, dtype=int))
    
    # Execute flow
    result = training_flow(
        tune_hyperparameters=False,
        data_config_path=str(data_config_path),
        model_config_path=str(model_config_path),
    )
    
    assert "metrics" in result
    assert "promoted" in result
    assert result["metrics"]["test_accuracy"] == 0.75
    assert result["promoted"] is True


def test_check_new_data_task_no_existing(mock_data_config, sample_matches, monkeypatch):
    """Test check_new_data_task when no existing data."""
    def mock_load_matches(self):
        return sample_matches
    
    from match_predictor.data import data_loader
    monkeypatch.setattr(data_loader.DataLoader, "load_matches", mock_load_matches)
    
    from match_predictor.ml_pipeline.flows import check_new_data_task
    
    # Execute task
    has_new = check_new_data_task(mock_data_config)
    
    assert has_new is True
