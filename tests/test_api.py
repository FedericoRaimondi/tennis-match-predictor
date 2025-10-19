"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from match_predictor.api.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK", "status": "running"}


def test_info_endpoint():
    """Test the info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data
    assert "version" in data
    assert data["name"] == "tennis_predictor"


def test_predict_winner_missing_model(monkeypatch):
    """Test prediction endpoint when model is not loaded."""
    # Mock the model to be None
    import match_predictor.api.main as api_main

    original_clf = api_main.clf
    api_main.clf = None

    response = client.post(
        "/predict_winner",
        json={
            "player1": "Novak Djokovic",
            "player2": "Rafael Nadal",
            "tournament": "Wimbledon"
        }
    )

    # Restore original
    api_main.clf = original_clf

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_latest_matches_no_data():
    """Test latest matches endpoint when data is not available."""
    response = client.get("/latest_matches")
    # Should return 404 or empty result depending on data availability
    assert response.status_code in [200, 404]


def test_latest_matches_with_player():
    """Test latest matches endpoint with player filter."""
    response = client.get("/latest_matches?player=Roger Federer&limit=3")
    # Should return 200 or 404 depending on data availability
    assert response.status_code in [200, 404]


def test_predict_winner_validation():
    """Test prediction endpoint with invalid data."""
    response = client.post(
        "/predict_winner",
        json={
            "player1": "Test Player",
            # Missing player2 and tournament
        }
    )
    assert response.status_code == 422  # Validation error


def test_info_endpoint_model_loaded():
    """Test info endpoint includes model_loaded status."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_predict_winner_request_model():
    """Test PredictRequest model validation."""
    from match_predictor.api.main import PredictRequest

    request = PredictRequest(player1="Player A", player2="Player B", tournament="Wimbledon")
    assert request.player1 == "Player A"
    assert request.player2 == "Player B"
    assert request.tournament == "Wimbledon"


def test_predict_response_model():
    """Test PredictResponse model."""
    from match_predictor.api.main import PredictResponse

    response = PredictResponse(
        player1="Player A",
        player2="Player B",
        tournament="Wimbledon",
        player1_win_probability=0.65,
        player2_win_probability=0.35,
        predicted_winner="Player A",
    )
    assert response.player1 == "Player A"
    assert response.predicted_winner == "Player A"
    assert response.player1_win_probability == 0.65


def test_match_stats_model():
    """Test MatchStats model."""
    from match_predictor.api.main import MatchStats

    stats = MatchStats(
        date="2021-01-01", player1="Player A", player2="Player B", winner="Player A", tournament="US Open"
    )
    assert stats.date == "2021-01-01"
    assert stats.winner == "Player A"
    assert stats.surface is None


def test_match_stats_model_with_surface():
    """Test MatchStats model with surface."""
    from match_predictor.api.main import MatchStats

    stats = MatchStats(
        date="2021-01-01",
        player1="Player A",
        player2="Player B",
        winner="Player A",
        tournament="Roland Garros",
        surface="Clay",
    )
    assert stats.surface == "Clay"


def test_latest_matches_default_limit():
    """Test latest matches endpoint with default limit."""
    response = client.get("/latest_matches")
    # Should return 200 or 404 depending on data availability
    assert response.status_code in [200, 404]


def test_latest_matches_custom_limit():
    """Test latest matches endpoint with custom limit."""
    response = client.get("/latest_matches?limit=10")
    # Should return 200 or 404 depending on data availability
    assert response.status_code in [200, 404]


def test_health_check_returns_correct_status():
    """Test health check returns expected keys."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["health_check"] == "OK"
    assert data["status"] == "running"


def test_info_version():
    """Test info endpoint returns version."""
    response = client.get("/info")
    data = response.json()
    assert data["version"] == "1.0.0"


def test_api_title():
    """Test API has correct title."""
    from match_predictor.api.main import app

    assert app.title == "Tennis Match Predictor API"


def test_api_description():
    """Test API has description."""
    from match_predictor.api.main import app

    assert "predicting tennis match outcomes" in app.description.lower()


def test_api_version():
    """Test API has version."""
    from match_predictor.api.main import app

    assert app.version == "1.0.0"
