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
