"""FastAPI application for tennis match prediction."""

import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from match_predictor.data.data_loader import DataLoader

# Initialize FastAPI app
app = FastAPI(
    title="Tennis Match Predictor API",
    description="API for predicting tennis match outcomes",
    version="1.0.0"
)

# Model info
MODEL_NAME = "tennis_predictor_model"
DATA_PATH = Path("data")

# Load the champion model
try:
    clf = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
    logger.info("Champion model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load champion model: {e}. API will start but predictions will fail.")
    clf = None


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    player1: str
    player2: str
    tournament: str


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    player1: str
    player2: str
    tournament: str
    player1_win_probability: float
    player2_win_probability: float
    predicted_winner: str


class MatchStats(BaseModel):
    """Model for match statistics."""

    date: str
    player1: str
    player2: str
    winner: str
    tournament: str
    surface: str | None = None


@app.get("/")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"health_check": "OK", "status": "running"}


@app.get("/info")
def info() -> dict[str, str]:
    """Get API information."""
    return {
        "name": "tennis_predictor",
        "description": "Predict the outcome of tennis matches.",
        "version": "1.0.0",
        "model_loaded": clf is not None
    }


@app.post("/predict_winner", response_model=PredictResponse)
async def predict_winner(request: PredictRequest) -> PredictResponse:
    """
    Predict the winner of a tennis match.

    Args:
        request: PredictRequest containing player names and tournament

    Returns:
        PredictResponse with win probabilities and predicted winner
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Load player stats
        player_stats_file = DATA_PATH / "player_stats_hist.pkl"
        if not player_stats_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Player stats data not found. Please ensure data is available."
            )

        with open(player_stats_file, "rb") as f:
            player_stats = pickle.load(f)

        # Get latest stats for both players
        player1_stats = player_stats[player_stats["player_name"] == request.player1].tail(1)
        player2_stats = player_stats[player_stats["player_name"] == request.player2].tail(1)

        if player1_stats.empty or player2_stats.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Stats not found for one or both players: {request.player1}, {request.player2}"
            )

        # Prepare features for prediction (this is a simplified version)
        # In a real implementation, you would combine features according to your model's requirements
        features = pd.DataFrame({
            # Add feature columns based on your model's expected input
            # This is a placeholder - adjust based on your actual feature set
        })

        # Make prediction
        probabilities = clf.predict_proba(features)[0]
        player1_prob = float(probabilities[0])
        player2_prob = float(probabilities[1])

        predicted_winner = request.player1 if player1_prob > player2_prob else request.player2

        return PredictResponse(
            player1=request.player1,
            player2=request.player2,
            tournament=request.tournament,
            player1_win_probability=player1_prob,
            player2_win_probability=player2_prob,
            predicted_winner=predicted_winner
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/latest_matches")
async def get_latest_matches(player: str | None = None, limit: int = 5) -> dict[str, Any]:
    """
    Get the latest match statistics.

    Args:
        player: Optional player name to filter matches
        limit: Number of matches to return (default: 5)

    Returns:
        Dictionary with latest match statistics
    """
    try:
        matches_file = DATA_PATH / "matches_results.pkl"
        if not matches_file.exists():
            raise HTTPException(status_code=404, detail="Match data not found")

        with open(matches_file, "rb") as f:
            matches_df = pickle.load(f)

        # Filter by player if specified
        if player:
            matches_df = matches_df[
                (matches_df["winner_name"] == player) | (matches_df["loser_name"] == player)
            ]

        # Get latest matches
        latest_matches = matches_df.sort_values("tourney_date", ascending=False).head(limit)

        # Format matches
        matches_list = []
        for _, row in latest_matches.iterrows():
            matches_list.append({
                "date": str(row["tourney_date"]),
                "player1": row["winner_name"],
                "player2": row["loser_name"],
                "winner": row["winner_name"],
                "tournament": row["tourney_name"],
                "surface": row.get("surface"),
                "score": row.get("score"),
            })

        return {
            "count": len(matches_list),
            "matches": matches_list
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch matches: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
