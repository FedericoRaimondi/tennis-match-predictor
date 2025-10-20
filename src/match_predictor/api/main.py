"""FastAPI application for tennis match prediction."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from match_predictor.ml_pipeline.training import ModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="Tennis Match Predictor API", description="API for predicting tennis match outcomes", version="1.0.0"
)

# Model info
MODEL_NAME = "champion_model.pkl"
MODEL_PATH = Path("model") / MODEL_NAME
DATA_PATH = Path("data")

# Load the champion model
try:
    # load model from pickle
    model_trainer = ModelTrainer.load_model(MODEL_PATH)
    clf = model_trainer.model
    feature_names = model_trainer.feature_names
    logger.info("Champion model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load champion model: {e}. API will start but predictions will fail.")
    clf = None


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    player1: int
    player2: int
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
def info() -> dict[str, str | bool]:
    """Get API information."""
    return {
        "name": "tennis_predictor",
        "description": "Predict the outcome of tennis matches.",
        "version": "1.0.0",
        "model_loaded": clf is not None,
    }


@app.post("/predict_winner", response_model=PredictResponse)
async def predict_winner(request: PredictRequest) -> PredictResponse:
    """Predict the winner of a tennis match.

    This endpoint retrieves the champion model from the registry and uses it to predict
    the match outcome based on:
    - Latest player statistics (from saved player stats)
    - Tournament information (surface, location, etc.)
    - Dynamically constructed inference-ready features

    Args:
        request: PredictRequest containing player names and tournament

    Returns:
        PredictResponse with win probabilities, predicted winner, and latest 5 match stats
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Load latest player stats from CSV/Parquet (saved by save_latest_player_stats)
        player_stats_file_csv = DATA_PATH / "player_stats_latest.csv"
        player_stats_file_parquet = DATA_PATH / "player_stats_latest.parquet"

        # Try to load from either CSV or Parquet
        if player_stats_file_parquet.exists():
            player_stats = pd.read_parquet(player_stats_file_parquet)
        elif player_stats_file_csv.exists():
            player_stats = pd.read_csv(player_stats_file_csv)
        else:
            raise HTTPException(
                status_code=404,
                detail="Player stats data not found. Please ensure save_latest_player_stats() has been called.",
            )

        # Load tournament information
        tournament_info_file = DATA_PATH / "tournament_info.csv"
        if not tournament_info_file.exists():
            raise HTTPException(
                status_code=404, detail="Tournament info data not found. Please ensure data is available."
            )

        tournament_info = pd.read_csv(tournament_info_file)

        # Get latest stats for both players
        player1_stats = player_stats[player_stats["p_id"] == request.player1]
        player2_stats = player_stats[player_stats["p_id"] == request.player2]

        player1_name = player1_stats["player_name"].values[0] if not player1_stats.empty else "Unknown Player 1"
        player2_name = player2_stats["player_name"].values[0] if not player2_stats.empty else "Unknown Player 2"

        if player1_stats.empty or player2_stats.empty:
            raise HTTPException(
                status_code=404, detail=f"Stats not found for one or both players: {request.player1}, {request.player2}"
            )

        player2_stats.columns = [f"{col}_p2" for col in player2_stats.columns]

        # Get tournament information
        tournament_data = tournament_info[tournament_info["tourney_name"] == request.tournament]

        if tournament_data.empty:
            # Use default tournament info if not found
            logger.warning(f"Tournament '{request.tournament}' not found, using default values")
            tournament_dict = {
                "tourney_name": "Default Tournament",
                "surface": "Hard",
                "tourney_level": "A",
                "draw_size": 32,
            }
            tournament_data = pd.DataFrame([tournament_dict])

        # Check the dfs are only one row (latest stats)
        assert len(player1_stats) == 1, "Expected single row for player 1 stats"
        assert len(player2_stats) == 1, "Expected single row for player 2 stats"
        assert len(tournament_data) == 1, "Expected at most one row for tournament info"

        # Merge all info into a single df
        inference_df = pd.concat(
            [
                tournament_data.reset_index(drop=True),
                player1_stats.reset_index(drop=True),
                player2_stats.reset_index(drop=True),
            ],
            axis=1,
        )
        assert len(inference_df) == 1, "Expected single row for combined inference data"

        # Filter for feature names used in training
        inference_df = inference_df.reindex(columns=feature_names)

        # Make prediction using the exact features the model expects
        probabilities = clf.predict_proba(inference_df)[0]
        player1_prob = float(probabilities[0])
        player2_prob = float(probabilities[1])

        predicted_winner = request.player1 if player1_prob > player2_prob else request.player2

        return PredictResponse(
            player1=player1_name,
            player2=player2_name,
            tournament=request.tournament,
            player1_win_probability=player1_prob,
            player2_win_probability=player2_prob,
            predicted_winner=predicted_winner,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")  # noqa: B904


@app.get("/latest_matches")
async def get_latest_matches(player: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Get the latest match statistics.

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
            matches_df = matches_df[(matches_df["winner_name"] == player) | (matches_df["loser_name"] == player)]

        # Get latest matches
        latest_matches = matches_df.sort_values("tourney_date", ascending=False).head(limit)

        # Format matches
        matches_list = []
        for _, row in latest_matches.iterrows():
            matches_list.append(
                {
                    "date": str(row["tourney_date"]),
                    "player1": row["winner_name"],
                    "player2": row["loser_name"],
                    "winner": row["winner_name"],
                    "tournament": row["tourney_name"],
                    "surface": row.get("surface"),
                    "score": row.get("score"),
                }
            )

        return {"count": len(matches_list), "matches": matches_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch matches: {str(e)}")  # noqa: B904


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
