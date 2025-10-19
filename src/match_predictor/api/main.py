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
    latest_matches: list[dict] = []


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
        "model_loaded": clf is not None
    }


@app.post("/predict_winner", response_model=PredictResponse)
async def predict_winner(request: PredictRequest) -> PredictResponse:
    """
    Predict the winner of a tennis match.

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
                detail="Player stats data not found. Please ensure save_latest_player_stats() has been called."
            )

        # Load tournament information
        tournament_info_file = DATA_PATH / "tournament_info.pkl"
        if not tournament_info_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Tournament info data not found. Please ensure data is available."
            )
        
        with open(tournament_info_file, "rb") as f:
            tournament_info = pickle.load(f)

        # Get latest stats for both players
        player1_stats = player_stats[player_stats["player_name"] == request.player1]
        player2_stats = player_stats[player_stats["player_name"] == request.player2]

        if player1_stats.empty or player2_stats.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Stats not found for one or both players: {request.player1}, {request.player2}"
            )

        # Get the latest stats (last row for each player)
        player1_latest = player1_stats.iloc[-1]
        player2_latest = player2_stats.iloc[-1]

        # Get tournament information
        tournament_data = tournament_info[tournament_info["tourney_name"] == request.tournament]
        
        if tournament_data.empty:
            # Use default tournament info if not found
            logger.warning(f"Tournament '{request.tournament}' not found, using default surface")
            surface = "Hard"
            tourney_level = "A"
        else:
            latest_tournament = tournament_data.iloc[-1]
            surface = latest_tournament.get("surface", "Hard")
            tourney_level = latest_tournament.get("tourney_level", "A")

        # Construct inference features
        # Combine player1 stats, player2 stats, and tournament info
        from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer
        
        # Create a row similar to training data structure
        inference_row = {
            # Player 1 stats (these columns come from get_player_stats)
            "player_1": player1_latest.get("player_id"),
            "player_2": player2_latest.get("player_id"),
            "player_name": request.player1,
            "opponent_name": request.player2,
            "surface": surface,
            "tourney_level": tourney_level,
        }
        
        # Add all player stats with proper suffixes
        for col in player1_latest.index:
            if col.startswith("player_") or col.startswith("p_") or col.startswith("elo") or col in ["player_rank", "player_rank_points", "player_age", "player_hand", "player_ht", "player_ioc"]:
                inference_row[col] = player1_latest[col]
            if col.startswith("opponent_") or col.startswith("o_"):
                # For player 2, we'll add these from player2_latest but as opponent stats
                pass
        
        # Add player 2 stats as opponent for player 1's perspective
        for col in player2_latest.index:
            if col.startswith("player_"):
                new_col = col.replace("player_", "opponent_")
                inference_row[new_col] = player2_latest[col]
            elif col.startswith("p_"):
                new_col = col.replace("p_", "o_")
                inference_row[new_col] = player2_latest[col]

        # Create DataFrame with single row for inference
        inference_df = pd.DataFrame([inference_row])
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer()
        inference_df_engineered = feature_engineer.engineer_features(inference_df)
        
        # Get features (without target since we're predicting)
        feature_names = feature_engineer.get_feature_names(inference_df_engineered)
        
        # Prepare features - ensure all expected columns exist
        X_inference = pd.DataFrame()
        for col in feature_names:
            if col in inference_df_engineered.columns:
                X_inference[col] = inference_df_engineered[col]
            else:
                # Fill missing features with 0 or appropriate default
                X_inference[col] = 0
        
        # Convert categorical columns to numeric
        for col in X_inference.select_dtypes(include=["object", "category"]).columns:
            X_inference[col] = pd.Categorical(X_inference[col]).codes

        # Make prediction
        probabilities = clf.predict_proba(X_inference)[0]
        player1_prob = float(probabilities[0])
        player2_prob = float(probabilities[1])

        predicted_winner = request.player1 if player1_prob > player2_prob else request.player2

        # Get latest 5 matches for display
        matches_file = DATA_PATH / "matches_results.pkl"
        latest_matches = []
        
        if matches_file.exists():
            with open(matches_file, "rb") as f:
                matches_df = pickle.load(f)
            
            # Get matches involving either player
            player_matches = matches_df[
                (matches_df["winner_name"] == request.player1) | 
                (matches_df["loser_name"] == request.player1) |
                (matches_df["winner_name"] == request.player2) | 
                (matches_df["loser_name"] == request.player2)
            ]
            
            # Get latest 5 matches
            latest = player_matches.sort_values("tourney_date", ascending=False).head(5)
            
            for _, row in latest.iterrows():
                latest_matches.append({
                    "date": str(row["tourney_date"]),
                    "player1": row["winner_name"],
                    "player2": row["loser_name"],
                    "winner": row["winner_name"],
                    "tournament": row["tourney_name"],
                    "surface": row.get("surface"),
                    "score": row.get("score"),
                })

        return PredictResponse(
            player1=request.player1,
            player2=request.player2,
            tournament=request.tournament,
            player1_win_probability=player1_prob,
            player2_win_probability=player2_prob,
            predicted_winner=predicted_winner,
            latest_matches=latest_matches
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
