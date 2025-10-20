"""Streamlit application for tennis match prediction."""

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from loguru import logger

# Page configuration
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="wide", initial_sidebar_state="expanded")

# Constants
API_URL = "http://localhost:8000"
DATA_PATH = Path("data")


def load_player_names() -> Tuple[list[str], pd.DataFrame]:
    """Load available player names and ids from data."""
    try:
        matches_file = DATA_PATH / "matches_results.csv"
        if not matches_file.exists():
            return []

        matches_df = pd.read_csv(matches_file)

        # Get unique player names and ids
        winner_names = matches_df[["winner_name", "winner_id"]].rename(
            columns={"winner_name": "player_name", "winner_id": "player_id"}
        )
        loser_names = matches_df[["loser_name", "loser_id"]].rename(
            columns={"loser_name": "player_name", "loser_id": "player_id"}
        )
        all_players = pd.concat([winner_names, loser_names]).drop_duplicates().reset_index(drop=True)
        players = sorted(all_players["player_name"].unique())

        return players, all_players
    except Exception as e:
        logger.error(f"Error loading player names: {e}")
        return []


def load_tournament_names() -> list[str]:
    """Load available tournament names from data."""
    try:
        tournament_file = DATA_PATH / "tournament_info.csv"
        if not tournament_file.exists():
            return []

        tournament_df = pd.read_csv(tournament_file)
        tournament_df["tourney_name"] = tournament_df["tourney_name"].str.strip().upper()

        tournaments = sorted(tournament_df["tourney_name"].unique())
        return tournaments
    except Exception as e:
        logger.error(f"Error loading tournament names: {e}")
        return []


def predict_match(player1: int, player2: int, tournament: str) -> dict | None:
    """Call the API to predict match outcome."""
    try:
        response = requests.post(
            f"{API_URL}/predict_winner",
            json={"player1": player1, "player2": player2, "tournament": tournament},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the API server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def get_latest_matches(player: str | None = None, limit: int = 5) -> dict | None:
    """Get latest match statistics from API."""
    try:
        params = {"limit": limit}
        if player:
            params["player"] = player

        response = requests.get(f"{API_URL}/latest_matches", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to API for match data.")
        return None
    except Exception as e:
        st.warning(f"Failed to fetch matches: {e}")
        return None


def create_probability_chart(player1: str, player2: str, prob1: float, prob2: float) -> go.Figure:
    """Create a bar chart showing win probabilities."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[player1, player2],
                y=[prob1, prob2],
                text=[f"{prob1:.1%}", f"{prob2:.1%}"],
                textposition="auto",
                marker=dict(color=[prob1, prob2], colorscale="RdYlGn", cmin=0, cmax=1),
            )
        ]
    )

    fig.update_layout(
        title="Win Probability",
        yaxis_title="Probability",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        showlegend=False,
        height=400,
    )

    return fig


def display_match_stats(matches: list[dict]) -> None:
    """Display match statistics in a table."""
    if not matches:
        st.info("No match data available.")
        return

    df = pd.DataFrame(matches)
    # Format date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "date": "Date",
            "player1": "Player 1",
            "player2": "Player 2",
            "winner": "Winner",
            "tournament": "Tournament",
            "surface": "Surface",
            "score": "Score",
        },
    )


def create_head_to_head_chart(player1: str, player2: str) -> go.Figure | None:
    """Create head-to-head record visualization."""
    try:
        matches_file = DATA_PATH / "matches_results.pkl"
        if not matches_file.exists():
            return None

        with open(matches_file, "rb") as f:
            matches_df = pickle.load(f)

        # Filter head-to-head matches
        h2h = matches_df[
            ((matches_df["winner_name"] == player1) & (matches_df["loser_name"] == player2))
            | ((matches_df["winner_name"] == player2) & (matches_df["loser_name"] == player1))
        ]

        if h2h.empty:
            return None

        # Count wins
        player1_wins = len(h2h[h2h["winner_name"] == player1])
        player2_wins = len(h2h[h2h["winner_name"] == player2])

        # Create pie chart
        fig = go.Figure(data=[go.Pie(labels=[player1, player2], values=[player1_wins, player2_wins], hole=0.3)])

        fig.update_layout(title=f"Head-to-Head Record ({player1_wins}-{player2_wins})", height=300)

        return fig

    except Exception as e:
        logger.error(f"Error creating head-to-head chart: {e}")
        return None


def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🎾 Tennis Match Predictor")
    st.markdown("""
    Predict tennis match outcomes using machine learning. Select two players and a tournament
    to see win probabilities and relevant statistics.
    """)

    # Sidebar for inputs
    with st.sidebar:
        st.header("Match Configuration")

        # Load players and tournaments
        players, all_players = load_player_names()
        tournaments = load_tournament_names()

        if not players:
            st.error("No player data available. Please ensure data files are present.")
            return

        # Player selection
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox(
                "Player 1",
                options=players,
                index=0 if players else None,
                key="player1",
                help="Type to search for a player",
            )

        with col2:
            player2 = st.selectbox(
                "Player 2",
                options=[p for p in players if p != player1],
                index=0 if len(players) > 1 else None,
                key="player2",
                help="Type to search for a player",
            )

        # Tournament selection
        tournament = st.selectbox(
            "Tournament",
            options=tournaments if tournaments else ["Wimbledon"],
            index=0 if tournaments else None,
            key="tournament",
        )

        # Predict button
        predict_button = st.button("Predict Match", type="primary", width="stretch")
        player_1_id = all_players[all_players["player_name"] == player1]["player_id"].values[0] if player1 else None
        player_2_id = all_players[all_players["player_name"] == player2]["player_id"].values[0] if player2 else None

    # Main content area
    if predict_button and player1 and player2 and tournament:
        with st.spinner("Predicting match outcome..."):
            prediction = predict_match(player_1_id, player_2_id, tournament)

        if prediction:
            # Display prediction results
            st.success(f"**Predicted Winner:** {prediction['predicted_winner']}")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Probability chart
                fig = create_probability_chart(
                    player1, player2, prediction["player1_win_probability"], prediction["player2_win_probability"]
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                # Win probabilities as metrics
                st.metric(label=f"{player1} Win Probability", value=f"{prediction['player1_win_probability']:.1%}")
                st.metric(label=f"{player2} Win Probability", value=f"{prediction['player2_win_probability']:.1%}")

            # Head-to-head record
            st.subheader("Head-to-Head Record")
            h2h_fig = create_head_to_head_chart(player1, player2)
            if h2h_fig:
                st.plotly_chart(h2h_fig, width="stretch")
            else:
                st.info(f"No previous matches found between {player1} and {player2}")

    # Display latest matches
    st.subheader("Latest Matches")

    tab1, tab2, tab3 = st.tabs(["All Matches", player1 if player1 else "Player 1", player2 if player2 else "Player 2"])

    with tab1:
        matches_data = get_latest_matches(limit=10)
        if matches_data:
            display_match_stats(matches_data.get("matches", []))

    with tab2:
        if player1:
            matches_data = get_latest_matches(player=player1, limit=5)
            if matches_data:
                display_match_stats(matches_data.get("matches", []))

    with tab3:
        if player2:
            matches_data = get_latest_matches(player=player2, limit=5)
            if matches_data:
                display_match_stats(matches_data.get("matches", []))

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source:** [Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/)

    > Tennis data is provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
    """)


if __name__ == "__main__":
    main()
