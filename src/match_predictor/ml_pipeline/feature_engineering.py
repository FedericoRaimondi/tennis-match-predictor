"""Feature engineering module for ML pipeline."""

import pandas as pd
from loguru import logger

from match_predictor.config import DataConfig


class FeatureEngineer:
    """Class for feature engineering operations."""

    def __init__(self, config: DataConfig | None = None):
        """Initialize feature engineer.

        Args:
            config: Data configuration (optional, defaults to DataConfig())
        """
        self.config = config or DataConfig()
        self.logger = logger

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations.

        Args:
            df: Input dataframe with match data

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")

        # Create a copy to avoid modifying the original
        df_engineered = df.copy()

        # Add derived features
        df_engineered = self._add_match_statistics(df_engineered)
        df_engineered = self._add_player_rankings(df_engineered)
        df_engineered = self._add_temporal_features(df_engineered)

        self.logger.info(f"Feature engineering complete. Final shape: {df_engineered.shape}")
        return df_engineered

    def _add_match_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add match-level statistics."""
        # Example: Add service percentage features if available
        if "p_1stIn" in df.columns and "p_svpt" in df.columns:
            df["p_1st_serve_pct"] = df["p_1stIn"] / df["p_svpt"].replace(0, 1)

        if "p_1stWon" in df.columns and "p_1stIn" in df.columns:
            df["p_1st_serve_win_pct"] = df["p_1stWon"] / df["p_1stIn"].replace(0, 1)

        # Break point conversion
        if "p_bpSaved" in df.columns and "p_bpFaced" in df.columns:
            df["p_bp_saved_pct"] = df["p_bpSaved"] / df["p_bpFaced"].replace(0, 1)

        return df

    def _add_player_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on player rankings."""
        # Example: Rank difference between players
        if "player_rank" in df.columns and "opponent_rank" in df.columns:
            df["rank_difference"] = df["player_rank"] - df["opponent_rank"]

        if "player_rank_points" in df.columns and "opponent_rank_points" in df.columns:
            df["rank_points_ratio"] = df["player_rank_points"] / df["opponent_rank_points"].replace(0, 1)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"])
            df["month"] = df["tourney_date"].dt.month
            df["quarter"] = df["tourney_date"].dt.quarter
            df["day_of_year"] = df["tourney_date"].dt.dayofyear

        return df

    def prepare_features_for_training(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training.

        Args:
            df: Input dataframe with engineered features

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        self.logger.info("Preparing features for training...")

        # Identify feature columns (exclude target and metadata)
        exclude_columns = [
            "winner",
            "player_1",
            "player_2",
            "player_id",
            "opponent_id",
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "match_num",
            "player_name",
            "opponent_name",
        ]

        # exclude datetime64[ns]
        date_columns = df.select_dtypes(include="datetime64[ns]").columns.tolist()

        # Also exclude columns with suffixes like _p1, _p2, _t for metadata columns
        exclude_patterns = ["tourney_id", "tourney_date", "match_num", "player_id", "opponent_id"]
        
        feature_columns = []
        for col in df.columns:
            # Skip if column is in exclude list
            if col in exclude_columns + date_columns:
                continue
            # Skip if column matches any exclusion pattern with suffix
            if any(col.startswith(pattern + "_") for pattern in exclude_patterns):
                continue
            feature_columns.append(col)

        # Check if target exists
        if "winner" not in df.columns:
            raise ValueError("Target column 'winner' not found in dataframe")

        X = df[feature_columns].copy()
        y = df["winner"].copy()

        # Handle missing values
        # X = X.fillna(X.median())

        # Convert categorical columns to numeric if needed
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = pd.Categorical(X[col]).codes

        self.logger.info(f"Prepared {len(feature_columns)} features for training")
        return X, y

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature names that will be used for training.

        Args:
            df: Input dataframe

        Returns:
            List of feature names
        """
        exclude_columns = [
            "winner",
            "player_1",
            "player_2",
            "player_id",
            "opponent_id",
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "match_num",
            "player_name",
            "opponent_name",
        ]

        # exclude datetime64[ns]
        date_columns = df.select_dtypes(include="datetime64[ns]").columns.tolist()

        # Also exclude columns with suffixes like _p1, _p2, _t for metadata columns
        exclude_patterns = ["tourney_id", "tourney_date", "match_num", "player_id", "opponent_id"]
        
        feature_columns = []
        for col in df.columns:
            # Skip if column is in exclude list
            if col in exclude_columns + date_columns:
                continue
            # Skip if column matches any exclusion pattern with suffix
            if any(col.startswith(pattern + "_") for pattern in exclude_patterns):
                continue
            feature_columns.append(col)

        return feature_columns
