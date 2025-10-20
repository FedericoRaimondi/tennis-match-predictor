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
            "p_id",
            "o_id",
            "p_id_p2",
            "o_id_p2",
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "tourney_year",
            "match_num",
            "p_name",
            "o_name",
            "p_name_p2",
            "o_name_p2",
        ]

        # exclude datetime64[ns]
        date_columns = df.select_dtypes(include="datetime64[ns]").columns.tolist()

        # Also exclude columns with suffixes like _p1, _p2, _t for metadata columns
        exclude_patterns = ["tourney_id", "tourney_date", "match_num", "p_id", "o_id"]

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
            "p_id",
            "o_id",
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "match_num",
            "p_name",
            "o_name",
        ]

        # exclude datetime64[ns]
        date_columns = df.select_dtypes(include="datetime64[ns]").columns.tolist()

        # Also exclude columns with suffixes like _p1, _p2, _t for metadata columns
        exclude_patterns = ["tourney_id", "tourney_date", "match_num", "p_id", "o_id"]

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
