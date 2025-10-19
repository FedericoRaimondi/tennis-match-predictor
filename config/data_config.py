"""Data configuration using Pydantic."""

from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""

    github_repo: str = Field(
        default="JeffSackmann/tennis_atp",
        description="GitHub repository for tennis data"
    )
    selected_year: int = Field(
        default=1991,
        description="Starting year for match data filtering"
    )
    tourney_levels: list[str] = Field(
        default=["G", "F", "M", "A"],
        description="Tournament levels to include (G=Grand Slam, F=Tour Finals, M=Masters, A=ATP Tour)"
    )


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    rolling_windows: list[int] = Field(
        default=[3, 5, 10],
        description="Window sizes for rolling statistics"
    )
    stats_columns_mean: list[str] = Field(
        default=[
            "p_ace",
            "p_df",
            "p_svpt",
            "p_1stIn",
            "p_1stWon",
            "p_2ndWon",
            "p_SvGms",
            "p_bpSaved",
            "p_bpFaced",
            "opponent_rank_points",
            "o_ace",
            "o_df",
            "o_svpt",
            "o_1stIn",
            "o_1stWon",
            "o_2ndWon",
            "o_SvGms",
            "o_bpSaved",
            "o_bpFaced",
            "minutes",
        ],
        description="Columns for which to calculate rolling mean"
    )
    stats_columns_sum: list[str] = Field(
        default=["minutes", "results"],
        description="Columns for which to calculate rolling sum"
    )
    elo_k_factor: float = Field(
        default=32.0,
        description="K-factor for ELO rating calculation"
    )
    elo_initial_rating: float = Field(
        default=1500.0,
        description="Initial ELO rating for new players"
    )


class DataConfig(BaseModel):
    """Main data configuration."""

    source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    inference_data_path: str = Field(
        default="data/",
        description="Path to store inference data"
    )
    matches_results_file: str = Field(
        default="matches_results.pkl",
        description="Filename for matches results data"
    )
    tournament_info_file: str = Field(
        default="tournament_info.pkl",
        description="Filename for tournament info data"
    )
