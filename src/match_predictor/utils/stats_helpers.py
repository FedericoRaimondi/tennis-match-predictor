"""Helper functions for calculating statistics on tennis match data."""

from typing import List, Union

import pandas as pd


def add_rolling_stats(
    df: pd.DataFrame,
    stats_columns: Union[str, List[str]],
    agg_type: str = "mean",
    window: int = 5,
    min_periods: int = 1,
    shift_periods: int = 1,
    group_col: str = "player_id",
    sort_cols: List[str] = ["player_id", "tourney_date", "tourney_id", "match_num"],  # noqa: B006
) -> pd.DataFrame:
    """Add rolling statistics to a DataFrame for each player.

    Args:
        df: Input DataFrame containing match data.
        stats_columns: Column name or list of column names to calculate rolling statistics on.
        agg_type: Type of aggregation ('mean', 'sum', 'std', 'min', 'max', 'median').
        window: Size of the rolling window.
        min_periods: Minimum number of observations in window required to have a value.
        shift_periods: Number of periods to shift the result (to avoid data leakage).
        group_col: Column name to group by (e.g., player identifier).
        sort_cols: List of columns to sort by for chronological order.

    Returns:
        pd.DataFrame: DataFrame with added rolling statistics columns
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Ensure proper sorting
    result_df = result_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Convert single column to list
    if isinstance(stats_columns, str):
        stats_columns = [stats_columns]

    # Calculate rolling statistics for each column
    for col in stats_columns:
        if col in result_df.columns:
            # Create column name for the new statistic
            new_col_name = f"{col}_{agg_type}_last{window}"

            # Calculate rolling statistic
            rolling_obj = result_df.groupby(group_col)[col].rolling(window=window, min_periods=min_periods)

            # Apply the specified aggregation
            if agg_type == "mean":
                rolling_stat = rolling_obj.mean()
            elif agg_type == "sum":
                rolling_stat = rolling_obj.sum()
            elif agg_type == "std":
                rolling_stat = rolling_obj.std()
            elif agg_type == "min":
                rolling_stat = rolling_obj.min()
            elif agg_type == "max":
                rolling_stat = rolling_obj.max()
            elif agg_type == "median":
                rolling_stat = rolling_obj.median()
            else:
                raise ValueError(f"Unsupported aggregation type: {agg_type}")

            # Apply shift and add to DataFrame
            result_df[new_col_name] = rolling_stat.shift(shift_periods).reset_index(level=0, drop=True)

            # Set first record for each player to NaN
            first_records = result_df.groupby(group_col).head(1).index
            result_df.loc[first_records, new_col_name] = pd.NA
        else:
            print(f"Warning: Column '{col}' not found in DataFrame")

    return result_df


def calculate_elo(
    df: pd.DataFrame,
    player_col: str = "player_id",
    opponent_col: str = "opponent_id",
    result_col: str = "results",
    k: float = 32,
    base_elo: float = 1500,
    sort_cols: list = ["player_id", "tourney_date", "tourney_id", "match_num"],  # noqa: B006
) -> pd.DataFrame:
    """Calculate Elo ratings for players based on match outcomes.

    Args:
        df (pd.DataFrame): DataFrame with at least player_id, opponent_id, results, and match order columns.
        player_col (str): Column name for player ID.
        opponent_col (str): Column name for opponent ID.
        result_col (str): Column name for match result (1=win, 0=loss).
        k (float): Elo K-factor.
        base_elo (float): Starting Elo for new players.
        sort_cols (list): Columns to sort by for chronological order.

    Returns:
        pd.DataFrame: DataFrame with an added 'elo_before' and 'elo_after' column for each match.
    """
    df = df.copy()
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Store Elo ratings
    elo_dict = {}

    elo_before = []
    elo_after = []

    for idx, row in df.iterrows():
        p1 = row[player_col]
        p2 = row[opponent_col]
        result = row[result_col]

        # Get current Elo or assign base
        elo_p1 = elo_dict.get(p1, base_elo)
        elo_p2 = elo_dict.get(p2, base_elo)

        # Expected score
        expected_p1 = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))

        # Update Elo
        new_elo_p1 = elo_p1 + k * (result - expected_p1)
        new_elo_p2 = elo_p2 + k * ((1 - result) - (1 - expected_p1))

        elo_before.append(elo_p1)
        elo_after.append(new_elo_p1)

        # Save new ratings
        elo_dict[p1] = new_elo_p1
        elo_dict[p2] = new_elo_p2

    df["elo_rating"] = elo_before
    # df["elo_after"] = elo_after
    return df
