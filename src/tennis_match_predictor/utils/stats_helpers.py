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

    Parameters:
        df (pd.DataFrame): The input DataFrame
        player_id_col (str): Column name for player identification
        sort_cols (List[str]): Columns to sort by to ensure proper chronological order
        stats_columns (Union[str, List[str], None]): Column(s) to calculate rolling statistics for
        agg_type (str): Type of aggregation ('mean', 'sum', 'std', 'min', 'max', 'median')
        window (int): Number of periods for the rolling window
        min_periods (int): Minimum number of observations required to have a value
        shift_periods (int): Number of periods to shift (1 = exclude current match, 0 = include current match)

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


def calculate_elo():
    """Calculate Elo ratings for players based on match outcomes."""
    # TODO: Implement Elo rating calculation
    pass  # pragma: no cover
