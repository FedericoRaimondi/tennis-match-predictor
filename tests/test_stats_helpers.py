import pandas as pd
import pytest

from match_predictor.utils import stats_helpers


@pytest.fixture
def sample_df():
    # Create a simple DataFrame for rolling stats
    data = {
        "player_id": [1, 1, 1, 2, 2, 2],
        "tourney_date": pd.to_datetime(
            ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-01", "2021-01-02", "2021-01-03"]
        ),
        "tourney_id": ["A", "A", "A", "B", "B", "B"],
        "match_num": [1, 2, 3, 1, 2, 3],
        "aces": [5, 10, 15, 2, 4, 6],
        "double_faults": [1, 2, 3, 0, 1, 2],
    }
    return pd.DataFrame(data)


def test_add_rolling_stats_mean(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="mean", window=2, min_periods=1)
    assert "aces_mean_last2" in df.columns
    # First row for each player should be NA
    assert pd.isna(df.loc[0, "aces_mean_last2"])
    assert pd.isna(df.loc[3, "aces_mean_last2"])
    # Second row for player 1: mean of first value (5)
    assert df.loc[1, "aces_mean_last2"] == pytest.approx(5.0)
    # Third row for player 1: mean of [5,10]
    assert df.loc[2, "aces_mean_last2"] == pytest.approx(7.5)


def test_add_rolling_stats_sum(sample_df):
    df = stats_helpers.add_rolling_stats(
        sample_df, stats_columns="double_faults", agg_type="sum", window=2, min_periods=1
    )
    assert "double_faults_sum_last2" in df.columns
    # Second row for player 2: sum of first value (0)
    assert df.loc[4, "double_faults_sum_last2"] == pytest.approx(0.0)
    # Third row for player 2: sum of [0,1]
    assert df.loc[5, "double_faults_sum_last2"] == pytest.approx(1.0)


def test_add_rolling_stats_multiple_columns(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns=["aces", "double_faults"], agg_type="mean", window=2)
    assert "aces_mean_last2" in df.columns
    assert "double_faults_mean_last2" in df.columns


def test_add_rolling_stats_invalid_column(sample_df, capsys):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="not_a_column", agg_type="mean", window=2)
    captured = capsys.readouterr()
    assert "Warning: Column 'not_a_column' not found in DataFrame" in captured.out


def test_add_rolling_stats_invalid_agg_type(sample_df):
    with pytest.raises(ValueError):
        stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="not_a_real_agg", window=2)


def test_add_rolling_stats_custom_group_and_sort():
    data = {
        "team": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01", "2021-01-02"]),
        "score": [1, 2, 3, 4],
        "game": [1, 2, 1, 2],
    }
    df = pd.DataFrame(data)
    result = stats_helpers.add_rolling_stats(
        df, stats_columns="score", agg_type="sum", window=2, group_col="team", sort_cols=["team", "date", "game"]
    )
    assert "score_sum_last2" in result.columns
    assert pd.isna(result.loc[0, "score_sum_last2"])
    assert result.loc[1, "score_sum_last2"] == 1


def test_add_rolling_stats_std(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="std", window=2, min_periods=1)
    assert "aces_std_last2" in df.columns
    # First row for each player should be NA
    assert pd.isna(df.loc[0, "aces_std_last2"])
    assert pd.isna(df.loc[3, "aces_std_last2"])
    # Second row for player 1: std of [5] is nan, but pandas returns nan for std with 1 value
    assert pd.isna(df.loc[1, "aces_std_last2"])
    # Third row for player 1: std of [5,10]
    assert df.loc[2, "aces_std_last2"] == pytest.approx(3.5355339, rel=1e-4)


def test_add_rolling_stats_min(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="min", window=2, min_periods=1)
    assert "aces_min_last2" in df.columns
    assert pd.isna(df.loc[0, "aces_min_last2"])
    assert pd.isna(df.loc[3, "aces_min_last2"])
    assert df.loc[1, "aces_min_last2"] == 5
    assert df.loc[2, "aces_min_last2"] == 5


def test_add_rolling_stats_max(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="max", window=2, min_periods=1)
    assert "aces_max_last2" in df.columns
    assert pd.isna(df.loc[0, "aces_max_last2"])
    assert pd.isna(df.loc[3, "aces_max_last2"])
    assert df.loc[1, "aces_max_last2"] == 5
    assert df.loc[2, "aces_max_last2"] == 10


def test_add_rolling_stats_median(sample_df):
    df = stats_helpers.add_rolling_stats(sample_df, stats_columns="aces", agg_type="median", window=2, min_periods=1)
    assert "aces_median_last2" in df.columns
    assert pd.isna(df.loc[0, "aces_median_last2"])
    assert pd.isna(df.loc[3, "aces_median_last2"])
    assert df.loc[1, "aces_median_last2"] == 5
    assert df.loc[2, "aces_median_last2"] == pytest.approx(7.5)


@pytest.fixture
def sample_elo_df():
    # Create a simple DataFrame for ELO testing
    data = {
        "player_id": [1, 2, 1, 2, 1],
        "opponent_id": [2, 1, 2, 1, 2],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]),
        "tourney_id": ["A", "A", "A", "A", "A"],
        "match_num": [1, 2, 3, 4, 5],
        "results": [1, 0, 1, 1, 0],
    }
    return pd.DataFrame(data)


def test_calculate_elo_basic(sample_elo_df):
    df = stats_helpers.calculate_elo(sample_elo_df)
    assert "elo_rating" in df.columns
    # First match: both players start at 1500
    assert df.loc[0, "elo_rating"] == 1500.0
    # ELO should change after matches
    assert df.loc[1, "elo_rating"] != 1500.0


def test_calculate_elo_winner_increases():
    data = {
        "player_id": [1, 1],
        "opponent_id": [2, 2],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "tourney_id": ["A", "A"],
        "match_num": [1, 2],
        "results": [1, 1],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df)

    # Player 1 wins both matches, so their ELO before second match should be higher than 1500
    assert result_df.loc[1, "elo_rating"] > 1500.0


def test_calculate_elo_custom_k_factor():
    data = {
        "player_id": [1, 2],
        "opponent_id": [2, 1],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "tourney_id": ["A", "A"],
        "match_num": [1, 2],
        "results": [1, 0],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df, k=64)

    assert "elo_rating" in result_df.columns
    # With higher K-factor, changes should be more dramatic
    elo_change = abs(result_df.loc[0, "elo_rating"] - 1500.0)
    assert elo_change == 0  # First match starts at base


def test_calculate_elo_custom_base_elo():
    data = {
        "player_id": [1, 2],
        "opponent_id": [2, 1],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "tourney_id": ["A", "A"],
        "match_num": [1, 2],
        "results": [1, 0],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df, base_elo=2000)

    # First player should start at custom base ELO
    assert result_df.loc[0, "elo_rating"] == 2000.0


def test_calculate_elo_consistent_updates(sample_elo_df):
    df = stats_helpers.calculate_elo(sample_elo_df)

    # ELO ratings should be calculated for all matches
    assert df["elo_rating"].notna().all()


def test_calculate_elo_multiple_players():
    data = {
        "player_id": [1, 2, 3, 1, 2, 3],
        "opponent_id": [2, 3, 1, 3, 1, 2],
        "tourney_date": pd.to_datetime(
            ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06"]
        ),
        "tourney_id": ["A", "A", "A", "A", "A", "A"],
        "match_num": [1, 2, 3, 4, 5, 6],
        "results": [1, 1, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df)

    # All players should start at base ELO
    assert result_df.loc[0, "elo_rating"] == 1500.0
    assert result_df.loc[1, "elo_rating"] == 1500.0
    assert result_df.loc[2, "elo_rating"] == 1500.0


def test_calculate_elo_custom_columns():
    data = {
        "p1": [1, 2],
        "p2": [2, 1],
        "tourney_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "tourney_id": ["A", "A"],
        "match_num": [1, 2],
        "win": [1, 0],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df, player_col="p1", opponent_col="p2", result_col="win")

    assert "elo_rating" in result_df.columns
    assert result_df.loc[0, "elo_rating"] == 1500.0


def test_calculate_elo_preserves_original_columns(sample_elo_df):
    original_columns = set(sample_elo_df.columns)
    result_df = stats_helpers.calculate_elo(sample_elo_df)

    # All original columns should still be present
    for col in original_columns:
        assert col in result_df.columns


def test_calculate_elo_sorted_chronologically():
    # Create unsorted data
    data = {
        "player_id": [1, 1, 1],
        "opponent_id": [2, 2, 2],
        "tourney_date": pd.to_datetime(["2021-01-03", "2021-01-01", "2021-01-02"]),
        "tourney_id": ["A", "A", "A"],
        "match_num": [3, 1, 2],
        "results": [1, 0, 1],
    }
    df = pd.DataFrame(data)
    result_df = stats_helpers.calculate_elo(df)

    # ELO should be calculated chronologically
    # First chronological match should start at base ELO
    first_match_idx = result_df.sort_values(["player_id", "tourney_date", "tourney_id", "match_num"]).index[0]
    assert result_df.loc[first_match_idx, "elo_rating"] == 1500.0
