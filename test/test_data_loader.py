import numpy as np
import pandas as pd
import pytest

from tennis_match_predictor.data.data_loader import DataLoader


@pytest.fixture
def sample_matches_df():
    data = {
        "tourney_id": ["2021-001", "2021-001"],
        "tourney_name": ["Test Open", "Test Open"],
        "surface": ["Hard", "Hard"],
        "draw_size": [32, 32],
        "tourney_level": ["A", "A"],
        "tourney_date": ["20210101", "20210101"],
        "match_num": [1, 2],
        "winner_id": [100, 101],
        "winner_seed": [1, 2],
        "winner_entry": ["", ""],
        "winner_name": ["Player A", "Player B"],
        "winner_hand": ["R", "L"],
        "winner_ht": [180, 185],
        "winner_ioc": ["USA", "ESP"],
        "winner_age": [25, 27],
        "winner_rank": [10, 20],
        "winner_rank_points": [2000, 1500],
        "w_ace": [5, 6],
        "w_df": [1, 2],
        "w_svpt": [50, 48],
        "w_1stIn": [30, 28],
        "w_1stWon": [20, 18],
        "w_2ndWon": [10, 9],
        "w_SvGms": [10, 10],
        "w_bpSaved": [2, 3],
        "w_bpFaced": [3, 4],
        "loser_id": [101, 100],
        "loser_seed": [2, 1],
        "loser_entry": ["", ""],
        "loser_name": ["Player B", "Player A"],
        "loser_hand": ["L", "R"],
        "loser_ht": [185, 180],
        "loser_ioc": ["ESP", "USA"],
        "loser_age": [27, 25],
        "loser_rank": [20, 10],
        "loser_rank_points": [1500, 2000],
        "l_ace": [3, 4],
        "l_df": [2, 1],
        "l_svpt": [45, 47],
        "l_1stIn": [25, 27],
        "l_1stWon": [15, 17],
        "l_2ndWon": [8, 9],
        "l_SvGms": [10, 10],
        "l_bpSaved": [1, 2],
        "l_bpFaced": [2, 3],
        "score": ["6-3 6-4", "7-5 6-2"],
        "best_of": [3, 3],
        "round": ["R32", "R16"],
        "minutes": [90, 95],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def data_loader(monkeypatch):
    dl = DataLoader("dummy/repo")
    monkeypatch.setattr(dl, "list_files", lambda: ["atp_matches_2021.csv"])
    monkeypatch.setattr("tennis_match_predictor.utils.gh_utils.read_csv_from_github", lambda repo, file: pd.DataFrame())
    return dl


def test_basic_matches_cleaning(sample_matches_df):
    dl = DataLoader("dummy/repo")
    cleaned = dl._basic_matches_cleaning(sample_matches_df.copy())
    assert "tourney_year" in cleaned.columns
    assert cleaned["tourney_year"].min() >= 1991
    assert all(cleaned["tourney_level"].isin(["G", "F", "M", "A"]))
    assert not cleaned["tourney_name"].str.contains("Laver Cup").any()


def test_get_player_stats_latest(sample_matches_df):
    dl = DataLoader("dummy/repo")
    cleaned = dl._basic_matches_cleaning(sample_matches_df.copy())
    stats = dl.get_player_stats(df=cleaned, latest=True)
    assert isinstance(stats, pd.DataFrame)
    assert "player_id" in stats.columns
    assert stats.groupby("player_id").size().max() == 1


def test_get_player_stats_all(sample_matches_df):
    dl = DataLoader("dummy/repo")
    cleaned = dl._basic_matches_cleaning(sample_matches_df.copy())
    stats = dl.get_player_stats(df=cleaned, latest=False)
    assert isinstance(stats, pd.DataFrame)
    assert "player_id" in stats.columns
    assert stats["player_id"].nunique() == 2


def test_get_tournament_info(sample_matches_df):
    dl = DataLoader("dummy/repo")
    cleaned = dl._basic_matches_cleaning(sample_matches_df.copy())
    info = dl.get_tournament_info(df=cleaned)
    assert isinstance(info, pd.DataFrame)
    assert "tourney_id" in info.columns
    assert info.shape[0] == 1


def test_get_ml_data(sample_matches_df, monkeypatch):
    dl = DataLoader("dummy/repo")
    cleaned = dl._basic_matches_cleaning(sample_matches_df.copy())
    monkeypatch.setattr(np.random, "randint", lambda a, b, size: np.zeros(size, dtype=int))
    ml_data = dl.get_ml_data(df=cleaned)
    assert isinstance(ml_data, pd.DataFrame)
    assert "player_1" in ml_data.columns
    assert "player_2" in ml_data.columns
    assert "winner" in ml_data.columns
    assert ml_data.shape[0] == cleaned.shape[0]


def test_basic_matches_cleaning_missing_column():
    dl = DataLoader("dummy/repo")
    df = pd.DataFrame({"tourney_name": ["Test Open"], "tourney_level": ["A"], "tourney_date": ["20210101"]})
    # Should raise KeyError because 'tourney_year' is not created if 'tourney_date' is missing or malformed
    with pytest.raises(KeyError):
        _ = dl._basic_matches_cleaning(df[["tourney_name", "tourney_level"]])


def test_get_player_stats_missing_columns():
    dl = DataLoader("dummy/repo")
    df = pd.DataFrame({"winner_id": [1], "loser_id": [2]})
    # Should raise KeyError because required columns are missing
    with pytest.raises(KeyError):
        _ = dl.get_player_stats(df=df)


def test_get_tournament_info_missing_columns():
    dl = DataLoader("dummy/repo")
    df = pd.DataFrame({"tourney_id": [1]})
    # Should raise KeyError because required columns are missing
    with pytest.raises(KeyError):
        _ = dl.get_tournament_info(df=df)


def test_get_ml_data_missing_columns():
    dl = DataLoader("dummy/repo")
    df = pd.DataFrame({"winner_id": [1], "loser_id": [2]})
    # Should raise KeyError because required columns are missing
    with pytest.raises(KeyError):
        _ = dl.get_ml_data(df=df)


def test_list_files_success(monkeypatch):
    dl = DataLoader("dummy/repo")
    monkeypatch.setattr(
        "tennis_match_predictor.data.data_loader.list_github_files", lambda repo: ["file1.csv", "file2.csv"]
    )
    files = dl.list_files()
    assert files == ["file1.csv", "file2.csv"]


def test_list_files_failure(monkeypatch):
    dl = DataLoader("dummy/repo")
    monkeypatch.setattr("tennis_match_predictor.data.data_loader.list_github_files", lambda repo: None)
    assert dl.list_files() is None


def test_load_matches_success(monkeypatch):
    # Simulate two CSV files, each with a small DataFrame
    dl = DataLoader("dummy/repo")
    monkeypatch.setattr(dl, "list_files", lambda: ["atp_matches_2021.csv", "atp_matches_2022.csv", "README.md"])
    df1 = pd.DataFrame(
        {
            "tourney_id": ["2021-001"],
            "tourney_name": ["Test Open"],
            "surface": ["Hard"],
            "draw_size": [32],
            "tourney_level": ["A"],
            "tourney_date": ["20210101"],
            "match_num": [1],
            "winner_id": [100],
            "winner_seed": [1],
            "winner_entry": [""],
            "winner_name": ["Player A"],
            "winner_hand": ["R"],
            "winner_ht": [180],
            "winner_ioc": ["USA"],
            "winner_age": [25],
            "winner_rank": [10],
            "winner_rank_points": [2000],
            "w_ace": [5],
            "w_df": [1],
            "w_svpt": [50],
            "w_1stIn": [30],
            "w_1stWon": [20],
            "w_2ndWon": [10],
            "w_SvGms": [10],
            "w_bpSaved": [2],
            "w_bpFaced": [3],
            "loser_id": [101],
            "loser_seed": [2],
            "loser_entry": [""],
            "loser_name": ["Player B"],
            "loser_hand": ["L"],
            "loser_ht": [185],
            "loser_ioc": ["ESP"],
            "loser_age": [27],
            "loser_rank": [20],
            "loser_rank_points": [1500],
            "l_ace": [3],
            "l_df": [2],
            "l_svpt": [45],
            "l_1stIn": [25],
            "l_1stWon": [15],
            "l_2ndWon": [8],
            "l_SvGms": [10],
            "l_bpSaved": [1],
            "l_bpFaced": [2],
            "score": ["6-3 6-4"],
            "best_of": [3],
            "round": ["R32"],
            "minutes": [90],
        }
    )
    df2 = df1.copy()
    df2["tourney_id"] = "2022-001"
    df2["tourney_date"] = "20220101"
    monkeypatch.setattr(
        "tennis_match_predictor.data.data_loader.read_csv_from_github",
        lambda repo, file: df1 if "2021" in file else df2,
    )
    result = dl.load_matches()
    # Should contain both years, cleaned, and have 'tourney_year' column
    assert isinstance(result, pd.DataFrame)
    assert "tourney_year" in result.columns
    assert set(result["tourney_year"]) == {2021, 2022}


def test_load_matches_no_files(monkeypatch):
    dl = DataLoader("dummy/repo")
    monkeypatch.setattr(dl, "list_files", lambda: None)
    with pytest.raises(ValueError):
        dl.load_matches()
