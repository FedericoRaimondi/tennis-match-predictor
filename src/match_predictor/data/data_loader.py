"""Data loading utilities for tennis match prediction."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from match_predictor.config import DataConfig
from match_predictor.utils.gh_utils import list_github_files, read_csv_from_github
from match_predictor.utils.stats_helpers import add_rolling_stats, calculate_elo


class DataLoader:
    """Class for loading tennis match data from a GitHub repository."""

    def __init__(self, data_config: DataConfig):
        """Initializes the DataLoader with the specified GitHub repository name.

        Parameters:
            repo_name (str): The name of the GitHub repository (e.g., "owner/repo").
        """
        self.data_config = data_config
        self.repo_name = data_config.source.github_repo
        self.logger = logger

    def list_files(self) -> list | None:
        """Lists files in the specified GitHub repository.

        Returns:
            list | None: A list of file names in the repository, or None if the request fails.
        """
        return list_github_files(self.repo_name)

    def _basic_matches_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs basic cleaning on the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        self.logger.info("Performing basic cleaning on the DataFrame...")
        # Convert the 'tourney_date' column to datetime format
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
        # Create tournay_year column
        df["tourney_year"] = df["tourney_date"].dt.year
        # Filter for matches after selected year
        selected_year = self.data_config.source.selected_year
        df = df[df["tourney_year"] >= selected_year].reset_index(drop=True)
        # Filter for tourney levels in ['G', 'F', 'M', 'A'], for more info see the documentation
        df = df[df["tourney_level"].isin(self.data_config.source.tourney_levels)].reset_index(drop=True)
        # Exclude Laver Cup matches
        df = df[~df["tourney_name"].str.contains("Laver Cup")].reset_index(drop=True)

        return df

    def load_matches(self) -> pd.DataFrame:
        """Load the atp matches specified GitHub repository into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.
        """
        file_names = self.list_files()
        if file_names is None:
            self.logger.error("Failed to retrieve file list from the repository.")
            raise ValueError("Failed to retrieve file list from the repository.")

        # Filter files to only include CSV files
        file_names = [file for file in file_names if file.endswith(".csv")]

        self.logger.info(f"Starting to load data from '{self.repo_name}'...")
        # Filter files to only include those that follow the "atp_matches_YYYY" pattern
        atp_matches = [file for file in file_names if re.match(r"atp_matches_\d{4}\.csv", file)]

        # Read and aggregate ATP matches data
        atp_matches_dfs = [read_csv_from_github(self.repo_name, file) for file in atp_matches]
        atp_matches_df = pd.concat(atp_matches_dfs, ignore_index=True)

        # Perform basic cleaning
        atp_matches_df = self._basic_matches_cleaning(atp_matches_df)

        self.logger.info("Data loading complete.")

        return atp_matches_df

    def get_player_stats(self, df: pd.DataFrame = None, latest: bool = False) -> pd.DataFrame:
        """Get statistics for the whole players dataset.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing match data. If None, loads matches from GitHub.
            latest (bool): If True, return only the latest statistics for each player.

        Returns:
            pd.DataFrame: A DataFrame containing the players' statistics.
        """
        self.logger.info("Calculating player statistics...")
        # Define relevant columns to keep
        cols = [
            "tourney_id",
            "tourney_name",
            "surface",
            "draw_size",
            "tourney_level",
            "tourney_date",
            "tourney_year",
            "match_num",  # tournament info
            "winner_id",
            "winner_seed",
            "winner_entry",
            "winner_name",
            "winner_hand",
            "winner_ht",
            "winner_ioc",
            "winner_age",
            "winner_rank",
            "winner_rank_points",  # winner info
            "w_ace",
            "w_df",
            "w_svpt",
            "w_1stIn",
            "w_1stWon",
            "w_2ndWon",
            "w_SvGms",
            "w_bpSaved",
            "w_bpFaced",  # winner stats
            "loser_id",
            "loser_seed",
            "loser_entry",
            "loser_name",
            "loser_hand",
            "loser_ht",
            "loser_ioc",
            "loser_age",
            "loser_rank",
            "loser_rank_points",  # loser info
            "l_ace",
            "l_df",
            "l_svpt",
            "l_1stIn",
            "l_1stWon",
            "l_2ndWon",
            "l_SvGms",
            "l_bpSaved",
            "l_bpFaced",
            "score",
            "best_of",
            "round",
            "minutes",  # match info
        ]  # from config ?

        if df is None:
            atp_matches_df = self.load_matches()  # pragma: no cover
        else:
            atp_matches_df = df.copy()
        # create a dataframe with only the relevant columns
        player_stats_hist_df = atp_matches_df[cols].copy()
        player_stats_hist_df_l = atp_matches_df[cols].copy()  # loser stats dataframe

        # add a column to flag winner and loser
        player_stats_hist_df["results"] = 1
        player_stats_hist_df_l["results"] = 0

        # Rename columns to remove "winner_" prefix and "w_" prefix only if they start the column name
        player_stats_hist_df.columns = [
            col.replace("winner_", "p_", 1)
            if col.startswith("winner_")
            else col.replace("w_", "p_", 1)
            if col.startswith("w_")
            else col.replace("loser_", "o_", 1)
            if col.startswith("loser_")
            else col.replace("l_", "o_", 1)
            if col.startswith("l_")
            else col
            for col in player_stats_hist_df.columns
        ]

        player_stats_hist_df_l.columns = [
            col.replace("loser_", "p_", 1)
            if col.startswith("loser_")
            else col.replace("l_", "p_", 1)
            if col.startswith("l_")
            else col.replace("winner_", "o_", 1)
            if col.startswith("winner_")
            else col.replace("w_", "o_", 1)
            if col.startswith("w_")
            else col
            for col in player_stats_hist_df_l.columns
        ]

        # Append the loser stats to the winner stats
        player_stats_hist_df = pd.concat([player_stats_hist_df, player_stats_hist_df_l], ignore_index=True)

        # First ensure the DataFrame is properly sorted by the specified order
        player_stats_hist_df = player_stats_hist_df.sort_values(
            by=["p_id", "tourney_date", "tourney_id", "match_num"]
        ).reset_index(drop=True)

        cols_to_mean = self.data_config.features.stats_columns_mean
        cols_to_sum = self.data_config.features.stats_columns_sum

        # add p_ and o_ prefixes to cols_to_mean, except for minutes
        cols_to_mean = [col if col == "minutes" else f"p_{col}" for col in cols_to_mean] + [
            col if col == "minutes" else f"o_{col}" for col in cols_to_mean
        ]
        # cols_to_mean = [
        #     "p_ace",
        #     "p_df",
        #     "p_svpt",
        #     "p_1stIn",
        #     "p_1stWon",
        #     "p_2ndWon",
        #     "p_SvGms",
        #     "p_bpSaved",
        #     "p_bpFaced",  # player stats
        #     "opponent_rank_points",  # opponent info
        #     "o_ace",
        #     "o_df",
        #     "o_svpt",
        #     "o_1stIn",
        #     "o_1stWon",
        #     "o_2ndWon",
        #     "o_SvGms",
        #     "o_bpSaved",
        #     "o_bpFaced",  # opponent stats
        #     "minutes",  # match info
        # ]

        # cols_to_sum = [
        #     "minutes",  # match info
        #     "results",  # match wins
        # ]

        self.logger.info("Calculating rolling statistics for players...")
        for i in self.data_config.features.rolling_windows:
            # Add rolling statistics for the last i matches
            player_stats_hist_df = add_rolling_stats(
                player_stats_hist_df, stats_columns=cols_to_mean, agg_type="mean", window=i
            )

            player_stats_hist_df = add_rolling_stats(
                player_stats_hist_df, stats_columns=cols_to_sum, agg_type="sum", window=i
            )

        # add elo rating
        self.logger.info("Calculating elo ratings...")
        player_stats_hist_df = calculate_elo(player_stats_hist_df)

        # drop original cols_to_mean and cols_to_sum to avoid data leakage
        player_stats_hist_df = player_stats_hist_df.drop(columns=cols_to_mean + cols_to_sum)

        # drop opponent info to avoid double reporting
        to_remove = ["id", "seed", "entry", "name", "hand", "ht", "ioc", "age", "rank"]
        to_remove = [f"o_{col}" for col in to_remove]
        player_stats_hist_df = player_stats_hist_df.drop(columns=to_remove)

        # If latest is True, return only the latest statistics for each player
        if latest:
            self.logger.info("Extracting latest statistics for each player...")
            player_stats_hist_df = player_stats_hist_df.sort_values(
                by=["p_id", "tourney_date", "tourney_id", "match_num"]
            ).reset_index(drop=True)
            player_stats_hist_df = player_stats_hist_df.groupby("p_id").tail(1).reset_index(drop=True)
            to_remove = [
                "tourney_id",
                "tourney_name",
                "surface",
                "draw_size",
                "tourney_level",
                "tourney_date",
                "tourney_year",
                "match_num",
            ]
            player_stats_hist_df = player_stats_hist_df.drop(columns=to_remove)

        self.logger.info("Player statistics calculation complete.")
        return player_stats_hist_df

    def get_tournament_info(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get basic information about tournaments from the matches dataset.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing match data. If None, loads matches from GitHub.

        Returns:
            pd.DataFrame: A DataFrame containing basic tournament information.
        """
        if df is None:
            atp_matches_df = self.load_matches()  # pragma: no cover
        else:
            atp_matches_df = df.copy()
        tournament_info_cols = [
            "tourney_id",
            "tourney_name",
            "surface",
            "draw_size",
            "tourney_level",
            "tourney_date",
            "tourney_year",
        ]
        tournament_info_df = atp_matches_df[tournament_info_cols]
        tournament_info_df["tourney_name"] = tournament_info_df["tourney_name"].str.strip().upper()
        # sort by tourney_date descending and drop duplicates to keep only the latest info
        tournament_info_df = tournament_info_df.sort_values(by="tourney_date", ascending=False)
        # drop tourney id, date, year columns
        tournament_info_df = tournament_info_df.drop(columns=["tourney_id", "tourney_date", "tourney_year"])
        # keep only the first occurrence of each tournament name. Basically latest info for each tournament.
        tournament_info_df = tournament_info_df.drop_duplicates(subset=["tourney_name"]).reset_index(drop=True)
        self.logger.info(f"Loaded info for {tournament_info_df.shape[0]} tournaments.")

        return tournament_info_df

    def get_ml_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare and return the dataset for machine learning tasks.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing match data. If None, loads matches from GitHub.

        Returns:
            pd.DataFrame: A DataFrame ready for machine learning tasks.
        """
        if df is None:
            atp_matches_df = self.load_matches()  # pragma: no cover
        else:
            atp_matches_df = df.copy()

        self.logger.info("Preparing dataset for machine learning tasks...")
        # Create a new column with a list of [winner_id, loser_id] for each row
        atp_matches_df["p_ids"] = atp_matches_df.apply(lambda row: [row["winner_id"], row["loser_id"]], axis=1)

        # Randomly select one as player_1 and the other as player_2
        rand_choice = np.random.randint(0, 2, size=len(atp_matches_df))
        atp_matches_df["player_1"] = atp_matches_df.apply(lambda row: row["p_ids"][rand_choice[row.name]], axis=1)
        atp_matches_df["player_2"] = atp_matches_df.apply(lambda row: row["p_ids"][1 - rand_choice[row.name]], axis=1)

        # Create a column "winner": 0 if player_1 is the first element of p_ids, else 1
        atp_matches_df["winner"] = atp_matches_df.apply(
            lambda row: 0 if row["player_1"] == row["p_ids"][0] else 1, axis=1
        )
        ml_dataset = atp_matches_df[
            ["tourney_id", "tourney_date", "match_num", "player_1", "player_2", "winner"]
        ].copy()

        # left join with player_stats_hist_df
        ml_dataset = ml_dataset.merge(
            self.get_player_stats(df=atp_matches_df, latest=False),
            how="left",
            left_on=["player_1", "tourney_id", "tourney_date", "match_num"],
            right_on=["p_id", "tourney_id", "tourney_date", "match_num"],
            suffixes=("", "_p1"),
        )

        # add also p2 stats
        ml_dataset = ml_dataset.merge(
            self.get_player_stats(df=atp_matches_df, latest=False),
            how="left",
            left_on=["player_2", "tourney_id", "tourney_date", "match_num"],
            right_on=["p_id", "tourney_id", "tourney_date", "match_num"],
            suffixes=("", "_p2"),
        )
        to_remove = [
            "tourney_id",
            "tourney_name",
            "surface",
            "draw_size",
            "tourney_level",
            "tourney_date",
            "tourney_year",
            "match_num",
        ]
        # add _p1 and _p2 prefixes to to_remove cols
        to_remove = [f"{col}_p1" for col in to_remove] + [f"{col}_p2" for col in to_remove]
        ml_dataset = ml_dataset.drop(columns=to_remove)

        # # add tournament info
        # ml_dataset = ml_dataset.merge(
        #     self.get_tournament_info(df=atp_matches_df),
        #     how="left",
        #     left_on="tourney_id",
        #     right_on="tourney_id",
        #     suffixes=("", "_t"),
        # )

        self.logger.info("Dataset preparation complete.")
        self.logger.info(f"Final dataset shape: {ml_dataset.shape}")

        return ml_dataset

    def save_latest_player_stats(
        self, df: pd.DataFrame = None, output_path: str = "data/player_stats_latest.csv"
    ) -> Path:
        """Save the latest player statistics to a file for quick access during inference.

        This method extracts and saves the most recent performance statistics for each player,
        making them available for feature generation during model inference.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing match data. If None, loads matches from GitHub.
            output_path (str): Path where the player stats will be saved. Defaults to "data/player_stats_latest.csv".
                Can be .csv or .parquet format based on the file extension.

        Returns:
            Path: The path to the saved file.

        Examples:
            >>> data_loader = DataLoader("JeffSackmann/tennis_atp")
            >>> stats_path = data_loader.save_latest_player_stats()
            >>> print(f"Stats saved to: {stats_path}")
        """
        self.logger.info("Extracting and saving latest player statistics...")

        # Get the latest stats for each player
        player_stats = self.get_player_stats(df=df, latest=True)

        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save based on file extension
        if output_path.endswith(".parquet"):
            player_stats.to_parquet(output_file, index=False)
            self.logger.info(f"Latest player stats saved to {output_file} (parquet format)")
        else:
            player_stats.to_csv(output_file, index=False)
            self.logger.info(f"Latest player stats saved to {output_file} (csv format)")

        self.logger.info(f"Saved stats for {len(player_stats)} players")

        return output_file


# if __name__ == "__main__":
#     # Example usage
#     repo_name = "JeffSackmann/tennis_atp"
#     data_loader = DataLoader(repo_name)
#     matches_df = data_loader.load_matches()
#     print(matches_df.head())
#     player_stats_df = data_loader.get_player_stats(latest=True)
#     print(player_stats_df.head())
#     tournament_info_df = data_loader.get_tournament_info()
#     print(tournament_info_df.head())
#     ml_data_df = data_loader.get_ml_data()
#     print(ml_data_df.head())
