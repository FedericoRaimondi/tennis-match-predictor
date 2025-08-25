"""Utility functions for interacting with GitHub repositories."""

from io import StringIO

import pandas as pd
import requests


def list_github_files(repo_name: str) -> list | None:
    """Lists files in a GitHub repository.

    Parameters:
        repo_name (str): The name of the GitHub repository (e.g., "owner/repo").

    Returns:
        list | None: A list of file names in the repository, or None if the request fails.
    """
    url = f"https://api.github.com/repos/{repo_name}/contents/"
    response = requests.get(url)
    if response.status_code == 200:
        files = response.json()
        file_names = [file["name"] for file in files]
        return file_names
    else:
        print(f"Failed to retrieve files: {response.status_code}")


def read_csv_from_github(repo_name: str, file_name: str) -> pd.DataFrame:
    """Reads a CSV file from a GitHub repository into a Pandas DataFrame.

    Parameters:
        repo_name (str): The name of the GitHub repository (e.g., "owner/repo").
        file_name (str): The name of the CSV file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.
    """
    file_url = f"https://raw.githubusercontent.com/{repo_name}/master/{file_name}"
    response = requests.get(file_url, verify=True)
    csv_content = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_content))
    return df
