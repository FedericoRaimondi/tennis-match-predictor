from unittest.mock import Mock, patch

import pandas as pd

from match_predictor.utils import gh_utils


def test_list_github_files_success():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"name": "file1.csv"},
        {"name": "file2.csv"},
        {"name": "README.md"},
    ]
    with patch("requests.get", return_value=mock_response) as mock_get:
        files = gh_utils.list_github_files("owner/repo")
        assert files == ["file1.csv", "file2.csv", "README.md"]
        mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo/contents/")


def test_list_github_files_failure(capsys):
    mock_response = Mock()
    mock_response.status_code = 404
    with patch("requests.get", return_value=mock_response):
        files = gh_utils.list_github_files("owner/repo")
        assert files is None
        captured = capsys.readouterr()
        assert "Failed to retrieve files" in captured.out


def test_read_csv_from_github(monkeypatch):
    csv_content = "a,b\n1,2\n3,4"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response) as mock_get:
        df = gh_utils.read_csv_from_github("owner/repo", "file.csv")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert df.shape == (2, 2)
        mock_get.assert_called_once_with("https://raw.githubusercontent.com/owner/repo/master/file.csv", verify=True)
