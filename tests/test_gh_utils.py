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


def test_list_github_files_empty_repo():
    """Test listing files in empty repository."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch("requests.get", return_value=mock_response):
        files = gh_utils.list_github_files("owner/empty-repo")
        assert files == []


def test_list_github_files_server_error(capsys):
    """Test listing files with server error."""
    mock_response = Mock()
    mock_response.status_code = 500
    with patch("requests.get", return_value=mock_response):
        files = gh_utils.list_github_files("owner/repo")
        assert files is None
        captured = capsys.readouterr()
        assert "500" in captured.out


def test_read_csv_from_github_with_headers():
    """Test reading CSV with various column headers."""
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response):
        df = gh_utils.read_csv_from_github("owner/repo", "data.csv")
        assert list(df.columns) == ["name", "age", "city"]
        assert df.shape == (2, 3)
        assert df.loc[0, "name"] == "Alice"


def test_read_csv_from_github_with_special_characters():
    """Test reading CSV with special characters."""
    csv_content = "player,score\nTest,Player,100\nAnother,Player,200"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response):
        df = gh_utils.read_csv_from_github("owner/repo", "scores.csv")
        assert isinstance(df, pd.DataFrame)
        assert "player" in df.columns


def test_read_csv_from_github_empty_file():
    """Test reading empty CSV file."""
    csv_content = "a,b\n"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response):
        df = gh_utils.read_csv_from_github("owner/repo", "empty.csv")
        assert df.shape[0] == 0
        assert list(df.columns) == ["a", "b"]


def test_list_github_files_various_file_types():
    """Test listing files with various file types."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"name": "data.csv"},
        {"name": "script.py"},
        {"name": "README.md"},
        {"name": "config.yaml"},
    ]
    with patch("requests.get", return_value=mock_response):
        files = gh_utils.list_github_files("owner/repo")
        assert len(files) == 4
        assert "data.csv" in files
        assert "script.py" in files


def test_read_csv_from_github_utf8_encoding():
    """Test reading CSV with UTF-8 encoding."""
    csv_content = "name,description\nTest,Café"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response):
        df = gh_utils.read_csv_from_github("owner/repo", "utf8.csv")
        assert df.loc[0, "description"] == "Café"


def test_list_github_files_url_construction():
    """Test URL construction for list_github_files."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch("requests.get", return_value=mock_response) as mock_get:
        gh_utils.list_github_files("test-owner/test-repo")
        expected_url = "https://api.github.com/repos/test-owner/test-repo/contents/"
        mock_get.assert_called_once_with(expected_url)


def test_read_csv_from_github_url_construction():
    """Test URL construction for read_csv_from_github."""
    csv_content = "a,b\n1,2"
    mock_response = Mock()
    mock_response.content = csv_content.encode("utf-8")
    with patch("requests.get", return_value=mock_response) as mock_get:
        gh_utils.read_csv_from_github("test-owner/test-repo", "test-file.csv")
        expected_url = "https://raw.githubusercontent.com/test-owner/test-repo/master/test-file.csv"
        mock_get.assert_called_once_with(expected_url, verify=True)
