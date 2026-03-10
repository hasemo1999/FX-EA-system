# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch
import pytest
from gdrive_utils.drive import DriveClient, MIME_MAP


@pytest.fixture
def mock_drive_client(gdrive_config):
    """DriveClient を認証をモックして生成。"""
    with patch("gdrive_utils.drive.get_credentials") as mock_creds, \
         patch("gdrive_utils.drive.build_drive_service") as mock_build:
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        client = DriveClient(gdrive_config)
        client._mock_service = mock_service
        yield client


def test_guess_mime():
    client_cls = DriveClient
    # テストは静的メソッド的に呼べないのでMIME_MAPを直接テスト
    assert MIME_MAP[".csv"] == "text/csv"
    assert MIME_MAP[".json"] == "application/json"
    assert MIME_MAP[".png"] == "image/png"
    assert MIME_MAP[".md"] == "text/markdown"


def test_list_files(mock_drive_client):
    mock_service = mock_drive_client._mock_service
    mock_service.files().list().execute.return_value = {
        "files": [
            {"id": "f1", "name": "test.csv", "mimeType": "text/csv", "modifiedTime": "2026-01-01T00:00:00Z"},
        ],
        "nextPageToken": None,
    }
    files = mock_drive_client.list_files(folder_id="folder123")
    assert len(files) == 1
    assert files[0]["name"] == "test.csv"


def test_upload_file_new(mock_drive_client, tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("a,b\n1,2")

    mock_service = mock_drive_client._mock_service
    # _find_file_by_name returns None (new file)
    mock_service.files().list().execute.return_value = {"files": []}
    mock_service.files().create().execute.return_value = {"id": "new_id", "name": "test.csv"}

    fid = mock_drive_client.upload_file(str(test_file), folder_id="folder123")
    assert fid == "new_id"


def test_delete_file(mock_drive_client):
    mock_service = mock_drive_client._mock_service
    mock_service.files().delete().execute.return_value = None
    mock_drive_client.delete_file("file_id_123")
    # No exception = success
