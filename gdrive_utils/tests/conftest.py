# -*- coding: utf-8 -*-
import json
import pytest


@pytest.fixture
def mock_sa_key(tmp_path):
    """テスト用の偽サービスアカウントJSONを作成。"""
    key = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key-id",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJBALRiMLAH\n-----END RSA PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    path = tmp_path / "sa_key.json"
    path.write_text(json.dumps(key))
    return str(path)


@pytest.fixture
def gdrive_config(mock_sa_key):
    """テスト用のGDriveConfig。"""
    from gdrive_utils.config import GDriveConfig
    return GDriveConfig(
        sa_key_path=mock_sa_key,
        folder_id="test-folder-id",
        sheet_id="test-sheet-id",
        sync_root="/tmp/test_sync",
        sync_direction="both",
        max_retries=1,
    )
