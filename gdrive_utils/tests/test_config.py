# -*- coding: utf-8 -*-
import json
import os
import pytest
from gdrive_utils.config import GDriveConfig


def test_from_env(monkeypatch):
    monkeypatch.setenv("GDRIVE_SA_KEY_PATH", "/path/to/key.json")
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "folder123")
    monkeypatch.setenv("GDRIVE_SHEET_ID", "sheet456")
    monkeypatch.setenv("GDRIVE_SYNC_ROOT", "/sync/dir")
    monkeypatch.setenv("GDRIVE_SYNC_DIR", "upload")
    monkeypatch.setenv("GDRIVE_MAX_RETRIES", "5")

    cfg = GDriveConfig.from_env()
    assert cfg.sa_key_path == "/path/to/key.json"
    assert cfg.folder_id == "folder123"
    assert cfg.sheet_id == "sheet456"
    assert cfg.sync_root == "/sync/dir"
    assert cfg.sync_direction == "upload"
    assert cfg.max_retries == 5


def test_from_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "sa_key_path": "/json/key.json",
        "folder_id": "json_folder",
        "sheet_id": "json_sheet",
        "sync_root": "/json/sync",
        "sync_direction": "download",
        "max_retries": 2,
    }))

    cfg = GDriveConfig.from_json(str(config_file))
    assert cfg.sa_key_path == "/json/key.json"
    assert cfg.folder_id == "json_folder"
    assert cfg.max_retries == 2


def test_load_env_overrides_json(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "sa_key_path": "/json/key.json",
        "folder_id": "json_folder",
    }))
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "env_folder")

    cfg = GDriveConfig.load(str(config_file))
    assert cfg.sa_key_path == "/json/key.json"  # JSON値
    assert cfg.folder_id == "env_folder"  # 環境変数で上書き


def test_load_no_file(monkeypatch):
    monkeypatch.delenv("GDRIVE_SA_KEY_PATH", raising=False)
    monkeypatch.delenv("GDRIVE_FOLDER_ID", raising=False)
    cfg = GDriveConfig.load("/nonexistent/path.json")
    assert cfg.folder_id == ""
