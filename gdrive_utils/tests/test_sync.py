# -*- coding: utf-8 -*-
import json
import os
import pytest
from datetime import datetime, timezone
from gdrive_utils.sync import SyncEngine, SyncResult, MANIFEST_FILENAME


def test_sync_result_str():
    r = SyncResult(uploaded=["a.csv"], downloaded=["b.csv"], conflicts=[], errors=[])
    s = str(r)
    assert "uploaded=1" in s
    assert "downloaded=1" in s


def test_load_manifest_empty(tmp_path):
    from unittest.mock import MagicMock
    from gdrive_utils.config import GDriveConfig

    cfg = GDriveConfig(
        sa_key_path="",
        folder_id="test",
        sync_root=str(tmp_path),
    )
    dc = MagicMock()
    engine = SyncEngine(dc, cfg)
    manifest = engine._load_manifest()
    assert manifest == {}


def test_save_and_load_manifest(tmp_path):
    from unittest.mock import MagicMock
    from gdrive_utils.config import GDriveConfig

    cfg = GDriveConfig(
        sa_key_path="",
        folder_id="test",
        sync_root=str(tmp_path),
    )
    dc = MagicMock()
    engine = SyncEngine(dc, cfg)

    test_manifest = {
        "file.csv": {
            "local_mtime": "2026-01-01T00:00:00+00:00",
            "drive_mtime": "2026-01-01T00:00:00Z",
            "drive_id": "abc123",
            "md5": "d41d8cd98f00b204e9800998ecf8427e",
        }
    }
    engine._save_manifest(test_manifest)

    loaded = engine._load_manifest()
    assert loaded == test_manifest


def test_scan_local(tmp_path):
    from unittest.mock import MagicMock
    from gdrive_utils.config import GDriveConfig

    (tmp_path / "data.csv").write_text("a,b\n1,2")
    (tmp_path / ".hidden").write_text("hidden")

    cfg = GDriveConfig(sa_key_path="", folder_id="test", sync_root=str(tmp_path))
    dc = MagicMock()
    engine = SyncEngine(dc, cfg)

    files = engine._scan_local(str(tmp_path))
    assert "data.csv" in files
    assert ".hidden" not in files  # ドットファイルは除外


def test_resolve_conflict_newer_wins():
    from unittest.mock import MagicMock
    from gdrive_utils.config import GDriveConfig

    cfg = GDriveConfig(sa_key_path="", folder_id="test")
    dc = MagicMock()
    engine = SyncEngine(dc, cfg)

    # ローカルが新しい場合
    result = engine._resolve_conflict(
        "test.csv",
        "2026-03-10T10:00:00+00:00",
        "2026-03-09T10:00:00Z",
        "newer_wins",
    )
    assert result == "upload"

    # Driveが新しい場合
    result = engine._resolve_conflict(
        "test.csv",
        "2026-03-08T10:00:00+00:00",
        "2026-03-10T10:00:00Z",
        "newer_wins",
    )
    assert result == "download"


def test_resolve_conflict_strategies():
    from unittest.mock import MagicMock
    from gdrive_utils.config import GDriveConfig

    cfg = GDriveConfig(sa_key_path="", folder_id="test")
    dc = MagicMock()
    engine = SyncEngine(dc, cfg)

    assert engine._resolve_conflict("f", "t1", "t2", "local_wins") == "upload"
    assert engine._resolve_conflict("f", "t1", "t2", "drive_wins") == "download"
