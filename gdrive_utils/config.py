# -*- coding: utf-8 -*-
"""設定管理 — 環境変数を優先し、JSONファイルにフォールバック"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GDriveConfig:
    sa_key_path: str = ""
    folder_id: str = ""
    sheet_id: Optional[str] = None
    sync_root: str = "."
    sync_direction: str = "both"  # "upload" | "download" | "both"
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "GDriveConfig":
        return cls(
            sa_key_path=os.environ.get("GDRIVE_SA_KEY_PATH", "~/.config/gdrive/sa_key.json"),
            folder_id=os.environ.get("GDRIVE_FOLDER_ID", ""),
            sheet_id=os.environ.get("GDRIVE_SHEET_ID"),
            sync_root=os.environ.get("GDRIVE_SYNC_ROOT", "."),
            sync_direction=os.environ.get("GDRIVE_SYNC_DIR", "both"),
            max_retries=int(os.environ.get("GDRIVE_MAX_RETRIES", "3")),
        )

    @classmethod
    def from_json(cls, path: str) -> "GDriveConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            sa_key_path=data.get("sa_key_path", "~/.config/gdrive/sa_key.json"),
            folder_id=data.get("folder_id", ""),
            sheet_id=data.get("sheet_id"),
            sync_root=data.get("sync_root", "."),
            sync_direction=data.get("sync_direction", "both"),
            max_retries=data.get("max_retries", 3),
        )

    @classmethod
    def load(cls, json_path: Optional[str] = None) -> "GDriveConfig":
        """環境変数を優先し、JSONにフォールバック。環境変数が設定されていれば上書き。"""
        if json_path and os.path.exists(json_path):
            cfg = cls.from_json(json_path)
        else:
            cfg = cls()

        # 環境変数で上書き
        if os.environ.get("GDRIVE_SA_KEY_PATH"):
            cfg.sa_key_path = os.environ["GDRIVE_SA_KEY_PATH"]
        if os.environ.get("GDRIVE_FOLDER_ID"):
            cfg.folder_id = os.environ["GDRIVE_FOLDER_ID"]
        if os.environ.get("GDRIVE_SHEET_ID"):
            cfg.sheet_id = os.environ["GDRIVE_SHEET_ID"]
        if os.environ.get("GDRIVE_SYNC_ROOT"):
            cfg.sync_root = os.environ["GDRIVE_SYNC_ROOT"]
        if os.environ.get("GDRIVE_SYNC_DIR"):
            cfg.sync_direction = os.environ["GDRIVE_SYNC_DIR"]
        if os.environ.get("GDRIVE_MAX_RETRIES"):
            cfg.max_retries = int(os.environ["GDRIVE_MAX_RETRIES"])

        return cfg
