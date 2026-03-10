# -*- coding: utf-8 -*-
"""SyncEngine — ローカル ⇔ Google Drive 双方向同期"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from gdrive_utils.config import GDriveConfig
from gdrive_utils.drive import DriveClient

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".gdrive_sync_manifest.json"


@dataclass
class SyncResult:
    uploaded: list = field(default_factory=list)
    downloaded: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def __str__(self):
        return (
            f"uploaded={len(self.uploaded)}, downloaded={len(self.downloaded)}, "
            f"conflicts={len(self.conflicts)}, errors={len(self.errors)}"
        )


class SyncEngine:
    """manifest ベースの双方向同期エンジン。"""

    def __init__(self, drive_client: DriveClient, config: GDriveConfig):
        self._drive = drive_client
        self._config = config
        self._manifest_path = os.path.join(config.sync_root, MANIFEST_FILENAME)

    def sync(
        self,
        direction: Optional[str] = None,
        conflict_strategy: str = "newer_wins",
    ) -> SyncResult:
        """同期を実行。direction: "upload" | "download" | "both" (None=config値)"""
        direction = direction or self._config.sync_direction
        folder_id = self._config.folder_id
        local_dir = self._config.sync_root
        os.makedirs(local_dir, exist_ok=True)

        manifest = self._load_manifest()
        result = SyncResult()

        # ローカルファイル一覧
        local_files = self._scan_local(local_dir)
        # Driveファイル一覧
        drive_files = {f["name"]: f for f in self._drive.list_files(folder_id)}

        all_names = set(local_files.keys()) | set(drive_files.keys()) | set(manifest.keys())

        for name in all_names:
            if name == MANIFEST_FILENAME:
                continue
            try:
                self._sync_file(
                    name, local_dir, folder_id,
                    local_files.get(name),
                    drive_files.get(name),
                    manifest.get(name),
                    manifest, direction, conflict_strategy, result,
                )
            except Exception as e:
                logger.error("Sync error for %s: %s", name, e)
                result.errors.append({"name": name, "error": str(e)})

        self._save_manifest(manifest)
        return result

    def _sync_file(
        self, name, local_dir, folder_id,
        local_info, drive_info, manifest_entry,
        manifest, direction, conflict_strategy, result,
    ):
        local_path = os.path.join(local_dir, name)
        local_exists = local_info is not None
        drive_exists = drive_info is not None
        prev = manifest_entry or {}

        local_mtime = local_info["mtime"] if local_info else None
        drive_mtime = drive_info.get("modifiedTime", "") if drive_info else None

        local_changed = local_exists and local_mtime != prev.get("local_mtime")
        drive_changed = drive_exists and drive_mtime != prev.get("drive_mtime")

        # 新規ファイル（片方のみ）
        if local_exists and not drive_exists:
            if direction in ("upload", "both"):
                fid = self._drive.upload_file(local_path, folder_id)
                meta = self._drive.get_file_metadata(fid)
                manifest[name] = {
                    "local_mtime": local_mtime,
                    "drive_mtime": meta.get("modifiedTime", ""),
                    "drive_id": fid,
                    "md5": local_info.get("md5", ""),
                }
                result.uploaded.append(name)
            return

        if drive_exists and not local_exists:
            if direction in ("download", "both"):
                self._drive.download_file(drive_info["id"], local_path)
                new_local = self._file_info(local_path)
                manifest[name] = {
                    "local_mtime": new_local["mtime"],
                    "drive_mtime": drive_mtime,
                    "drive_id": drive_info["id"],
                    "md5": new_local.get("md5", ""),
                }
                result.downloaded.append(name)
            return

        if not local_exists and not drive_exists:
            manifest.pop(name, None)
            return

        # 両方存在
        if local_changed and drive_changed:
            # コンフリクト
            action = self._resolve_conflict(
                name, local_mtime, drive_mtime, conflict_strategy
            )
            result.conflicts.append({"name": name, "resolved": action})
            if action == "upload" and direction != "download":
                fid = self._drive.upload_file(local_path, folder_id)
                meta = self._drive.get_file_metadata(fid)
                manifest[name] = {
                    "local_mtime": local_mtime,
                    "drive_mtime": meta.get("modifiedTime", ""),
                    "drive_id": fid,
                    "md5": local_info.get("md5", ""),
                }
            elif action == "download" and direction != "upload":
                self._drive.download_file(drive_info["id"], local_path)
                new_local = self._file_info(local_path)
                manifest[name] = {
                    "local_mtime": new_local["mtime"],
                    "drive_mtime": drive_mtime,
                    "drive_id": drive_info["id"],
                    "md5": new_local.get("md5", ""),
                }
        elif local_changed and direction in ("upload", "both"):
            fid = drive_info["id"]
            self._drive.upload_file(local_path, folder_id)
            meta = self._drive.get_file_metadata(fid)
            manifest[name] = {
                "local_mtime": local_mtime,
                "drive_mtime": meta.get("modifiedTime", ""),
                "drive_id": fid,
                "md5": local_info.get("md5", ""),
            }
            result.uploaded.append(name)
        elif drive_changed and direction in ("download", "both"):
            self._drive.download_file(drive_info["id"], local_path)
            new_local = self._file_info(local_path)
            manifest[name] = {
                "local_mtime": new_local["mtime"],
                "drive_mtime": drive_mtime,
                "drive_id": drive_info["id"],
                "md5": new_local.get("md5", ""),
            }
            result.downloaded.append(name)
        else:
            # 変更なし — manifest更新のみ
            if name not in manifest and drive_info:
                manifest[name] = {
                    "local_mtime": local_mtime,
                    "drive_mtime": drive_mtime,
                    "drive_id": drive_info["id"],
                    "md5": local_info.get("md5", "") if local_info else "",
                }

    def _resolve_conflict(self, name, local_mtime, drive_mtime, strategy):
        if strategy == "local_wins":
            return "upload"
        elif strategy == "drive_wins":
            return "download"
        else:  # newer_wins
            try:
                local_dt = datetime.fromisoformat(local_mtime) if local_mtime else datetime.min
                drive_dt = datetime.fromisoformat(
                    drive_mtime.replace("Z", "+00:00")
                ) if drive_mtime else datetime.min.replace(tzinfo=timezone.utc)
                if local_dt.tzinfo is None:
                    local_dt = local_dt.replace(tzinfo=timezone.utc)
                return "upload" if local_dt >= drive_dt else "download"
            except Exception:
                return "upload"  # デフォルトはローカル優先

    def _scan_local(self, local_dir: str) -> dict:
        """ローカルファイルの情報を収集。"""
        files = {}
        for entry in os.scandir(local_dir):
            if entry.is_file() and not entry.name.startswith("."):
                files[entry.name] = self._file_info(entry.path)
        return files

    def _file_info(self, filepath: str) -> dict:
        stat = os.stat(filepath)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        md5 = self._md5(filepath)
        return {"mtime": mtime, "md5": md5, "size": stat.st_size}

    @staticmethod
    def _md5(filepath: str) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_manifest(self) -> dict:
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
