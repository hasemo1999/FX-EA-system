# -*- coding: utf-8 -*-
"""DriveClient — Google Drive ファイル操作"""

import io
import logging
import mimetypes
import os
from typing import Optional

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from gdrive_utils.auth import get_credentials, build_drive_service
from gdrive_utils.config import GDriveConfig
from gdrive_utils.retry import retry_on_api_error

logger = logging.getLogger(__name__)

MIME_MAP = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".png": "image/png",
    ".log": "text/plain",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


class DriveClient:
    """Google Drive API v3 操作クライアント。"""

    def __init__(self, config: GDriveConfig):
        self._config = config
        self._creds = get_credentials(config.sa_key_path)
        self._service = build_drive_service(self._creds)

    def _guess_mime(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        return MIME_MAP.get(ext, mimetypes.guess_type(filepath)[0] or "application/octet-stream")

    @retry_on_api_error(max_retries=3)
    def upload_file(
        self,
        local_path: str,
        folder_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> str:
        """ファイルをアップロード。同名ファイルが存在すれば更新。Drive file IDを返す。"""
        folder_id = folder_id or self._config.folder_id
        filename = os.path.basename(local_path)
        mime = mime_type or self._guess_mime(local_path)

        # 既存ファイルを検索
        existing = self._find_file_by_name(filename, folder_id)
        media = MediaFileUpload(local_path, mimetype=mime, resumable=True)

        if existing:
            # 更新
            result = self._service.files().update(
                fileId=existing["id"],
                media_body=media,
            ).execute()
            logger.info("Updated: %s (id=%s)", filename, result["id"])
        else:
            # 新規作成
            metadata = {"name": filename}
            if folder_id:
                metadata["parents"] = [folder_id]
            result = self._service.files().create(
                body=metadata,
                media_body=media,
                fields="id, name",
            ).execute()
            logger.info("Uploaded: %s (id=%s)", filename, result["id"])

        return result["id"]

    @retry_on_api_error(max_retries=3)
    def download_file(self, file_id: str, local_path: str) -> str:
        """file_idのファイルをローカルにダウンロード。"""
        request = self._service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        logger.info("Downloaded: %s -> %s", file_id, local_path)
        return local_path

    def download_by_name(
        self, name: str, local_path: str, folder_id: Optional[str] = None
    ) -> str:
        """ファイル名で検索してダウンロード。"""
        folder_id = folder_id or self._config.folder_id
        found = self._find_file_by_name(name, folder_id)
        if not found:
            raise FileNotFoundError(f"Drive file not found: {name} in folder {folder_id}")
        return self.download_file(found["id"], local_path)

    @retry_on_api_error(max_retries=3)
    def list_files(
        self, folder_id: Optional[str] = None, name_pattern: Optional[str] = None
    ) -> list:
        """フォルダ内のファイル一覧を返す。"""
        folder_id = folder_id or self._config.folder_id
        query_parts = []
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        query_parts.append("trashed = false")
        if name_pattern:
            query_parts.append(f"name contains '{name_pattern}'")
        query = " and ".join(query_parts)

        results = []
        page_token = None
        while True:
            resp = self._service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum, size)",
                pageToken=page_token,
                pageSize=100,
            ).execute()
            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return results

    @retry_on_api_error(max_retries=3)
    def delete_file(self, file_id: str) -> None:
        """ファイルを削除。"""
        self._service.files().delete(fileId=file_id).execute()
        logger.info("Deleted: %s", file_id)

    @retry_on_api_error(max_retries=3)
    def get_file_metadata(self, file_id: str) -> dict:
        """ファイルのメタデータを取得。"""
        return self._service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, modifiedTime, md5Checksum, size",
        ).execute()

    @retry_on_api_error(max_retries=3)
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """フォルダを作成し、IDを返す。"""
        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]
        result = self._service.files().create(
            body=metadata, fields="id"
        ).execute()
        logger.info("Created folder: %s (id=%s)", name, result["id"])
        return result["id"]

    def _find_file_by_name(self, name: str, folder_id: Optional[str] = None) -> Optional[dict]:
        """フォルダ内で名前一致するファイルを1件返す。"""
        query_parts = [f"name = '{name}'", "trashed = false"]
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        query = " and ".join(query_parts)
        resp = self._service.files().list(
            q=query, fields="files(id, name, modifiedTime)", pageSize=1
        ).execute()
        files = resp.get("files", [])
        return files[0] if files else None
