# -*- coding: utf-8 -*-
"""サービスアカウント認証"""

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_credentials(sa_key_path: str) -> service_account.Credentials:
    """サービスアカウントJSONからCredentialsを生成。"""
    path = os.path.expanduser(sa_key_path)
    return service_account.Credentials.from_service_account_file(path, scopes=SCOPES)


def build_drive_service(creds: service_account.Credentials):
    """Drive API v3 サービスオブジェクトを返す。"""
    return build("drive", "v3", credentials=creds)


def build_sheets_service(creds: service_account.Credentials):
    """Sheets API v4 サービスオブジェクトを返す。"""
    return build("sheets", "v4", credentials=creds)


def get_gspread_client(sa_key_path: str):
    """gspread クライアントを返す。"""
    import gspread
    path = os.path.expanduser(sa_key_path)
    return gspread.service_account(filename=path)
