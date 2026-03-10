# -*- coding: utf-8 -*-
"""SheetsClient — Google Sheets 読み書き（gspread + pandas）"""

import logging
from typing import Optional

import pandas as pd

from gdrive_utils.auth import get_gspread_client
from gdrive_utils.config import GDriveConfig

logger = logging.getLogger(__name__)


class SheetsClient:
    """Google Sheets 操作クライアント。"""

    def __init__(self, config: GDriveConfig):
        self._config = config
        self._gc = get_gspread_client(config.sa_key_path)

    def read_sheet(
        self, spreadsheet_id: str, worksheet_name: str = "Sheet1"
    ) -> pd.DataFrame:
        """ワークシート全体をDataFrameとして読み取り。"""
        sh = self._gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name)
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        logger.info("Read %d rows from %s/%s", len(df), spreadsheet_id, worksheet_name)
        return df

    def write_sheet(
        self,
        df: pd.DataFrame,
        spreadsheet_id: str,
        worksheet_name: str = "Sheet1",
        mode: str = "replace",
    ) -> None:
        """DataFrameをワークシートに書き込み。mode='replace'でクリア後書き込み、'append'で追記。"""
        sh = self._gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name)

        if mode == "replace":
            ws.clear()
            data = [df.columns.tolist()] + df.values.tolist()
            ws.update(range_name="A1", values=data)
            logger.info("Wrote %d rows (replace) to %s/%s", len(df), spreadsheet_id, worksheet_name)
        elif mode == "append":
            rows = df.values.tolist()
            ws.append_rows(rows, value_input_option="USER_ENTERED")
            logger.info("Appended %d rows to %s/%s", len(df), spreadsheet_id, worksheet_name)

    def append_rows(
        self,
        rows: list,
        spreadsheet_id: str,
        worksheet_name: str = "Sheet1",
    ) -> None:
        """行リストをワークシート末尾に追記。"""
        sh = self._gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name)
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        logger.info("Appended %d rows to %s/%s", len(rows), spreadsheet_id, worksheet_name)

    def read_range(self, spreadsheet_id: str, range_str: str) -> list:
        """A1表記で指定した範囲を読み取り。"""
        sh = self._gc.open_by_key(spreadsheet_id)
        return sh.values_get(range_str).get("values", [])

    def write_range(
        self, values: list, spreadsheet_id: str, range_str: str
    ) -> None:
        """A1表記で指定した範囲に書き込み。"""
        sh = self._gc.open_by_key(spreadsheet_id)
        sh.values_update(
            range_str,
            params={"valueInputOption": "USER_ENTERED"},
            body={"values": values},
        )
        logger.info("Wrote range %s to %s", range_str, spreadsheet_id)

    def create_spreadsheet(self, title: str) -> str:
        """新規スプレッドシートを作成し、IDを返す。"""
        sh = self._gc.create(title)
        logger.info("Created spreadsheet: %s (id=%s)", title, sh.id)
        return sh.id
