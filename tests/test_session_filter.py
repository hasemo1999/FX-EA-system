# -*- coding: utf-8 -*-
"""session_filter.py のテスト"""
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_filter import build_session_mask, _to_min, ensure_tz


class TestToMin:
    """時間文字列→分変換のテスト"""

    def test_midnight(self):
        assert _to_min("00:00") == 0

    def test_noon(self):
        assert _to_min("12:00") == 720

    def test_end_of_day(self):
        assert _to_min("23:59") == 1439

    def test_morning(self):
        assert _to_min("09:30") == 570


class TestEnsureTz:
    """タイムゾーン変換のテスト"""

    def test_naive_to_tokyo(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        result = ensure_tz(idx, "Asia/Tokyo")
        assert str(result.tz) == "Asia/Tokyo"

    def test_utc_to_tokyo(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        result = ensure_tz(idx, "Asia/Tokyo")
        assert str(result.tz) == "Asia/Tokyo"


class TestBuildSessionMask:
    """セッションマスクのテスト"""

    @pytest.fixture
    def trading_day_index(self):
        """1日分の5分足インデックス（UTC）"""
        return pd.date_range(
            "2024-01-08 00:00", "2024-01-08 23:55",
            freq="5min", tz="UTC"
        )

    def test_full_day_session(self, trading_day_index):
        """00:00-23:59で全バーが通る"""
        mask = build_session_mask(
            trading_day_index,
            sessions=[("00:00", "23:59")],
            tz="UTC"
        )
        # 00:00〜23:55の全288バーが23:59未満なので全部True
        assert mask.sum() == len(trading_day_index)

    def test_restricted_session(self, trading_day_index):
        """09:00-15:00で制限"""
        mask = build_session_mask(
            trading_day_index,
            sessions=[("09:00", "15:00")],
            tz="UTC"
        )
        # 09:00〜14:55の72バー
        assert mask.sum() == 72

    def test_overnight_session(self, trading_day_index):
        """日跨ぎセッション（22:00-06:00）"""
        mask = build_session_mask(
            trading_day_index,
            sessions=[("22:00", "06:00")],
            tz="UTC"
        )
        assert mask.sum() > 0
        # 22:00以降 OR 06:00未満

    def test_empty_sessions_fail_safe(self, trading_day_index):
        """セッションなしの場合"""
        mask = build_session_mask(
            trading_day_index,
            sessions=[],
            tz="UTC"
        )
        assert mask.sum() == 0

    def test_weekday_filter(self):
        """曜日フィルタ"""
        # 月-金（2024-01-08は月曜日）
        idx = pd.date_range("2024-01-08", periods=288, freq="5min", tz="UTC")
        mask = build_session_mask(
            idx,
            sessions=[("00:00", "23:59")],
            tz="UTC",
            weekdays=[0, 1, 2, 3, 4]  # Mon-Fri
        )
        assert mask.sum() > 0
