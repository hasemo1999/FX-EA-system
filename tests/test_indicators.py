# -*- coding: utf-8 -*-
"""rough_backtest.py のユーティリティ関数テスト"""
import numpy as np
import pandas as pd
import pytest
import json
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rough_backtest import _ema, _make_indicators, load_config


class TestEma:
    """EMA計算のテスト"""

    def test_ema_returns_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(s, 3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_ema_first_value_equals_first_input(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = _ema(s, 3)
        assert result.iloc[0] == 10.0

    def test_ema_monotonically_increasing_input(self):
        s = pd.Series(range(1, 11), dtype=float)
        result = _ema(s, 3)
        # EMAは入力が単調増加なら単調増加になる
        assert all(result.diff().dropna() > 0)

    def test_ema_constant_input_returns_constant(self):
        s = pd.Series([5.0] * 10)
        result = _ema(s, 3)
        np.testing.assert_array_almost_equal(result.values, [5.0] * 10)


class TestMakeIndicators:
    """インジケーター生成のテスト"""

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用OHLCVデータ"""
        n = 50
        np.random.seed(42)
        close = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
        return pd.DataFrame({
            "open": close + np.random.randn(n) * 0.05,
            "high": close + np.abs(np.random.randn(n) * 0.1),
            "low": close - np.abs(np.random.randn(n) * 0.1),
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })

    def test_adds_ema_columns(self, sample_ohlcv):
        df = _make_indicators(sample_ohlcv.copy(), fast=9, slow=21)
        assert "ema_fast" in df.columns
        assert "ema_slow" in df.columns

    def test_adds_volume_ratio(self, sample_ohlcv):
        df = _make_indicators(sample_ohlcv.copy())
        assert "vol_ratio" in df.columns
        assert "vol_ma20" in df.columns
        assert not df["vol_ratio"].isna().all()

    def test_adds_atr(self, sample_ohlcv):
        df = _make_indicators(sample_ohlcv.copy())
        assert "atr" in df.columns
        assert (df["atr"] >= 0).all()

    def test_no_volume_column(self):
        """volume列がない場合もエラーにならない"""
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
        })
        result = _make_indicators(df.copy())
        assert result["vol_ratio"].isna().all()


class TestLoadConfig:
    """設定ファイル読み込みのテスト"""

    def test_load_valid_config(self, tmp_path):
        config = {"ema_fast": 9, "ema_slow": 21, "tp_pts": 2.0}
        path = tmp_path / "test_config.json"
        path.write_text(json.dumps(config))
        result = load_config(str(path))
        assert result["ema_fast"] == 9
        assert result["tp_pts"] == 2.0

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")
