# -*- coding: utf-8 -*-
"""
rough_backtest.py
実データ(CSV)前提の簡易バックテストハーネス（ダミー生成なし）。
- 入力: ./backtest_input/*.csv のOHLCV（TradingView/OANDA/Dukascopy等のエクスポート）
- 出力: ./backtest_output/ に trades.csv / equity.csv / metrics.json / walk_forward.csv / summary.md / 図PNG
- ルール（初期値・編集可）:
  * 低位足: EMA9/EMA21のGC/DC + 出来高比(Vol / VolMA20) >= 1.5
  * 上位足(60分): **3モード**（`higher_tf_mode`: "resample" | "csv" | "off"）
    - resample: 低位足から内部リサンプル（推奨）
    - csv: `./backtest_input_1h/*.csv` の1Hを使用
    - off: MTF無効
  * セッション: デフォは日中（指数向け）。FXは24hに変更推奨
  * コストモデル: `spread_pts` / `slippage_pts`（片側）/ `commission_value`（1回の取引あたり固定額）
  * 方向: `direction`: "long" | "short" | "both"
  * 期間フィルタ: `start_date`, `end_date`（ISO文字列）
- 指標: PF, MaxDD, WinRate, NetProfit, AvgR, Walk-Forward合格率（7分割, PF_OOS>=1.0基準）

使い方:
  python rough_backtest.py --cfg backtest_config.json
  # 例) 1H CSVを使う
  python rough_backtest.py --cfg backtest_config.json --higher_tf_mode csv --higher_tf_csv_dir ./backtest_input_1h
  # 例) MTF OFF
  python rough_backtest.py --cfg backtest_config.json --use_mtf false
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import logging
from datetime import datetime, timedelta
import pytz
from session_filter import build_session_mask

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _safe_timestamp_to_iso(ts):
    """タイムスタンプを安全にISO形式に変換"""
    try:
        if hasattr(ts, 'isoformat'):
            return ts.isoformat()
        elif isinstance(ts, (int, np.integer)):
            # numpy.int64の場合、適切な単位で変換
            if ts > 1e18:  # ナノ秒単位
                return pd.Timestamp(ts, unit='ns').isoformat()
            elif ts > 1e15:  # マイクロ秒単位
                return pd.Timestamp(ts, unit='us').isoformat()
            elif ts > 1e12:  # ミリ秒単位
                return pd.Timestamp(ts, unit='ms').isoformat()
            else:  # 秒単位
                return pd.Timestamp(ts, unit='s').isoformat()
        else:
            return pd.Timestamp(ts).isoformat()
    except Exception as e:
        logger.warning(f"タイムスタンプ変換エラー: {e}, 値: {ts}")
        return str(ts)

# -------------------- 共通ユーティリティ --------------------

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_input_files(in_dir):
    files = []
    if not os.path.isdir(in_dir):
        return files
    for fn in os.listdir(in_dir):
        if fn.lower().endswith(".csv"):
            files.append(os.path.join(in_dir, fn))
    return sorted(files)


def _parse_csv(path, tz="Asia/Tokyo", ts_col_candidates=("time","timestamp","date","datetime","Time","UTC","Time (UTC)")):
    df = pd.read_csv(path)
    # 推定: カラム名の正規化
    cols = {c.lower(): c for c in df.columns}
    # 時刻
    ts_col = None
    for c in ts_col_candidates:
        if c.lower() in cols:
            ts_col = cols[c.lower()]
            break
    if ts_col is None:
        raise ValueError("Timestamp column not found. Expected one of: %s" % (ts_col_candidates,))
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Timestamp parse failed for some rows.")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    # OHLCV
    mapper = {
        "open": ["open","o","Open"],
        "high": ["high","h","High"],
        "low": ["low","l","Low"],
        "close": ["close","c","Close"],
        "volume": ["volume","v","Volume","Volume USD","Volume USDT"],
    }
    out = {}
    for k, cands in mapper.items():
        found = None
        for c in cands:
            if c in df.columns:
                found = c; break
        if found is None and k != "volume":  # volumeは無い市場もあるのでNaN可
            raise ValueError(f"Column for {k} not found. Tried {cands}")
        out[k] = df[found] if found else np.nan
    outdf = pd.DataFrame(out)
    outdf["timestamp"] = df["timestamp"]
    outdf = outdf.dropna(subset=["open","high","low","close"])
    outdf = outdf.sort_values("timestamp").reset_index(drop=True)
    return outdf


def _resample_ohlcv(df, rule="5min"):
    """OHLCVデータのリサンプリング（非推奨警告対応）"""
    df = df.set_index("timestamp")
    # 'T'を'min'に変換
    if isinstance(rule, str) and rule.endswith('T'):
        rule = rule.replace('T', 'min')
    
    ohlc = df[["open","high","low","close"]].resample(rule).agg({
        "open":"first","high":"max","low":"min","close":"last"})
    if "volume" in df.columns:
        vol = df["volume"].resample(rule).sum()
        ohlc["volume"] = vol
    else:
        ohlc["volume"] = np.nan
    ohlc = ohlc.dropna(subset=["open","high","low","close"])
    return ohlc


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _make_indicators(df, fast=9, slow=21):
    df["ema_fast"] = _ema(df["close"], fast)
    df["ema_slow"] = _ema(df["close"], slow)
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    else:
        df["vol_ratio"] = np.nan
    
    # ATR計算（利確ロジック用）
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(14, min_periods=1).mean()
    
    return df

def _atr_quantile_filter(df, atr_period=14, atr_q_window=500, atr_q_lo=0.15, atr_q_hi=0.85):
    """
    ATR分位フィルタ
    指定した過去本数のATR分布から下限・上限の分位点を求め、
    現在のATRがその帯域内のときだけエントリーを許可
    
    Args:
        df: DataFrame with OHLCV data
        atr_period: ATR計算期間
        atr_q_window: 分位を計算する過去本数
        atr_q_lo: 下側分位（0.0〜1.0）
        atr_q_hi: 上側分位（0.0〜1.0）
    
    Returns:
        pd.Series: True if ATR is within quantile range
    """
    try:
        # ATR計算
        if "atr" not in df.columns:
            df = _make_indicators(df, atr_period=atr_period)
        
        # 分位フィルタの結果を格納
        atr_quantile_ok = pd.Series([True] * len(df), index=df.index)
        
        # デバッグ用カウンタ
        total_checks = 0
        rejected_count = 0
        
        # 各バーで分位計算
        for i in range(atr_q_window, len(df)):
            # 過去atr_q_window本のATRを取得
            atr_window = df["atr"].iloc[i-atr_q_window:i]
            
            # 分位点の計算
            atr_sorted = atr_window.sort_values()
            idx_lo = int(np.floor(atr_q_lo * (len(atr_sorted) - 1)))
            idx_hi = int(np.ceil(atr_q_hi * (len(atr_sorted) - 1)))
            
            idx_lo = max(0, min(len(atr_sorted) - 1, idx_lo))
            idx_hi = max(0, min(len(atr_sorted) - 1, idx_hi))
            
            q_lo = atr_sorted.iloc[idx_lo]
            q_hi = atr_sorted.iloc[idx_hi]
            
            # 現在のATRが帯域内かチェック
            atr_curr = df["atr"].iloc[i]
            is_within_range = (atr_curr >= q_lo and atr_curr <= q_hi)
            atr_quantile_ok.iloc[i] = is_within_range
            
            total_checks += 1
            if not is_within_range:
                rejected_count += 1
        
        # デバッグ情報をログ出力
        logger.info(f"ATR分位フィルタ: 総チェック数={total_checks}, 除外数={rejected_count}, 除外率={rejected_count/total_checks*100:.2f}%")
        logger.info(f"ATR分位設定: lo={atr_q_lo}, hi={atr_q_hi}, window={atr_q_window}")
        
        return atr_quantile_ok
        
    except Exception as e:
        logger.error(f"ATR分位フィルタエラー: {str(e)}")
        return pd.Series([True] * len(df), index=df.index)


def _session_mask(idx, start="09:00", end="15:00", friday_close_time="15:05"):
    """セッション時間マスク（堅牢版）"""
    try:
        # 金曜日の特別処理
        friday_close_h, friday_close_m = map(int, friday_close_time.split(":"))
        
        # 通常セッション
        sessions = [(start, end)]
        
        # 金曜日特別セッション
        friday_sessions = [(start, friday_close_time)]
        
        # 平日（月-木）のマスク
        weekdays_normal = [0, 1, 2, 3]  # 月-木
        normal_mask = build_session_mask(idx, sessions, tz="Asia/Tokyo", weekdays=weekdays_normal, strict=False)
        
        # 金曜日のマスク
        weekdays_friday = [4]  # 金曜日
        friday_mask = build_session_mask(idx, friday_sessions, tz="Asia/Tokyo", weekdays=weekdays_friday, strict=False)
        
        # 結合（非推奨警告対応）
        combined_mask = normal_mask | friday_mask
        return combined_mask
        
    except Exception as e:
        logger.error(f"セッションマスクエラー: {str(e)}")
        # フェイルオープン（24時間取引）
        return pd.Series(True, index=idx)


def _load_news_exclusions(news_csv_path, exclusion_minutes=30, tz="UTC"):
    """ニュース除外時刻の読み込み"""
    if not news_csv_path or not os.path.exists(news_csv_path):
        return []
    
    try:
        news_df = pd.read_csv(news_csv_path)
        # 時刻カラムの推定
        time_cols = [col for col in news_df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]
        if not time_cols:
            print(f"Warning: No time column found in {news_csv_path}")
            return []
        
        time_col = time_cols[0]
        news_times = pd.to_datetime(news_df[time_col])
        
        # タイムゾーンを設定
        if news_times.dt.tz is None:
            news_times = news_times.dt.tz_localize(tz)
        else:
            news_times = news_times.dt.tz_convert(tz)
        
        # 除外期間の計算（前後exclusion_minutes分）
        exclusion_periods = []
        for news_time in news_times:
            start_time = news_time - pd.Timedelta(minutes=exclusion_minutes)
            end_time = news_time + pd.Timedelta(minutes=exclusion_minutes)
            exclusion_periods.append((start_time, end_time))
        
        return exclusion_periods
    except Exception as e:
        print(f"Warning: Failed to load news exclusions from {news_csv_path}: {e}")
        return []


def _is_news_exclusion_time(timestamp, exclusion_periods):
    """指定時刻がニュース除外期間内かチェック"""
    if not exclusion_periods:
        return False
    
    for start_time, end_time in exclusion_periods:
        if start_time <= timestamp <= end_time:
            return True
    return False


def _time_session_mask(idx, sessions):
    """複数の時間帯セッションに対応したマスク（堅牢版）"""
    if not sessions:
        return pd.Series([True] * len(idx), index=idx)
    
    try:
        # セッションをタプルのリストに変換
        session_tuples = [(session["start"], session["end"]) for session in sessions]
        
        # 堅牢なセッションフィルタを使用
        return build_session_mask(idx, session_tuples, tz="UTC", strict=False)
        
    except Exception as e:
        logger.error(f"時間セッションマスクエラー: {str(e)}")
        # フェイルオープン（24時間取引）
        return pd.Series(True, index=idx)

# -------------------- MTF（上位足）構築 --------------------

def _higher_tf_filter_resample(df_low, rule_hi="60T", fast=9, slow=21):
    hi = _resample_ohlcv(df_low.reset_index()[["timestamp","open","high","low","close","volume"]], rule=rule_hi)
    hi = _make_indicators(hi, fast, slow)
    hi_sign = np.where(hi["ema_fast"] > hi["ema_slow"], 1, -1)
    hi_dir = pd.Series(hi_sign, index=hi.index)
    
    # タイムゾーン処理の強化
    try:
        # 両方のインデックスのタイムゾーン情報を確認
        low_tz = getattr(df_low.index, 'tz', None)
        hi_tz = getattr(hi_dir.index, 'tz', None)
        
        # タイムゾーンを統一
        if low_tz != hi_tz:
            if low_tz is not None and hi_tz is None:
                # hi_dirをlow_tzに合わせる
                hi_dir.index = hi_dir.index.tz_localize(low_tz)
            elif low_tz is None and hi_tz is not None:
                # hi_dirのタイムゾーンを削除
                hi_dir.index = hi_dir.index.tz_localize(None)
            elif low_tz is not None and hi_tz is not None:
                # hi_dirをlow_tzに変換
                hi_dir.index = hi_dir.index.tz_convert(low_tz)
        
        # reindex実行
        hi_dir = hi_dir.reindex(df_low.index, method="ffill")
    except Exception as e:
        logger.warning(f"タイムゾーン処理エラー: {e}")
        # エラー時はタイムゾーンを無視してreindex
        try:
            hi_dir = hi_dir.reindex(df_low.index, method="ffill")
        except Exception as e2:
            logger.warning(f"reindexエラー: {e2}")
            # 最終手段：インデックスを文字列化して処理
            try:
                hi_dir.index = hi_dir.index.astype(str)
                df_low_str_index = df_low.index.astype(str)
                hi_dir = hi_dir.reindex(df_low_str_index, method="ffill")
                hi_dir.index = df_low.index  # 元のインデックスに戻す
            except Exception as e3:
                logger.error(f"最終手段も失敗: {e3}")
                # 完全に失敗した場合はデフォルト値で埋める
                hi_dir = pd.Series(1, index=df_low.index, dtype=int)
    
    return hi_dir.fillna(1).astype(int)


def _higher_tf_slope_filter(df_low, rule_hi="60T", fast=9, slow=21):
    """MTFの傾きフィルタ（EMAの差分の変化で上向き判定）"""
    hi = _resample_ohlcv(df_low.reset_index()[["timestamp","open","high","low","close","volume"]], rule=rule_hi)
    hi = _make_indicators(hi, fast, slow)
    
    # EMAの差分を作成
    hi['_ema_diff'] = hi['ema_fast'] - hi['ema_slow']
    # 差分の変化（傾き）を計算
    hi['_ema_diff_slope'] = hi['_ema_diff'] - hi['_ema_diff'].shift(1)
    # 上向き判定（差分が増加中）
    hi_slope_up = (hi['_ema_diff_slope'] > 0).astype(int)  # 上向き=1, それ以外=0
    
    # 低次TFへffillで貼り付け
    hi_slope_up = hi_slope_up.reindex(df_low.index, method='ffill').fillna(1).astype(int)
    return hi_slope_up


def _higher_tf_filter_from_csv(df_low, csv_dir, tz="Asia/Tokyo", rule_hi="60T", fast=9, slow=21):
    files = _find_input_files(csv_dir)
    if not files:
        raise FileNotFoundError(f"No 1H CSV files found in: {csv_dir}")
    dfs = []
    for p in files:
        df0 = _parse_csv(p, tz=tz)
        dfs.append(df0)
    hi = pd.concat(dfs, axis=0).sort_values("timestamp").reset_index(drop=True)
    hi = _resample_ohlcv(hi, rule=rule_hi)  # 念のため60Tに正規化
    hi = _make_indicators(hi, fast, slow)
    hi_sign = np.where(hi["ema_fast"] > hi["ema_slow"], 1, -1)
    hi_dir = pd.Series(hi_sign, index=hi.index)
    hi_dir = hi_dir.reindex(df_low.index, method="ffill")
    return hi_dir.fillna(1).astype(int)


def _build_hi_dir(df_low, cfg):
    if not cfg.get("use_mtf", True):
        return pd.Series(1, index=df_low.index, dtype=int)
    mode = str(cfg.get("higher_tf_mode", "resample")).lower()
    rule = cfg.get("higher_tf_rule", cfg.get("higher_tf", "60T"))
    fast = cfg.get("ema_fast", 9)
    slow = cfg.get("ema_slow", 21)
    tz = cfg.get("tz", "Asia/Tokyo")
    if mode == "off":
        return pd.Series(1, index=df_low.index, dtype=int)
    if mode == "csv":
        csv_dir = cfg.get("higher_tf_csv_dir", "./backtest_input_1h")
        return _higher_tf_filter_from_csv(df_low, csv_dir, tz=tz, rule_hi=rule, fast=fast, slow=slow)
    # default: resample
    return _higher_tf_filter_resample(df_low, rule_hi=rule, fast=fast, slow=slow)

# -------------------- バックテスト本体 --------------------

def backtest(df, ema_fast=9, ema_slow=21, vol_ratio_th=1.5, stop_pts=1.0, tp_pts=2.0, 
             point_value=1.0, spread_pts=0.0, slippage_pts=0.0, commission_value=0.0,
             direction="long", use_mtf=False, higher_tf_mode="csv", higher_tf_rule="60T", 
             higher_tf_csv_dir=None, start_date=None, end_date=None, tz="UTC",
             session_start="09:00", session_end="15:00", friday_close_time="15:05",
             use_atr_tp=False, atr_multiplier=2.0, exclude_news=False, news_exclusion_minutes=30,
             news_exclude_csv=None, use_slope_filter=False, use_breakeven=False, breakeven_r=0.5,
             time_sessions=None, session_policy="flat", cfg=None):
    df_low = df.copy()
    df = _make_indicators(df_low, ema_fast, ema_slow)
    # MTF方向
    if use_mtf:
        hi_dir = _higher_tf_filter_resample(df_low, higher_tf_rule, ema_fast, ema_slow)
        if use_slope_filter:
            hi_slope = _higher_tf_slope_filter(df_low, higher_tf_rule, ema_fast, ema_slow)
        else:
            hi_slope = None
    else:
        hi_dir = pd.Series(1, index=df_low.index, dtype=int)  # デフォルトでロング許可
        hi_slope = None
    # セッション（時間帯除外対応）
    if time_sessions:
        # 時間帯除外セッション
        sess = _time_session_mask(df.index, time_sessions)
        # 金曜日特別処理は時間帯除外では無効化
    else:
        # 従来のセッション
        sess = _session_mask(df.index, start=session_start, end=session_end, friday_close_time=friday_close_time)
    
    # ニュース除外フィルタ
    news_excluded = pd.Series([False] * len(df), index=df.index)
    if exclude_news and news_exclusion_minutes is not None and news_exclude_csv is not None:
        news_excluded = _is_news_exclusion_time(df.index, _load_news_exclusions(news_exclude_csv, news_exclusion_minutes, tz))
    
    # シグナル
    cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    cross_dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    vol_ok = (df["vol_ratio"] >= vol_ratio_th) | df["vol_ratio"].isna()
    
    # MTF方向と傾きフィルタの組み合わせ
    use_slope = cfg.get("mtf_slope_filter", False) if 'cfg' in locals() and cfg is not None else False
    
    if use_mtf:
        long_gate = (hi_dir == 1) & (hi_slope == 1 if use_slope and hi_slope is not None else True)
        short_gate = (hi_dir == -1) & (hi_slope == 0 if use_slope and hi_slope is not None else True)
    else:
        long_gate = short_gate = True
    
    # 品質フィルタ（ATRレジーム + EMAギャップ）
    quality_gate_base = pd.Series([True] * len(df), index=df.index)
    if 'cfg' in locals() and cfg is not None:
        # ATRレジームフィルタ
        atr_floor_mult = cfg.get("atr_floor_mult", None)
        atr_floor_window = cfg.get("atr_floor_window", 2016)
        if atr_floor_mult is not None and "atr" in df.columns:
            atr_med = df["atr"].rolling(atr_floor_window, min_periods=1).median()
            atr_ok = (df["atr"] >= atr_floor_mult * atr_med)
            quality_gate_base = quality_gate_base & atr_ok
        
        # EMAギャップフィルタ
        ema_gap_atr_mult = cfg.get("ema_gap_atr_mult", None)
        if ema_gap_atr_mult is not None:
            gap = (df["ema_fast"] - df["ema_slow"]).abs()
            atr_for_gap = df["atr"] if "atr" in df.columns else pd.Series([1.0] * len(df), index=df.index)
            gap_ok = (gap >= ema_gap_atr_mult * atr_for_gap)
            quality_gate_base = quality_gate_base & gap_ok
    
    # ATR分位フィルタ
    if 'cfg' in locals() and cfg is not None:
        atr_quantile_enabled = cfg.get("atr_quantile_enabled", False)
        logger.info(f"ATR分位フィルタ設定確認: enabled={atr_quantile_enabled}")
        
        if atr_quantile_enabled:
            atr_q_lo = cfg.get("atr_quantile_lo", 0.15)
            atr_q_hi = cfg.get("atr_quantile_hi", 0.85)
            atr_q_window = cfg.get("atr_quantile_window", 500)
            logger.info(f"ATR分位フィルタを適用: lo={atr_q_lo}, hi={atr_q_hi}, window={atr_q_window}")
            atr_quantile_ok = _atr_quantile_filter(df, atr_period=14, atr_q_window=atr_q_window, atr_q_lo=atr_q_lo, atr_q_hi=atr_q_hi)
        else:
            logger.info("ATR分位フィルタは無効")
            atr_quantile_ok = pd.Series([True] * len(df), index=df.index)
    else:
        # cfgがNoneの場合はデフォルト値を使用
        logger.info("cfgがNoneのためATR分位フィルタは無効")
        atr_quantile_ok = pd.Series([True] * len(df), index=df.index)

    # ブロック内訳カウンタ（デバッグ用）
    global cnt
    cnt = {"cross_up": 0, "vol_reject": 0, "gap_reject": 0, "atr_reject": 0, "atr_quantile_reject": 0, "mtf_reject": 0, "session_reject": 0, "news_reject": 0, "pre_signal": 0, "enter": 0}
    
    # ロングシグナルのブロック内訳
    raw_up = cross_up.copy()
    raw_down = cross_dn.copy()
    cnt["cross_up"] = int(raw_up.sum())
    
    # 各フィルタのブロック数
    cnt["vol_reject"] = int((raw_up & ~vol_ok).sum())
    cnt["session_reject"] = int((raw_up & vol_ok & ~sess).sum())
    cnt["news_reject"] = int((raw_up & vol_ok & sess & news_excluded).sum())
    
    # 品質フィルタのブロック数
    if 'cfg' in locals() and cfg is not None:
        atr_floor_mult = cfg.get("atr_floor_mult", None)
        ema_gap_atr_mult = cfg.get("ema_gap_atr_mult", None)
        
        if atr_floor_mult is not None and "atr" in df.columns:
            atr_med = df["atr"].rolling(cfg.get("atr_floor_window", 2016), min_periods=1).median()
            atr_ok = (df["atr"] >= atr_floor_mult * atr_med)
            cnt["atr_reject"] = int((raw_up & vol_ok & sess & ~news_excluded & ~atr_ok).sum())
        
        if ema_gap_atr_mult is not None:
            gap = (df["ema_fast"] - df["ema_slow"]).abs()
            atr_for_gap = df["atr"] if "atr" in df.columns else pd.Series([1.0] * len(df), index=df.index)
            gap_ok = (gap >= ema_gap_atr_mult * atr_for_gap)
            cnt["gap_reject"] = int((raw_up & vol_ok & sess & ~news_excluded & gap_ok & ~gap_ok).sum())
    
    # ATR分位フィルタのブロック数（最終ゲート直前）
    pre_signal_mask = (cross_up & vol_ok & long_gate & sess & ~news_excluded & quality_gate_base)
    cnt["pre_signal"] = int(pre_signal_mask.sum())
    cnt["atr_quantile_reject"] = int((pre_signal_mask & ~atr_quantile_ok).sum())
    
    # MTFフィルタのブロック数
    cnt["mtf_reject"] = int((raw_up & vol_ok & sess & ~news_excluded & (long_gate == False)).sum())
    
    long_signal = cross_up & vol_ok & long_gate & sess & ~news_excluded & quality_gate_base & atr_quantile_ok
    short_signal = cross_dn & vol_ok & short_gate & sess & ~news_excluded & quality_gate_base & atr_quantile_ok

    want_long = direction in ("long","both")
    want_short = direction in ("short","both")

    in_pos = False
    side = 0  # +1 long, -1 short
    entry_price = None
    entry_time = None
    # エントリー時のATR値
    atr_entry_value = None
    breakeven_triggered = False  # ブレークイーブン発動フラグ
    be_armed = False  # ブレークイーブン準備フラグ
    trades = []

    for i in range(1, len(df)):
        ts = df.index[i]
        o,h,l,c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]

        # 金曜日強制クローズ（設定時間）
        try:
            if hasattr(ts, 'tz_convert'):
                ts_jst = ts.tz_convert("Asia/Tokyo")
            else:
                # tsがnumpy.int64の場合、pd.Timestampに変換
                ts_jst = pd.Timestamp(ts, tz="UTC").tz_convert("Asia/Tokyo")
            is_friday = ts_jst.weekday() == 4
        except Exception as e:
            logger.warning(f"タイムスタンプ変換エラー: {e}")
            # エラー時は金曜日判定をスキップ
            is_friday = False
        friday_close_h, friday_close_m = map(int, friday_close_time.split(":"))
        force_close = is_friday and (ts_jst.hour == friday_close_h and ts_jst.minute >= friday_close_m)

        if not in_pos:
            go_long = want_long and long_signal.iloc[i-1]
            go_short = want_short and short_signal.iloc[i-1]
            if go_long or go_short:
                side = 1 if go_long else -1
                # entry: 成行想定→スプレッド/スリッページ考慮
                if side == 1:
                    price_in = o + spread_pts + slippage_pts
                else:  # short entry at bid
                    price_in = o - spread_pts - slippage_pts
                entry_price = price_in
                entry_time = ts
                # エントリー時のATRを記録
                try:
                    atr_entry_value = float(df["atr"].iloc[i]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[i]) else None
                except Exception:
                    atr_entry_value = None
                in_pos = True
                cnt["enter"] += 1
        else:
            if side == 1:  # long
                # ブレークイーブン処理（+0.5Rで建値BE）
                be_after_r = cfg.get("breakeven_after_r", None) if 'cfg' in locals() and cfg is not None else None
                if be_after_r is not None and not be_armed:
                    # 進捗R（stop_pts基準）を評価
                    prog = (h - entry_price) / stop_pts if stop_pts > 0 else 0.0
                    if prog >= be_after_r:
                        be_armed = True
                
                # ストップ設定
                if be_armed:
                    sl = entry_price  # 建値BE
                else:
                    sl = entry_price - stop_pts
                
                # ATRベースの利確または固定利確
                if use_atr_tp and not pd.isna(df["atr"].iloc[i]):
                    tp = entry_price + (df["atr"].iloc[i] * atr_multiplier)
                else:
                    tp = entry_price + tp_pts
                hit_sl = l <= sl
                hit_tp = h >= tp
                exit_price = None
                exit_reason = None
                if hit_sl:
                    exit_price = sl - slippage_pts  # 悪化
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price = tp - slippage_pts  # 滑り
                    exit_reason = "TP"
                elif force_close or (session_policy == "flat" and not _session_mask(pd.DatetimeIndex([ts]), session_start, session_end).iloc[0]):
                    exit_price = c - slippage_pts
                    exit_reason = "SESSION_CLOSE"
                if exit_price is not None:
                    pnl_pts = (exit_price - (entry_price)) - spread_pts  # round-turnで実質もう半分のスプレッド
                    pnl_value = pnl_pts * point_value - commission_value
                    trades.append({
                        "entry_time": _safe_timestamp_to_iso(entry_time),
                        "exit_time": _safe_timestamp_to_iso(ts),
                        "entry": float(entry_price),
                        "exit": float(exit_price),
                        "pnl_pts": float(pnl_pts),
                        "pnl_value": float(pnl_value),
                        "atr_at_entry": float(atr_entry_value) if atr_entry_value is not None else None,
                        "reason": exit_reason,
                        "side": "LONG"
                    })
                    in_pos = False; entry_price=None; entry_time=None; side=0; atr_entry_value=None; breakeven_triggered=False; be_armed=False
            else:  # short
                # ブレークイーブン処理（-0.5Rで建値BE）
                be_after_r = cfg.get("breakeven_after_r", None) if 'cfg' in locals() and cfg is not None else None
                if be_after_r is not None and not be_armed:
                    # 進捗R（stop_pts基準）を評価
                    prog = (entry_price - l) / stop_pts if stop_pts > 0 else 0.0
                    if prog >= be_after_r:
                        be_armed = True
                
                # ストップ設定
                if be_armed:
                    sl = entry_price  # 建値BE
                else:
                    sl = entry_price + stop_pts
                
                # ATRベースの利確または固定利確
                if use_atr_tp and not pd.isna(df["atr"].iloc[i]):
                    tp = entry_price - (df["atr"].iloc[i] * atr_multiplier)
                else:
                    tp = entry_price - tp_pts
                hit_sl = h >= sl
                hit_tp = l <= tp
                exit_price = None
                exit_reason = None
                if hit_sl:
                    exit_price = sl + slippage_pts
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price = tp + slippage_pts
                    exit_reason = "TP"
                elif force_close or (session_policy == "flat" and not _session_mask(pd.DatetimeIndex([ts]), session_start, session_end).iloc[0]):
                    exit_price = c + slippage_pts
                    exit_reason = "SESSION_CLOSE"
                if exit_price is not None:
                    pnl_pts = ((entry_price) - exit_price) - spread_pts
                    pnl_value = pnl_pts * point_value - commission_value
                    trades.append({
                        "entry_time": _safe_timestamp_to_iso(entry_time),
                        "exit_time": _safe_timestamp_to_iso(ts),
                        "entry": float(entry_price),
                        "exit": float(exit_price),
                        "pnl_pts": float(pnl_pts),
                        "pnl_value": float(pnl_value),
                        "atr_at_entry": float(atr_entry_value) if atr_entry_value is not None else None,
                        "reason": exit_reason,
                        "side": "SHORT"
                    })
                    in_pos = False; entry_price=None; entry_time=None; side=0; atr_entry_value=None; breakeven_triggered=False

    # 残ポジはクローズ（終値）
    if in_pos:
        last_ts = df.index[-1]
        last_c = df["close"].iloc[-1]
        if side == 1:
            exit_price = last_c - slippage_pts
            pnl_pts = (exit_price - entry_price) - spread_pts
            pnl_value = pnl_pts * point_value - commission_value
            slabel = "LONG"
        else:
            exit_price = last_c + slippage_pts
            pnl_pts = (entry_price - exit_price) - spread_pts
            pnl_value = pnl_pts * point_value - commission_value
            slabel = "SHORT"
        trades.append({
            "entry_time": _safe_timestamp_to_iso(entry_time),
            "exit_time": _safe_timestamp_to_iso(last_ts),
            "entry": float(entry_price),
            "exit": float(exit_price),
            "pnl_pts": float(pnl_pts),
            "pnl_value": float(pnl_value),
            "atr_at_entry": float(atr_entry_value) if atr_entry_value is not None else None,
            "reason": "EOD_CLOSE",
            "side": slabel
        })

    trades_df = pd.DataFrame(trades)
    if len(trades_df)==0:
        return trades_df, pd.DataFrame(), {"PF":0,"MaxDD":0,"WinRate":0,"Trades":0}

    # エクイティ計算
    equity = trades_df["pnl_value"].cumsum()
    equity.index = pd.to_datetime(trades_df["exit_time"])
    equity = equity.asfreq("1min", method="pad")  # 視覚化用の1分補間
    peak = equity.cummax()
    dd = equity - peak
    maxdd = dd.min()

    # Metrics（ネット値ベース）
    wins = trades_df[trades_df["pnl_value"]>0]["pnl_value"].sum()
    losses = trades_df[trades_df["pnl_value"]<0]["pnl_value"].sum()
    PF = (wins / abs(losses)) if losses != 0 else float("inf")
    WinRate = (trades_df["pnl_value"]>0).mean()

    metrics = {
        "Trades": int(len(trades_df)),
        "PF": float(PF),
        "WinRate": float(WinRate),
        "NetProfit": float(trades_df["pnl_value"].sum()),
        "MaxDD": float(maxdd),
        "AvgR": float((trades_df["pnl_pts"]/abs(stop_pts)).mean()) if stop_pts!=0 else None
    }
    
    # ブロック内訳をメトリクスに追加
    if 'cnt' in globals():
        metrics.update(cnt)
        # 率の追加
        pre_sig = metrics.get("pre_signal", 0)
        metrics["atrq_reject_rate"] = (metrics.get("atr_quantile_reject", 0) / pre_sig) if pre_sig else 0.0
        cross = metrics.get("cross_up", 0)
        gate_mtf_reject = metrics.get("mtf_reject", 0)
        metrics["mtf_reject_rate"] = (gate_mtf_reject / cross) if cross else 0.0
    
    return trades_df, equity.to_frame(name="equity"), metrics

# -------------------- Walk-Forward --------------------

def walk_forward(df, cfg, n_splits=7, **bt_kw):
    idx = df.index
    splits = np.array_split(np.arange(len(idx)), n_splits)
    results = []
    pass_oos = 0
    for k in range(1, len(splits)):
        is_idx = splits[k-1]
        oos_idx = splits[k]
        is_df = df.iloc[is_idx]
        oos_df = df.iloc[oos_idx]
        # MTF方向
        hi_is = _build_hi_dir(is_df, cfg)
        hi_oos = _build_hi_dir(oos_df, cfg)
        # IS/OOS
        tr_is, _, m_is = backtest(is_df, cfg=cfg, **bt_kw)
        tr_oos, _, m_oos = backtest(oos_df, cfg=cfg, **bt_kw)
        np_is = m_is.get("NetProfit", 0.0)
        np_oos = m_oos.get("NetProfit", 0.0)
        wfe = (np_oos / np_is) if np_is != 0 else 0.0
        pf_oos = m_oos.get("PF", 0.0)
        results.append({"k":k, "IS_NetProfit":np_is, "OOS_NetProfit":np_oos, "WFE":wfe, "PF_OOS":pf_oos})
        if pf_oos >= 1.0:
            pass_oos += 1
    wf_rate = pass_oos / max(1, (len(splits)-1))
    return pd.DataFrame(results), float(wf_rate)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="backtest_config.json")
    ap.add_argument("--higher_tf_mode", choices=["resample","csv","off"], default=None,
                    help="Override cfg.higher_tf_mode")
    ap.add_argument("--higher_tf_csv_dir", default=None, help="Override cfg.higher_tf_csv_dir")
    ap.add_argument("--use_mtf", choices=["true","false"], default=None, help="Override cfg.use_mtf")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    if args.higher_tf_mode is not None:
        cfg["higher_tf_mode"] = args.higher_tf_mode
    if args.higher_tf_csv_dir is not None:
        cfg["higher_tf_csv_dir"] = args.higher_tf_csv_dir
    if args.use_mtf is not None:
        cfg["use_mtf"] = (args.use_mtf.lower() == "true")

    in_dir = cfg["input_dir"]
    out_dir = cfg["output_dir"]

    files = _find_input_files(in_dir)
    if not files:
        print("[ERROR] No CSV files found in:", in_dir)
        print("Put your OHLCV CSV(s) here and re-run:", in_dir)
        return

    # ロード＆結合（同一シンボル/同一列想定）
    dfs = []
    for p in files:
        df0 = _parse_csv(p, tz=cfg.get("tz","Asia/Tokyo"))
        dfs.append(df0)
    df = pd.concat(dfs, axis=0).sort_values("timestamp").reset_index(drop=True)

    # 期間フィルタ
    if cfg.get("start_date"):
        df = df[df["timestamp"] >= pd.Timestamp(cfg["start_date"]).tz_localize(cfg.get("tz","Asia/Tokyo"))]
    if cfg.get("end_date"):
        df = df[df["timestamp"] <= pd.Timestamp(cfg["end_date"]).tz_localize(cfg.get("tz","Asia/Tokyo"))]

    # リサンプル（低位足）
    low = _resample_ohlcv(df, rule=cfg.get("low_tf","5T"))
    low = low[(~low.index.duplicated())]

    # MTF方向
    hi_dir = _build_hi_dir(low, cfg)
    
    # MTF傾きフィルタ
    hi_slope = None
    if cfg.get("use_mtf", True) and cfg.get("use_slope_filter", False):
        hi_slope = _higher_tf_slope_filter(low, 
                                          rule_hi=cfg.get("higher_tf_rule", "60T"),
                                          fast=cfg.get("ema_fast", 9),
                                          slow=cfg.get("ema_slow", 21))
    
    # ニュース除外期間の読み込み
    news_exclusion_periods = _load_news_exclusions(
        cfg.get("news_exclude_csv"),
        cfg.get("news_exclusion_minutes", 30),
        cfg.get("tz", "UTC")
    )
    
    # 時間帯除外セッションの読み込み
    time_sessions = cfg.get("time_sessions", None)
    
    # セッションポリシーの読み込み
    session_policy = cfg.get("session_policy", "flat")

    # バックテスト本体
    trades, equity, metrics = backtest(
        df,
        ema_fast=cfg.get("ema_fast",9),
        ema_slow=cfg.get("ema_slow",21),
        vol_ratio_th=cfg.get("vol_ratio_th",1.5),
        stop_pts=cfg.get("stop_pts",1.0),
        tp_pts=cfg.get("tp_pts",2.0),
        point_value=cfg.get("point_value",1.0),
        spread_pts=cfg.get("spread_pts",0.0),
        slippage_pts=cfg.get("slippage_pts",0.0),
        commission_value=cfg.get("commission_value",0.0),
        direction=cfg.get("direction","long"),
        use_mtf=cfg.get("use_mtf",True),
        higher_tf_mode=cfg.get("higher_tf_mode","resample"),
        higher_tf_rule=cfg.get("higher_tf_rule","60T"),
        higher_tf_csv_dir=cfg.get("higher_tf_csv_dir",None),
        start_date=cfg.get("start_date",None),
        end_date=cfg.get("end_date",None),
        tz=cfg.get("tz","UTC"),
        session_start=cfg.get("session_start","09:00"),
        session_end=cfg.get("session_end","15:00"),
        friday_close_time=cfg.get("friday_close_time","15:05"),
        use_atr_tp=cfg.get("use_atr_tp",False),
        atr_multiplier=cfg.get("atr_multiplier",2.0),
        exclude_news=cfg.get("exclude_news",False),
        news_exclusion_minutes=cfg.get("news_exclusion_minutes",30),
        news_exclude_csv=cfg.get("news_exclude_csv",None),
        use_slope_filter=cfg.get("use_slope_filter",False),
        use_breakeven=cfg.get("use_breakeven",False),
        breakeven_r=cfg.get("breakeven_r",0.5),
        time_sessions=time_sessions,
        session_policy=session_policy,
        cfg=cfg,
    )

    # Walk-Forward（簡易）
    wf_df, wf_rate = walk_forward(
        df,
        cfg,
        n_splits=cfg.get("wf_splits",7),
        ema_fast=cfg.get("ema_fast",9),
        ema_slow=cfg.get("ema_slow",21),
        vol_ratio_th=cfg.get("vol_ratio_th",1.5),
        stop_pts=cfg.get("stop_pts",1.0),
        tp_pts=cfg.get("tp_pts",2.0),
        point_value=cfg.get("point_value",1.0),
        spread_pts=cfg.get("spread_pts",0.0),
        slippage_pts=cfg.get("slippage_pts",0.0),
        commission_value=cfg.get("commission_value",0.0),
        direction=cfg.get("direction","long"),
        use_mtf=cfg.get("use_mtf",True),
        higher_tf_mode=cfg.get("higher_tf_mode","resample"),
        higher_tf_rule=cfg.get("higher_tf_rule","60T"),
        higher_tf_csv_dir=cfg.get("higher_tf_csv_dir",None),
        start_date=cfg.get("start_date",None),
        end_date=cfg.get("end_date",None),
        tz=cfg.get("tz","UTC"),
        session_start=cfg.get("session_start","09:00"),
        session_end=cfg.get("session_end","15:00"),
        friday_close_time=cfg.get("friday_close_time","15:05"),
        use_atr_tp=cfg.get("use_atr_tp",False),
        atr_multiplier=cfg.get("atr_multiplier",2.0),
        exclude_news=cfg.get("exclude_news",False),
        news_exclusion_minutes=cfg.get("news_exclusion_minutes",30),
        news_exclude_csv=cfg.get("news_exclude_csv",None),
        use_slope_filter=cfg.get("use_slope_filter",False),
        use_breakeven=cfg.get("use_breakeven",False),
        breakeven_r=cfg.get("breakeven_r",0.5),
        time_sessions=time_sessions,
        session_policy=session_policy,
    )

    # タイムスタンプ付きファイル名生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    direction_str = cfg.get("direction", "long")
    spread_str = f"spread{cfg.get('spread_pts', 0.0)}"
    
    # 期間情報を追加
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    if start_date and end_date:
        period_str = f"{start_date[:4]}{start_date[5:7]}-{end_date[:4]}{end_date[5:7]}"
    else:
        period_str = "full"
    
    # バックテスト専用フォルダ作成
    test_folder_name = f"backtest_{direction_str}_{period_str}_{spread_str}_{timestamp}"
    test_folder_path = os.path.join(out_dir, test_folder_name)
    os.makedirs(test_folder_path, exist_ok=True)
    
    # 保存
    trades_path = os.path.join(test_folder_path, "trades.csv")
    equity_path = os.path.join(test_folder_path, "equity.csv")
    metrics_path = os.path.join(test_folder_path, "metrics.json")
    wf_path = os.path.join(test_folder_path, "walk_forward.csv")
    summary_path = os.path.join(test_folder_path, "summary.md")

    if len(trades)>0:
        trades.to_csv(trades_path, index=False)
        equity.to_csv(equity_path)
    wf_df.to_csv(wf_path, index=False)

    metrics_all = dict(metrics)
    metrics_all["WF_PassRate"] = wf_rate
    
    # ブロック内訳をメトリクスに追加
    if 'cnt' in globals():
        metrics_all.update(cnt)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# バックテスト要約\n")
        f.write(f"- 取引数: {metrics_all.get('Trades',0)}\n")
        f.write(f"- PF: {metrics_all.get('PF',0):.3f}\n")
        f.write(f"- WinRate: {metrics_all.get('WinRate',0):.3f}\n")
        f.write(f"- NetProfit: {metrics_all.get('NetProfit',0):.2f}\n")
        f.write(f"- MaxDD: {metrics_all.get('MaxDD',0):.2f}\n")
        f.write(f"- AvgR: {metrics_all.get('AvgR',0)}\n")
        f.write(f"- Walk-Forward合格率: {metrics_all.get('WF_PassRate',0):.2%}\n")

    # グラフ（データがあれば）
    if len(trades)>0:
        # Equity曲線
        plt.figure()
        equity["equity"].plot()
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        fig1_path = os.path.join(test_folder_path, "equity_curve.png")
        plt.savefig(fig1_path, dpi=150)
        plt.close()

        # ドローダウン
        peak = equity["equity"].cummax()
        dd = equity["equity"] - peak
        plt.figure()
        dd.plot()
        plt.title("Drawdown")
        plt.xlabel("Time")
        plt.ylabel("DD")
        fig2_path = os.path.join(test_folder_path, "drawdown.png")
        plt.savefig(fig2_path, dpi=150)
        plt.close()

    print(f"[OK] Created folder: {test_folder_name}")
    print(f"[OK] Wrote: {trades_path}, {equity_path}, {metrics_path}, {wf_path}, {summary_path}")

if __name__ == "__main__":
    main()
