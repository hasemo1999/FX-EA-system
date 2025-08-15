"""
analyze_atr_gap.py

約定履歴CSVから、ATR分位フィルタ (lo, hi) による除外がもたらす
fw（失われるGP割合）, fl（削れるGL割合）, gap = fl - fw を算出するユーティリティ。

前提:
- trades.csv が以下のいずれかの列を持つこと:
  - pnl_value（推奨。なければ pnl_pts でも可）
  - atr_percentile_at_entry（0.0-1.0）。無ければ atr_at_entry を使用して分位を算出
  - どちらも無い場合は、--ohlcv を指定してOHLCVから ATR を再構成して使用可能

分位の算出方針:
- atr_percentile_at_entry が無い場合は、trades.csv 内の atr_at_entry の全サンプル分布に対する
  percent rank（pandas.Series.rank(pct=True)）で近似します。

出力:
- コンソールに以下をINFOで表示
  - N, N_removed, remove_rate
  - GP, GL, fw, fl, gap
  - PF_old, PF_new_pred (理論式 PF_new = PF_old * (1 - fw) / (1 - fl))

使い方:
  python analyze_atr_gap.py --trades "path/to/trades.csv" --lo 0.15 --hi 0.85 \
         --pnl-col pnl_value --atr-pct-col atr_percentile_at_entry --atr-col atr_at_entry

注: atr列がどちらも無い場合は、--ohlcv を指定してOHLCVからの再計算に対応します。
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_trades_csv(path: str) -> pd.DataFrame:
    """trades.csv を読み込む。
    最低限、以下の列のどれかが必要: pnl_value または pnl_pts。
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"trades.csv の読み込みに失敗: {e}")
        raise


def ensure_pnl_column(df: pd.DataFrame, pnl_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """pnl列名を決定して返す。無ければエラー。"""
    if pnl_col is not None and pnl_col in df.columns:
        return df, pnl_col
    # 自動推定
    for cand in ("pnl_value", "pnl_pts"):
        if cand in df.columns:
            return df, cand
    raise ValueError("pnl列が見つかりません。'pnl_value' または 'pnl_pts' が必要です。")


def get_or_build_atr_percentile(
    df: pd.DataFrame,
    atr_pct_col: Optional[str],
    atr_col: Optional[str]
) -> pd.Series:
    """分位シリーズを返す。
    - atr_pct_col があればそれを使用（0.0-1.0想定）。
    - 無ければ atr_col の rank(pct=True) で近似して作る。
    - どちらも無ければエラー。
    """
    if atr_pct_col and atr_pct_col in df.columns:
        pct = pd.to_numeric(df[atr_pct_col], errors='coerce')
        if pct.isna().all():
            raise ValueError(f"指定の atr_percentile 列 '{atr_pct_col}' がすべてNaNです。")
        # 0-1にクリップ
        pct = pct.clip(lower=0.0, upper=1.0)
        return pct

    if atr_col and atr_col in df.columns:
        atr_vals = pd.to_numeric(df[atr_col], errors='coerce')
        if atr_vals.isna().all():
            raise ValueError(f"指定の atr 列 '{atr_col}' がすべてNaNです。")
        # percent rank（同値は平均ランク）。0-1。
        pct = atr_vals.rank(pct=True)
        pct = pct.clip(lower=0.0, upper=1.0)
        return pct

    raise ValueError("'atr_percentile_at_entry' も 'atr_at_entry' も見つかりません。")


def load_ohlcv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """OHLCV CSVをロバストに読み込み、timestampをtz付きにする。"""
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    # 推定: 時刻列
    ts_candidates = ("timestamp","time","datetime","date","Time","UTC","Time (UTC)")
    ts_col = None
    for c in ts_candidates:
        if c.lower() in cols_lower:
            ts_col = cols_lower[c.lower()]
            break
    if ts_col is None:
        raise ValueError("OHLCV: timestamp列が見つかりません。")
    ts = pd.to_datetime(df[ts_col], errors='coerce')
    if ts.isna().any():
        raise ValueError("OHLCV: timestampのパースに失敗しました。")
    if getattr(ts.dt, 'tz', None) is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    # OHLC
    def find_col(names):
        for n in names:
            if n in df.columns:
                return n
        return None
    open_col = find_col(["open","Open","o"])
    high_col = find_col(["high","High","h"])
    low_col  = find_col(["low","Low","l"])
    close_col= find_col(["close","Close","c"])
    if not all([open_col,high_col,low_col,close_col]):
        raise ValueError("OHLCV: open/high/low/close の列が見つかりません。")
    odf = pd.DataFrame({
        'open': pd.to_numeric(df[open_col], errors='coerce'),
        'high': pd.to_numeric(df[high_col], errors='coerce'),
        'low':  pd.to_numeric(df[low_col], errors='coerce'),
        'close':pd.to_numeric(df[close_col], errors='coerce'),
    })
    odf['timestamp'] = ts
    odf = odf.dropna(subset=['open','high','low','close'])
    odf = odf.sort_values('timestamp').reset_index(drop=True)
    odf = odf.set_index('timestamp')
    return odf


def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """SMA版ATR。"""
    high_low = ohlcv['high'] - ohlcv['low']
    high_close = (ohlcv['high'] - ohlcv['close'].shift()).abs()
    low_close = (ohlcv['low'] - ohlcv['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    atr.name = 'atr'
    return atr


def build_atr_and_quantiles_for_entries(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    atr_period: int = 14,
    q_window: int = 500,
    lo: float = 0.15,
    hi: float = 0.85,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """エントリー時刻に対する ATR値, q_lo, q_hi を返す。quantilesは直近windowで1本シフト。"""
    atr = compute_atr(ohlcv, period=atr_period)
    q_lo_series = atr.rolling(q_window).quantile(lo).shift(1)
    q_hi_series = atr.rolling(q_window).quantile(hi).shift(1)
    # エントリーにアライン（直近過去の値で埋める）
    atr_at_entry = atr.reindex(entries, method='pad')
    qlo_at_entry = q_lo_series.reindex(entries, method='pad')
    qhi_at_entry = q_hi_series.reindex(entries, method='pad')
    return atr_at_entry, qlo_at_entry, qhi_at_entry


def compute_fw_fl_gap(
    trades: pd.DataFrame,
    pnl_col: str,
    atr_pct: pd.Series,
    lo: float,
    hi: float
) -> Tuple[float, float, float, float, float, int, int]:
    """fw, fl, gap と PF_old, PF_new_pred, N_total, N_removed を返す。"""
    # 元PF用のGP, GL
    pnl = pd.to_numeric(trades[pnl_col], errors='coerce')
    pnl = pnl.fillna(0.0)
    gp_all = pnl[pnl > 0].sum()
    gl_all = pnl[pnl < 0].sum()  # 負値
    gl_abs_all = abs(gl_all)

    pf_old = (gp_all / gl_abs_all) if gl_abs_all > 0 else float('inf')

    # 帯域外（除外対象）
    out_band = (atr_pct < lo) | (atr_pct > hi)
    n_total = len(trades)
    n_removed = int(out_band.sum())

    gp_removed = pnl[(pnl > 0) & out_band].sum()
    gl_removed_abs = abs(pnl[(pnl < 0) & out_band].sum())

    fw = (gp_removed / gp_all) if gp_all > 0 else 0.0
    fl = (gl_removed_abs / gl_abs_all) if gl_abs_all > 0 else 0.0
    gap = fl - fw

    # 理論PF
    pf_new_pred = (pf_old * (1.0 - fw) / (1.0 - fl)) if (1.0 - fl) != 0 else float('inf')

    return float(fw), float(fl), float(gap), float(pf_old), float(pf_new_pred), int(n_total), int(n_removed)


def main() -> None:
    parser = argparse.ArgumentParser(description="ATR Quantileによる fw/fl/gap を算出")
    parser.add_argument('--trades', required=True, help='trades.csv のパス')
    parser.add_argument('--lo', type=float, default=0.15, help='下側分位（0-1）')
    parser.add_argument('--hi', type=float, default=0.85, help='上側分位（0-1）')
    parser.add_argument('--pnl-col', default=None, help='PnL列名（未指定なら pnl_value→pnl_pts の順に自動）')
    parser.add_argument('--atr-pct-col', default='atr_percentile_at_entry', help='ATR分位列名')
    parser.add_argument('--atr-col', default='atr_at_entry', help='ATR値列名（分位列が無い場合の近似に使用）')
    parser.add_argument('--ohlcv', default=None, help='OHLCV CSVのパス（tradesにATR列が無い場合に使用）')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR期間')
    parser.add_argument('--q-window', type=int, default=500, help='分位計算ウィンドウ')
    parser.add_argument('--tz', default='UTC', help='OHLCVのタイムゾーン（未指定時ローカライズ）')
    args = parser.parse_args()

    try:
        trades = load_trades_csv(args.trades)
        trades, pnl_col = ensure_pnl_column(trades, args.pnl_col)
        # エントリー時刻
        if 'entry_time' not in trades.columns:
            raise ValueError("trades.csv に 'entry_time' 列が必要です。")
        entry_times = pd.to_datetime(trades['entry_time'], errors='coerce')
        if entry_times.isna().any():
            raise ValueError("'entry_time' のパースに失敗した行があります。")
        if getattr(entry_times.dt, 'tz', None) is None:
            entry_times = entry_times.dt.tz_localize('UTC')

        # ATR分位の取得（列がある場合）
        atr_pct: Optional[pd.Series] = None
        try:
            atr_pct = get_or_build_atr_percentile(trades, args.atr_pct_col, args.atr_col)
        except Exception:
            atr_pct = None

        if atr_pct is None:
            if not args.ohlcv:
                raise ValueError("'atr_percentile_at_entry' も 'atr_at_entry' も無く、--ohlcv も未指定です。計算できません。")
            ohlcv = load_ohlcv(args.ohlcv, tz=args.tz)
            atr_at_entry, qlo_at_entry, qhi_at_entry = build_atr_and_quantiles_for_entries(
                ohlcv, entry_times, atr_period=args.atr_period, q_window=args.q_window, lo=args.lo, hi=args.hi
            )
            # 帯域内判定を直接行う（分位値がNaNの場合はTrue=許可）
            within = (atr_at_entry >= qlo_at_entry) & (atr_at_entry <= qhi_at_entry)
            # NaNは許可（除外しない）にする
            within = within.fillna(True)
            # out_bandを作成し、tradesの行順にインデックスを合わせる
            out_band = (~within)
            # 整合性: tradesのindexに合わせる（位置ベースで貼り付け）
            if len(out_band) != len(trades):
                raise ValueError("内部整合性エラー: マスク長とトレード件数が一致しません。")
            out_band = pd.Series(out_band.to_numpy(dtype=bool), index=trades.index)
            pnl = pd.to_numeric(trades[pnl_col], errors='coerce').fillna(0.0)
            gp_all = pnl[pnl > 0].sum()
            gl_abs_all = abs(pnl[pnl < 0].sum())
            gp_removed = pnl[(pnl > 0) & out_band].sum()
            gl_removed_abs = abs(pnl[(pnl < 0) & out_band].sum())
            fw = (gp_removed / gp_all) if gp_all > 0 else 0.0
            fl = (gl_removed_abs / gl_abs_all) if gl_abs_all > 0 else 0.0
            gap = fl - fw
            pf_old = (gp_all / gl_abs_all) if gl_abs_all > 0 else float('inf')
            pf_new_pred = (pf_old * (1.0 - fw) / (1.0 - fl)) if (1.0 - fl) != 0 else float('inf')
            n_total = len(trades)
            n_removed = int(out_band.sum())

            remove_rate = n_removed / max(1, n_total)
            logger.info(f"N={n_total}, Removed={n_removed} ({remove_rate*100:.2f}%)")
            logger.info(f"GP={trades[pnl_col][trades[pnl_col]>0].sum():.4f}, GL={trades[pnl_col][trades[pnl_col]<0].sum():.4f}")
            logger.info(f"fw={fw:.4f}, fl={fl:.4f}, gap={gap:.4f}")
            logger.info(f"PF_old={pf_old:.3f} -> PF_new_pred={pf_new_pred:.3f}")

            required_ratio = 1.30 / 1.188
            achieved_ratio = (1.0 - fw) / (1.0 - fl) if (1.0 - fl) != 0 else np.inf
            ok = achieved_ratio >= required_ratio
            logger.info(f"達成判定: ((1-fw)/(1-fl))={achieved_ratio:.6f} vs 必要 {required_ratio:.6f} => {'OK' if ok else 'NG'}")
            return

        # ここに来るのは atr_pct が得られた場合
        atr_pct = atr_pct

        fw, fl, gap, pf_old, pf_new_pred, n_total, n_removed = compute_fw_fl_gap(
            trades, pnl_col, atr_pct, args.lo, args.hi
        )

        remove_rate = n_removed / max(1, n_total)

        logger.info(f"N={n_total}, Removed={n_removed} ({remove_rate*100:.2f}%)")
        logger.info(f"GP={trades[pnl_col][trades[pnl_col]>0].sum():.4f}, GL={trades[pnl_col][trades[pnl_col]<0].sum():.4f}")
        logger.info(f"fw={fw:.4f}, fl={fl:.4f}, gap={gap:.4f}")
        logger.info(f"PF_old={pf_old:.3f} -> PF_new_pred={pf_new_pred:.3f}")

        # 判定ライン 1.188 -> 1.30 のための係数 1.094276...
        required_ratio = 1.30 / 1.188
        achieved_ratio = (1.0 - fw) / (1.0 - fl) if (1.0 - fl) != 0 else np.inf
        ok = achieved_ratio >= required_ratio
        logger.info(f"達成判定: ((1-fw)/(1-fl))={achieved_ratio:.6f} vs 必要 {required_ratio:.6f} => {'OK' if ok else 'NG'}")

    except Exception as e:
        logger.error(f"解析失敗: {e}")
        raise


if __name__ == '__main__':
    main()


