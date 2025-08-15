"""
atrq_grid_scan.py

結果ファイル（trades.csv 等）だけを使って、ATR分位フィルタの帯域 (lo, hi)
をグリッド走査し、fw/fl/gap と PF の推定（PF_est）および残存取引による直計算
PF（PF_prime）を評価する。

入力場所（優先順）:
  C:\mnt\data\backtest_trades.csv
  C:\mnt\data\trades.csv
  C:\mnt\data\results.csv
  C:\mnt\data\fills.csv
  C:\mnt\data\executions.csv
  C:\mnt\data\orders_fills.csv

必要列:
  - pnl（>0 勝ち, <0 負け）
  - atr_percentile_at_entry（0.0〜1.0） もしくは atr_at_entry（数値）
    ※ 分位列がない場合は atr_at_entry を全履歴に対して rank(pct=True) で近似

出力:
  - C:\mnt\data\atrq_grid_results_{timestamp}.csv（全候補）
  - C:\mnt\data\atrq_top_by_pf_{timestamp}.csv（PF_prime 上位）
  - C:\mnt\data\atrq_top_by_gap_{timestamp}.csv（gap 上位）
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_CANDIDATE_FILES: List[str] = [
    r"C:\\mnt\\data\\backtest_trades.csv",
    r"C:\\mnt\\data\\trades.csv",
    r"C:\\mnt\\data\\results.csv",
    r"C:\\mnt\\data\\fills.csv",
    r"C:\\mnt\\data\\executions.csv",
    r"C:\\mnt\\data\\orders_fills.csv",
]


def pick_existing_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_trades(path: str) -> pd.DataFrame:
    """結果CSVを読み込む。最低限 'pnl' 列が必要。"""
    df = pd.read_csv(path)
    # pnl が無い場合、よくある別名から補完
    if 'pnl' not in df.columns:
        for cand in ('pnl_value','pnl_pts','profit','pnl_value_net'):
            if cand in df.columns:
                df = df.rename(columns={cand:'pnl'})
                break
    if 'pnl' not in df.columns:
        raise ValueError("CSVに 'pnl' 列がありません。")
    return df


def ensure_atr_percentile(df: pd.DataFrame) -> pd.Series:
    """分位列を返す。存在しなければ atr_at_entry から rank(pct=True) で近似。"""
    if 'atr_percentile_at_entry' in df.columns:
        pct = pd.to_numeric(df['atr_percentile_at_entry'], errors='coerce').clip(0.0, 1.0)
        if not pct.isna().all():
            return pct
    if 'atr_at_entry' in df.columns:
        atr_vals = pd.to_numeric(df['atr_at_entry'], errors='coerce')
        pct = atr_vals.rank(pct=True).clip(0.0, 1.0)
        return pct
    raise ValueError("'atr_percentile_at_entry' も 'atr_at_entry' も見つかりません。")


def compute_metrics_for_band(
    pnl: pd.Series,
    atr_pct: pd.Series,
    lo: float,
    hi: float,
    target_pf: float,
    pf_old: float,
) -> Tuple[float, float, float, float, float, float, float, int, int]:
    """
    指定帯域で fw, fl, gap, remove_rate, pf_est, pf_prime を計算。
    戻り値: (fw, fl, gap, remove_rate, pf_est, pf_prime, achieved_ratio, n_total, n_removed)
    """
    out_band = (atr_pct < lo) | (atr_pct > hi)
    n_total = len(pnl)
    n_removed = int(out_band.sum())
    remove_rate = n_removed / max(1, n_total)

    gp_all = pnl[pnl > 0].sum()
    gl_abs_all = abs(pnl[pnl < 0].sum())

    gp_removed = pnl[(pnl > 0) & out_band].sum()
    gl_removed_abs = abs(pnl[(pnl < 0) & out_band].sum())

    fw = (gp_removed / gp_all) if gp_all > 0 else 0.0
    fl = (gl_removed_abs / gl_abs_all) if gl_abs_all > 0 else 0.0
    gap = fl - fw

    # 理論PF
    pf_est = (pf_old * (1.0 - fw) / (1.0 - fl)) if (1.0 - fl) != 0 else float('inf')
    achieved_ratio = ((1.0 - fw) / (1.0 - fl)) if (1.0 - fl) != 0 else np.inf

    # 直計算PF（残存）
    in_band = ~out_band
    wins = pnl[(pnl > 0) & in_band].sum()
    losses_abs = abs(pnl[(pnl < 0) & in_band].sum())
    pf_prime = (wins / losses_abs) if losses_abs > 0 else float('inf')

    return float(fw), float(fl), float(gap), float(remove_rate), float(pf_est), float(pf_prime), float(achieved_ratio), n_total, n_removed


def main() -> None:
    parser = argparse.ArgumentParser(description="ATR分位のグリッド走査（結果CSVのみ）")
    parser.add_argument('--file', default=None, help='入力CSVのパス（未指定ならデフォルト候補から自動選択）')
    parser.add_argument('--lo-start', type=float, default=0.10)
    parser.add_argument('--lo-end', type=float, default=0.30)
    parser.add_argument('--hi-start', type=float, default=0.70)
    parser.add_argument('--hi-end', type=float, default=0.90)
    parser.add_argument('--step', type=float, default=0.01)
    parser.add_argument('--target-pf', type=float, default=1.30)
    args = parser.parse_args()

    try:
        in_path = args.file or pick_existing_file(DEFAULT_CANDIDATE_FILES)
        if not in_path:
            raise FileNotFoundError("入力CSVが見つかりません。C:\\mnt\\data に配置してください。")
        logger.info(f"入力: {in_path}")

        df = load_trades(in_path)
        pnl = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
        pf_old = (pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl[pnl < 0].sum() != 0) else float('inf')

        atr_pct = ensure_atr_percentile(df)
        # 整合性（長さ）
        if len(atr_pct) != len(pnl):
            raise ValueError("内部整合性エラー: atr分位とpnlの長さが一致しません。")

        lo_values = np.round(np.arange(args.lo_start, args.lo_end + 1e-9, args.step), 4)
        hi_values = np.round(np.arange(args.hi_start, args.hi_end + 1e-9, args.step), 4)

        records = []
        for lo in lo_values:
            for hi in hi_values:
                if hi <= lo:
                    continue
                fw, fl, gap, remove_rate, pf_est, pf_prime, achieved_ratio, n_total, n_removed = compute_metrics_for_band(
                    pnl, atr_pct, float(lo), float(hi), args.target_pf, pf_old
                )
                records.append({
                    'lo': float(lo), 'hi': float(hi), 'remove_rate': remove_rate,
                    'fw': fw, 'fl': fl, 'gap': gap,
                    'pf_old': float(pf_old), 'pf_est': pf_est, 'pf_prime': pf_prime,
                    'achieved_ratio': achieved_ratio,
                    'meets_pf_est': pf_est >= args.target_pf,
                    'meets_pf_prime': pf_prime >= args.target_pf,
                    'n_total': int(n_total), 'n_removed': int(n_removed),
                })

        res = pd.DataFrame.from_records(records)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = r"C:\\mnt\\data"
        os.makedirs(out_dir, exist_ok=True)
        all_path = os.path.join(out_dir, f"atrq_grid_results_{ts}.csv")
        res.to_csv(all_path, index=False, encoding='utf-8')

        top_pf = res.sort_values(['pf_prime','gap'], ascending=[False, False]).head(30)
        top_gap = res.sort_values(['gap','pf_prime'], ascending=[False, False]).head(30)
        top_pf_path = os.path.join(out_dir, f"atrq_top_by_pf_{ts}.csv")
        top_gap_path = os.path.join(out_dir, f"atrq_top_by_gap_{ts}.csv")
        top_pf.to_csv(top_pf_path, index=False, encoding='utf-8')
        top_gap.to_csv(top_gap_path, index=False, encoding='utf-8')

        # 画面に上位を簡易表示
        print("Top by PF' (pf_prime):")
        print(top_pf[['lo','hi','pf_prime','pf_est','gap','remove_rate']].to_string(index=False))
        print()
        print("Top by gap:")
        print(top_gap[['lo','hi','gap','pf_prime','pf_est','remove_rate']].to_string(index=False))
        print()
        print(f"[OK] Wrote: {all_path}, {top_pf_path}, {top_gap_path}")

    except Exception as e:
        logger.error(f"グリッド走査失敗: {e}")
        raise


if __name__ == '__main__':
    main()


