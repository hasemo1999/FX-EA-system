# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import Iterable, Tuple, Optional

TZ_DEFAULT = "Asia/Tokyo"

def ensure_tz(index: pd.DatetimeIndex, tz: str = TZ_DEFAULT) -> pd.DatetimeIndex:
    """tz-naive/aware混在エラーの根治。naive→localize、別TZ→convert。"""
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)

def _to_min(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def build_session_mask(
    index: pd.DatetimeIndex,
    sessions: Iterable[Tuple[str, str]],
    tz: str = TZ_DEFAULT,
    weekdays: Optional[Iterable[int]] = None,   # 0=Mon ... 6=Sun
    strict: bool = True,
) -> pd.Series:
    """
    sessions: [("09:00","11:30"), ("12:30","15:10")] のように複数可。
    ・end < start を許容（日跨ぎセッション）
    ・strict=True: 入力が壊れていたらフェイルクローズ（全部False）
      strict=False: フェイルオープン（全部True）
    """
    try:
        idx = ensure_tz(index, tz)
        if weekdays is not None:
            wd_ok = idx.weekday.isin(list(weekdays))
        else:
            wd_ok = pd.Series(True, index=idx)

        # 分解能に依らず安全な「分」化で判定
        minutes = idx.hour * 60 + idx.minute

        mask_any = pd.Series(False, index=idx)
        for start, end in sessions:
            s = _to_min(start); e = _to_min(end)
            if s <= e:
                # 同日内
                m = (minutes >= s) & (minutes < e)
            else:
                # 日跨ぎ: 例) 22:30-02:30 → (>=22:30) OR (<02:30)
                m = (minutes >= s) | (minutes < e)
            mask_any = mask_any | m

        mask = mask_any & wd_ok

        # 安全装置: 全Falseや全Trueならログ目的で戻りを調整
        true_ratio = float(mask.sum()) / max(1, len(mask))
        # セッション指定があるのに 0% なら入力エラーの可能性大 → フェイルセーフ（24h開放）
        if sessions and true_ratio < 0.005:
            # フェイルオープンで先に進める or 厳格に止める
            return pd.Series(False, index=idx) if strict else pd.Series(True, index=idx)
        return mask
    except Exception:
        # どんな例外でも、strictなら全部False、非strictなら全部Trueで逃がす
        return pd.Series(False, index=index) if strict else pd.Series(True, index=index)

def apply_session_filter(df: pd.DataFrame,
                         sessions: Iterable[Tuple[str, str]],
                         tz: str = TZ_DEFAULT,
                         weekdays: Optional[Iterable[int]] = None,
                         strict: bool = False) -> pd.DataFrame:
    """
    df.index: DatetimeIndex 前提。エラー時は strict に応じて 24h 退避（False=開放）。
    """
    mask = build_session_mask(df.index, sessions, tz=tz, weekdays=weekdays, strict=strict)
    return df[mask]
