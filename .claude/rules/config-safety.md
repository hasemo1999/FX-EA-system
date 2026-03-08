---
description: 設定ファイル操作時の安全ルール
globs: "backtest_config*.json"
---

# 設定ファイル安全ルール

- `backtest_config_production.json` は直接編集しない
- 新設定をテストするときは `backtest_config_test_<目的>.json` を作成する
- 設定ファイルには以下の必須フィールドが含まれている必要がある:
  - ema_fast, ema_slow, vol_ratio_th, stop_pts, tp_pts
  - use_mtf, session_start, session_end
- ATR quantile を有効にする場合は atr_quantile_lo, atr_quantile_hi, atr_quantile_window も必須
- JSON のフォーマットは既存ファイルと統一する（インデント2スペース）
