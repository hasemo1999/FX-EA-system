---
description: トレーディングロジック変更時のルール
globs: "rough_backtest.py,session_filter.py"
---

# トレーディングロジック変更ルール

- シグナルフィルタのゲート順序を絶対に変えない: session → MTF → quality → ATR-quantile
- 新しいフィルタを追加する場合は ATR-quantile の前に挿入する
- パラメータ変更は必ず別の config ファイルで検証してから本番に反映
- Walk-Forward 7分割で OOS PF ≥ 1.0 を確認するまで本番設定を変更しない
- コスト計算（スプレッド・スリッページ・手数料）を省略しない
- タイムゾーン変換は必ず pytz を使う（naive datetime 禁止）
