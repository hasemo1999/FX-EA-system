# FX-EA-System プロジェクト指示書

プロジェクトの概要は @README.md を参照。
最適化の試行錯誤の記録は @.serena/memories/backtest_optimization_journey.md を参照。
本番ATRプロファイルの詳細は @.serena/memories/atrq_production_2025-08-15.md を参照。

## なぜこの構成なのか

- EMA 9/21クロス戦略を採用した理由：FXではシンプルなクロスオーバーが安定して機能し、過学習リスクが低い
- ATR Quantileゲートを最終フィルタにした理由：ボラティリティ・レジームを統計的に選別することでPFを1.188→1.246に改善できた
- Walk-Forward 7分割を採用した理由：OOS検証で過学習を防ぎ、83.33%の安定性を確認済み
- .serena/にAI記憶を保存している理由：最適化の試行錯誤の結果（何が効かなかったか）を失わないため

## 絶対にやってはいけないこと

- **breakeven stop（建値ストップ）を有効化するな** — 過去テストで勝率が40%低下し、トレード数が30%増加した。見た目は安全だが実績は悪化する
- **ニュース除外フィルタを追加するな** — テスト済み。EMA9/21シグナルと経済指標の時刻に重複がなく、効果ゼロだった
- **`rough_backtest.py`のゲート順序を変えるな** — session → MTF → quality → ATR-quantile の順番に意味がある。順序変更でフィルタの統計的意味が変わる
- **`backtest_config_production.json`を直接編集するな** — 新設定をテストするときは別ファイルを作り、Walk-Forwardで検証してから本番に反映する
- **CSVデータファイルをgitにコミットするな** — .gitignoreで除外済み。大容量でリポジトリが壊れる

## 過去のインシデントと教訓

| 施策 | 結果 | 教訓 |
|------|------|------|
| ニュース除外 | 効果ゼロ | シグナル発生時間帯と指標発表時間に重複なし |
| ブレークイーブンストップ | PF悪化 | 小さな利益を確定させすぎて全体収益が減少 |
| 時間帯除外(UTC 13:00-15:00) | PF改善だがMaxDD増加 | トレードオフが存在。MaxDD 10%超はNG |
| MTFスロープフィルタ | 実装問題あり | 理論は正しいが閾値チューニングが困難 |

## 現在の本番プロファイル

- **ATR Quantile Profile B**: lo=0.30, hi=0.72, window=500
- PF≈1.246 / MaxDD≈-10.93% / WF≈83.33% / Trades≈181
- バックアップ: Profile #1 (lo=0.30, hi=0.71) — PFほぼ同等、MaxDD若干改善
- フォールバック: Profile #2 (lo=0.15, hi=0.83) — PF≈1.221、MaxDD悪化

## Slack運用監視

- `slack_daily_summary.py` で日次メトリクスをSlack投稿
- **アラート閾値**:
  - atrq_reject_rate > 0.65 → Profile #2に切替検討
  - MaxDD < -10.93 → レビュー必要
  - WF < 75%（週次） → 再最適化
- 終了コード: OK=0, REOPTIMIZE=1, TEMP_SWITCH=2, REVIEW_SWITCH=3

## 目標KPI

| 指標 | 目標 | 現在値 | ギャップ |
|------|------|--------|----------|
| Profit Factor | ≥ 1.30 | 1.246 | -0.054 |
| MaxDD | ≤ 10% | 10.93% | +0.93% |
| Walk-Forward | ≥ 70% | 83.33% | 達成済み |

## 次の最適化候補（優先順）

1. MaxDDの抑制（Profile #1バックアップへの切替検討）
2. ATR quantile window サイズチューニング（300/1000を試す）
3. ショートポジションの追加（direction: "both"）で分散化
4. EUR/USD, GBP/USDへの通貨ペア拡張

## データとタイムゾーン

- 対象通貨: USD/JPY
- データ: 5分足OHLCV CSV（TradingView/OANDA/Dukascopy互換）
- バックテスト期間: 2023-01-01 〜 2024-12-31
- タイムゾーン: UTC基準、Asia/Tokyo変換対応
- 金曜クローズ: UTC 06:10（JST 15:10）
