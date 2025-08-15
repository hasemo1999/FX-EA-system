# EAプロジェクト 運用経緯・決定事項（2025-08-15）

- 背景/目標
  - 既存PF≈1.188→1.30を目指し、ATR分位フィルタで選択性ギャップ（fl−fw≥約6.6%）を確保する方針。
  - 理論: PF_new = PF_old * (1 - fw) / (1 - fl), 目標係数≥1.0943。

- 実測（バックテスト 2023-2024, spread 0.02）
  - (0.15,0.85): gap≈4.37% → PF推計≈1.191（未達）
  - (0.20,0.80): gap≈0.29% → PF推計≈1.125（未達）
  - (0.10,0.90): gap<0 → PF低下（未達）
  - グリッド走査（lo∈[0.10,0.30], hi∈[0.70,0.90]）で上位帯域を抽出。
    - PF'最良: (0.30,0.71) PF'≈1.321, gap≈6.98%, 除外≈58.9%
    - gap最良: (0.15,0.83) gap≈7.89%, PF'≈1.255, 除外≈31.9%

- プロファイル決定
  - 採用（B）: (0.30, 0.72) window=500, period=14
    - 実測: PF≈1.246, MaxDD≈-10.93, WF≈83.33%, Trades≈181, atrq_reject_rate≈0.593
  - バックアップ（#1）: (0.30, 0.71)
  - 一時退避（#2）: (0.15, 0.83)

- 実装/変更点（rough_backtest.py）
  - ATR分位フィルタを“最終ゲート”に適用（session→MTF→品質→ATR分位）。
  - カウンタ/メトリクス: pre_signal, atr_quantile_reject, enter, atrq_reject_rate, mtf_reject_rate。
  - タイムスタンプ/セッション/警告の堅牢化（tz処理, 'T'→'min', iloc警告対応）。
  - trades.csvに atr_at_entry を出力。

- 運用設定
  - `backtest_config_production.json`: lo=0.30, hi=0.72, window=500, period=14, friday_close_time=06:10(UTC)。
  - Slack日次サマリ: `slack_daily_summary.py`（OK/ALERT, 理由, Action/ExitCode含む）
  - 監視ロジック（優先度: MaxDD > atrq > WF）
    - atrq_reject_rate>0.65 → TEMP_SWITCH_PROFILE2（#2へ退避）
    - MaxDD< -10.93 → REVIEW_SWITCH（#1 or #2へ見直し）
    - WF<0.75（週次） → REOPTIMIZE_QUEUE
    - ExitCode: REVIEW_SWITCH=3, TEMP_SWITCH=2, REOPTIMIZE=1, OK=0

- 配置/自動化
  - OneDrive送信セット: `C:\Users\bnr39\OneDrive\EA通知\` 
    - SendSlackSummary.bat / send_summary_app.py / slack_daily_summary.py
    - 監視: EA_Monitor.ps1 / Run_EA_Monitor.bat（終了コードに準拠）
    - 鍵: slack_webhook.txt（1行目にWebhook, .urlも互換）
  - スケジューラ: Run_EA_Monitor.bat を日次実行。

- フォワード（MT4デモ）
  - レポート保存: `C:\Users\bnr39\OneDrive\EAフォワード\レポート\*.htm` 
  - 今後: HTMLレポート→集計→Slack配信を追加予定。

- リポジトリ
  - GitHub: hasemo1999/FX-EA-system（本決定とツール群を反映）。
