---
name: backtest
description: バックテストを実行して結果を分析する
user_invocable: true
---

# /backtest スキル

バックテストを実行し、結果を分析します。

## 手順

1. 引数が指定されていない場合は `backtest_config_production.json` を使用
2. 引数が指定されている場合はそのconfigファイルを使用
3. 以下のコマンドを実行:
   ```bash
   python rough_backtest.py --cfg <config_file>
   ```
4. 結果から以下のKPIを抽出して報告:
   - Profit Factor（目標: ≥ 1.30）
   - 取引数
   - 勝率
   - 最大ドローダウン（目標: ≤ 10%）
   - Walk-Forward合格率（目標: ≥ 70%）
   - ATR Quantile リジェクト率
5. 目標KPIとの差分を明示する
6. 前回結果との比較がある場合は変化を報告
