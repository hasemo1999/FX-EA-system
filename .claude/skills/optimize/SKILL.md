---
name: optimize
description: ATR Quantileパラメータのグリッドサーチを実行
user_invocable: true
---

# /optimize スキル

ATR Quantileパラメータの最適化を実行します。

## 手順

1. `atrq_grid_scan.py` を使用してグリッドサーチを実行:
   ```bash
   python atrq_grid_scan.py
   ```
2. 結果から上位5つのプロファイルを抽出
3. 各プロファイルについて以下を報告:
   - atr_quantile_lo / atr_quantile_hi
   - Profit Factor
   - MaxDD
   - Trade count
   - Walk-Forward合格率
4. 現在の本番プロファイル（Profile B: lo=0.30, hi=0.72）との比較
5. 推奨アクションを提示（切替/維持/追加検証）
6. **注意**: 本番設定の変更はユーザーの明示的な承認が必要
