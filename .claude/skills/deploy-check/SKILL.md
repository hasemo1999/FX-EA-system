---
name: deploy-check
description: 本番デプロイ前のチェックリスト確認
user_invocable: true
---

# /deploy-check スキル

設定変更を本番に反映する前の安全チェックを実行します。

## チェックリスト

1. **Walk-Forward検証**: OOS PF ≥ 1.0 が7分割中5分割以上で達成されているか
2. **MaxDD確認**: 最大ドローダウンが10%以内か
3. **トレード数確認**: 十分なサンプルサイズ（150トレード以上）があるか
4. **設定差分**: 本番設定との差分を表示
   ```bash
   diff backtest_config_production.json <新設定ファイル>
   ```
5. **Slack監視閾値との整合性**: atrq_reject_rate が0.65以下か
6. **バックアッププラン**: 切替後にKPIが悪化した場合のロールバック手順を確認

## 出力

全項目がPASSの場合のみ「デプロイ可能」と報告。
1つでもFAILがあれば理由を明示し、デプロイを推奨しない。
