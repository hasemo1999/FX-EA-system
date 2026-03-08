---
description: セキュリティルール
globs: "**/*"
---

# セキュリティ

- Slack webhook URL はコードにハードコードしない（`slack_webhook.txt` から読み込む）
- `slack_webhook.txt` は絶対にgitにコミットしない
- CSVデータファイルのパスにWindowsのユーザー名が含まれる場合、ログ出力やコミットに注意
- API キー・トークン・認証情報はすべて環境変数または .env ファイルから読み込む
- .env, slack_webhook.txt, *.key は .gitignore に追加されていること
