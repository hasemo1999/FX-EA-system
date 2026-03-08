---
description: Python コードスタイルルール
globs: "*.py"
---

# コードスタイル

- Python 3.11+ の機能を使用してよい
- 全関数に型アノテーションを付ける（既存コードに準拠）
- pandas DataFrame操作は `.loc[]` / `.iloc[]` を使い、チェーンインデックスを避ける
- 変数名・関数名は snake_case（既存コードに準拠）
- コメントは日本語OK
- f-string を優先使用
- numpy/pandas の警告を出さないコードを書く（SettingWithCopyWarning等）
