#!/bin/bash
# PreToolUse hook: Block edits to protected files
# Matcher: Edit|Write
#
# Prevents modification of production config, lock files, and generated code.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '
  .tool_input.file_path //
  .tool_input.filePath //
  empty
')

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

deny() {
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 0
}

# Production config — must not be edited directly
case "$FILE_PATH" in
  *backtest_config_production.json)
    deny "Blocked: 本番設定ファイルの直接編集は禁止です。別ファイル(backtest_config_test_*.json)で検証してください。"
    ;;
esac

# Secrets files
case "$FILE_PATH" in
  *slack_webhook.txt|*.env|*.env.*|*.key|*.pem)
    deny "Blocked: シークレットファイルの編集はClaude Codeでは行わないでください。"
    ;;
esac

# .serena memory files — historical records should not be modified
case "$FILE_PATH" in
  *.serena/memories/*)
    deny "Blocked: .serena/memories/ は過去の記録です。新しいファイルを追加してください。"
    ;;
esac

# Lock files
case "$FILE_PATH" in
  */pnpm-lock.yaml|*/package-lock.json|*/yarn.lock|*/poetry.lock)
    deny "Blocked: lock files should not be edited manually."
    ;;
esac

# Generated files
case "$FILE_PATH" in
  */generated/*|*/.generated.*|*/dist/*|*/build/*)
    deny "Blocked: this is a generated file. Edit the source instead."
    ;;
esac

exit 0
