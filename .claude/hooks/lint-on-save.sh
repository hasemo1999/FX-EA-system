#!/bin/bash
# PostToolUse hook: Auto-check Python files after Edit/Write
# Matcher: Edit|Write
#
# Runs Python syntax check on .py files.
# Reports errors back to Claude via transcript.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '
  .tool_input.file_path //
  .tool_input.filePath //
  .tool_result.filePath //
  .tool_result.file_path //
  empty
')

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

# Only check Python files
case "$FILE_PATH" in
  *.py)
    ;;
  *)
    exit 0
    ;;
esac

# Python syntax check
RESULT=$(python -m py_compile "$FILE_PATH" 2>&1) || true
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  jq -n --arg msg "$RESULT" --arg file "$FILE_PATH" '{
    hookSpecificOutput: {
      hookEventName: "PostToolUse"
    },
    transcript: ("⚠️ Python syntax error in " + $file + ":\n" + $msg)
  }'
fi

exit 0
