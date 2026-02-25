#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".github/workflows" ]]; then
  echo "[actionlint] no workflows directory found; skipping."
  exit 0
fi

if command -v actionlint >/dev/null 2>&1; then
  actionlint -color
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  uv run actionlint -color
  exit 0
fi

echo "[actionlint] command not found. Install actionlint or run via uv."
exit 1
