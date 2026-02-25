#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "tests" ]]; then
  echo "[tests] no tests directory found; skipping."
  exit 0
fi

echo "[tests] running unit and other non-integration suites."
base_args=(tests --maxfail=1)
if [[ -d "tests/integration" ]]; then
  base_args+=(--ignore=tests/integration)
fi
if [[ -d "tests/e2e" ]]; then
  base_args+=(--ignore=tests/e2e)
fi
uv run pytest "${base_args[@]}"

if [[ -d "tests/integration" ]]; then
  echo "[tests] running integration suite."
  uv run pytest tests/integration --maxfail=1
else
  echo "[tests] no integration suite found; skipping."
fi

if [[ -d "tests/e2e" ]]; then
  echo "[tests] running e2e suite."
  uv run pytest tests/e2e --maxfail=1
else
  echo "[tests] no e2e suite found; skipping."
fi
