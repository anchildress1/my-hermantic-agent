#!/usr/bin/env bash
set -euo pipefail

staged_files="$(git diff --cached --name-only)"

refresh_uv=false
refresh_npm=false

if [[ -f "pyproject.toml" && -f "uv.lock" ]]; then
  if printf '%s\n' "${staged_files}" | grep -Eq '^(pyproject\.toml|uv\.lock)$'; then
    refresh_uv=true
  fi
fi

if [[ -f "package.json" ]]; then
  if printf '%s\n' "${staged_files}" | grep -Eq '^(package\.json|package-lock\.json|npm-shrinkwrap\.json)$'; then
    refresh_npm=true
  fi
fi

if [[ "${refresh_uv}" == "false" && "${refresh_npm}" == "false" ]]; then
  echo "[lockfiles] no dependency manifest changes staged; skipping refresh."
  exit 0
fi

if [[ "${refresh_uv}" == "true" ]]; then
  echo "[lockfiles] refreshing uv.lock using declared version constraints."
  uv lock --upgrade
  git add uv.lock
fi

if [[ "${refresh_npm}" == "true" ]]; then
  echo "[lockfiles] refreshing npm lockfile using package.json ranges."
  if [[ -f "package-lock.json" || -f "npm-shrinkwrap.json" ]]; then
    npm update --package-lock-only --ignore-scripts --no-audit --no-fund
  else
    npm install --package-lock-only --ignore-scripts --no-audit --no-fund
  fi
  git add package-lock.json npm-shrinkwrap.json 2>/dev/null || true
fi
