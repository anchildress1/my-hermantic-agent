#!/usr/bin/env bash
set -euo pipefail

echo "[security] running static scan (bandit)."
if [[ -d "src" ]]; then
  uv run bandit -q -r src scripts -x tests
else
  echo "[security] no src directory found; skipping bandit."
fi

echo "[security] running dependency vulnerability audit (pip-audit)."
requirements_file="$(mktemp)"
cleanup() {
  rm -f "${requirements_file:-}"
}
trap cleanup EXIT

uv export --frozen --all-extras --format requirements-txt -o "${requirements_file}" --quiet
uv run pip-audit -r "${requirements_file}" --progress-spinner off

echo "[security] scanning repository for secrets (detect-secrets)."
baseline_path=".secrets.baseline"
exclude_files='(^|/)(\.git|\.venv|\.pytest_cache|\.ruff_cache|coverage|dist|build)/|(^|/)(uv\.lock|package-lock\.json|npm-shrinkwrap\.json)$'

if [[ ! -f "${baseline_path}" ]]; then
  uv run detect-secrets scan --all-files --exclude-files "${exclude_files}" > "${baseline_path}"
  echo "[security] created ${baseline_path}; review and commit it, then rerun checks."
  exit 1
fi

tracked_files=()
while IFS= read -r tracked_file; do
  tracked_files+=("${tracked_file}")
done < <(git ls-files)
if [[ "${#tracked_files[@]}" -eq 0 ]]; then
  echo "[security] no tracked files found for secret scan; skipping."
  exit 0
fi

uv run detect-secrets-hook \
  --baseline "${baseline_path}" \
  --exclude-files "${exclude_files}" \
  "${tracked_files[@]}"

echo "[security] no potential secrets detected."
