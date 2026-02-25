#!/usr/bin/env bash
set -euo pipefail

# Run qtts inside the project conda env (no manual activate required)
# Usage:
#   ./scripts/qtts-run.sh <qtts args...>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="$ROOT/.conda/qtts"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH" >&2
  exit 2
fi

if [[ ! -d "$ENV_PATH" ]]; then
  echo "ERROR: conda env not found: $ENV_PATH" >&2
  echo "Create it first (README): conda create -y -p $ENV_PATH python=3.10" >&2
  exit 3
fi

# Use module execution with PYTHONPATH bound to this workspace so copied envs
# do not accidentally execute a stale editable install from another directory.
exec conda run --no-capture-output -p "$ENV_PATH" env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m qtts.cli "$@"
