#!/usr/bin/env bash
set -euo pipefail

# Synthesize speech for an existing voice-id.
# Usage:
#   ./scripts/speak.sh <voice-id> <text> [out-wav]

VOICE_ID="${1:-}"
TEXT="${2:-}"
OUT_WAV="${3:-}"

if [[ -z "$VOICE_ID" || -z "$TEXT" ]]; then
  echo "Usage: $0 <voice-id> <text> [out-wav]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$OUT_WAV" ]]; then
  TS="$(date +"%Y%m%d-%H%M%S")"
  OUT_WAV="$ROOT/runtime/output/${VOICE_ID}_${TS}.wav"
fi

mkdir -p "$(dirname "$OUT_WAV")"

"$ROOT/scripts/qtts-run.sh" speak \
  --voice-id "$VOICE_ID" \
  --text "$TEXT" \
  --out "$OUT_WAV"

echo "$OUT_WAV"
