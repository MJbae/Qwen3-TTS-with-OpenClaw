#!/usr/bin/env bash
set -euo pipefail

# Clone a voice-id from a reference audio file.
# Converts input audio to WAV (16k) for compatibility.
#
# Usage:
#   ./scripts/clone-from-audio.sh <voice-id> <ref-audio-path> <ref-text>

VOICE_ID="${1:-}"
REF_AUDIO_IN="${2:-}"
REF_TEXT="${3:-}"

if [[ -z "$VOICE_ID" || -z "$REF_AUDIO_IN" || -z "$REF_TEXT" ]]; then
  echo "Usage: $0 <voice-id> <ref-audio-path> <ref-text>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/runtime/voices/$VOICE_ID"
mkdir -p "$OUT_DIR"

REF_WAV="$OUT_DIR/ref.wav"

# Prefer sox if available
if command -v sox >/dev/null 2>&1; then
  sox "$REF_AUDIO_IN" -r 16000 -c 1 -b 16 "$REF_WAV"
else
  # Fallback: just copy (may fail if non-wav)
  cp -f "$REF_AUDIO_IN" "$REF_WAV"
fi

"$ROOT/scripts/qtts-run.sh" clone create \
  --voice-id "$VOICE_ID" \
  --ref-audio "$REF_WAV" \
  --ref-text "$REF_TEXT"
