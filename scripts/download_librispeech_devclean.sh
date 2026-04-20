#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/librispeech}"
mkdir -p "$DEST"
URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
TAR="$DEST/dev-clean.tar.gz"
if [ ! -f "$TAR" ]; then
  echo "Downloading dev-clean (~337MB)..."
  curl -L --fail -o "$TAR" "$URL"
fi
tar -xzf "$TAR" -C "$DEST"
echo "Extracted to $DEST/LibriSpeech/dev-clean"
