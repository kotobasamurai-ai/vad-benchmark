#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/esc50}"
if [ -d "$DEST" ]; then
  echo "ESC-50 already at $DEST"
  exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git "$DEST"
echo "Done: $DEST"
