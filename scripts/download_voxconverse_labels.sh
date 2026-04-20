#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/voxconverse-labels}"
if [ -d "$DEST" ]; then
  echo "VoxConverse labels already at $DEST"
  exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 https://github.com/joonson/voxconverse "$DEST"
echo "Done: $DEST"
echo "Next: download audio (test set) from https://www.robots.ox.ac.uk/~vgg/data/voxconverse/"
echo "  and extract to data/voxconverse/test/"
