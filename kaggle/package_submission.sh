#!/usr/bin/env bash
# Package SAM for Kaggle submission (copy sam/ + checkpoint + env subset)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/kaggle/submission_bundle"
rm -rf "$OUT"
mkdir -p "$OUT/sam" "$OUT/environment_files"

cp -R "$ROOT/sam/"* "$OUT/sam/"
cp "$ROOT/scripts/run_sam.py" "$OUT/"
cp "$ROOT/kaggle/run_offline.py" "$OUT/"
cp -R "$ROOT/environment_files/ls20" "$OUT/environment_files/" 2>/dev/null || true

if [ -f "$ROOT/sam/checkpoints/global_sam.pt" ]; then
  mkdir -p "$OUT/sam/checkpoints"
  cp "$ROOT/sam/checkpoints/global_sam.pt" "$OUT/sam/checkpoints/"
fi

echo "Bundle ready at $OUT"
find "$OUT" -type f | head -30
