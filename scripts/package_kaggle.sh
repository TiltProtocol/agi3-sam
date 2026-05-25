#!/usr/bin/env bash
# Package SAM for offline Kaggle submission (no external API).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/notebooks/sam_kaggle"
WHEELS="$ROOT/data/kaggle/arc_agi_3_wheels"

mkdir -p "$OUT" "$ROOT/sam/checkpoints"

cp -r "$ROOT/sam" "$OUT/sam"
cp "$ROOT/scripts/run_sam.py" "$OUT/run_sam.py"

if [ -d "$WHEELS" ]; then
  mkdir -p "$OUT/wheels"
  cp "$WHEELS"/*.whl "$OUT/wheels/" 2>/dev/null || true
fi

cat > "$OUT/README.md" <<'EOF'
# SAM Kaggle Offline Package

1. Install wheels from `wheels/` (arc-agi, arcengine) if present.
2. Copy `environment_files/` into the notebook working directory.
3. Run `run_sam.py` with `OPERATION_MODE=offline` (default in Arcade OFFLINE).

No external API calls at inference time.
EOF

echo "Packaged to $OUT"
