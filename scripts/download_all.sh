#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "Creating Python 3.12 virtualenv ..."
  python3.12 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
fi

echo "=== Downloading public ARC-AGI-3 demo environments ==="
.venv/bin/python scripts/download_public_environments.py

echo
echo "=== Downloading Kaggle competition files ==="
bash scripts/download_kaggle.sh
