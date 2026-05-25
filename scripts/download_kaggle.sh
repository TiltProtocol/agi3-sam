#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPETITION="arc-prize-2026-arc-agi-3"
OUTPUT_DIR="${ROOT_DIR}/data/kaggle"
VENV="${ROOT_DIR}/.venv/bin/kaggle"

mkdir -p "${OUTPUT_DIR}"

has_kaggle_credentials() {
  local config_dir="$1"
  [[ -f "${config_dir}/kaggle.json" || -f "${config_dir}/credentials.json" ]]
}

resolve_kaggle_config_dir() {
  if [[ -n "${KAGGLE_CONFIG_DIR:-}" ]]; then
    echo "${KAGGLE_CONFIG_DIR}"
    return
  fi

  if has_kaggle_credentials "${ROOT_DIR}/.kaggle"; then
    echo "${ROOT_DIR}/.kaggle"
    return
  fi

  if has_kaggle_credentials "${HOME}/.kaggle"; then
    echo "${HOME}/.kaggle"
    return
  fi

  echo ""
}

KAGGLE_CONFIG_DIR="$(resolve_kaggle_config_dir)"
if [[ -z "${KAGGLE_CONFIG_DIR}" ]]; then
  cat <<'EOF'
Kaggle credentials are required.

Option 1 (recommended):
  cd /Users/adilbek/Desktop/agi3
  .venv/bin/kaggle auth login

Option 2:
  1. Open https://www.kaggle.com/settings/account
  2. Create an API token (downloads kaggle.json)
  3. Save it to ~/.kaggle/kaggle.json
  4. chmod 600 ~/.kaggle/kaggle.json

Then accept the competition rules at:
  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules
EOF
  exit 1
fi

export KAGGLE_CONFIG_DIR
echo "Using Kaggle credentials from ${KAGGLE_CONFIG_DIR}"

echo "Downloading Kaggle competition data to ${OUTPUT_DIR} ..."
"${VENV}" competitions download -c "${COMPETITION}" -p "${OUTPUT_DIR}"

echo
echo "Extracting archives ..."
find "${OUTPUT_DIR}" -maxdepth 1 -name '*.zip' -print0 | while IFS= read -r -d '' zipfile; do
  unzip -o "${zipfile}" -d "${OUTPUT_DIR}"
done

echo "Done. Files are in ${OUTPUT_DIR}"
