#!/usr/bin/env bash
# Build a "release zip" for course website: looks complete, excludes heavy/binary artifacts.
# Usage: from anywhere:  bash /path/to/cv_project/website_submission/scripts/pack_release.sh [output_zip_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts -> website_submission -> cv_project
CV_PROJECT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ZIP="${1:-$(cd "${CV_PROJECT}/.." && pwd)/cv_project_source_release.zip}"

TMP="$(mktemp -d)"
STAGING="${TMP}/presentation_code"
mkdir -p "${STAGING}"

echo "[info] Staging from ${CV_PROJECT} -> ${STAGING}"

rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.baseline_cache/' \
  --exclude='checkpoints/*.pth' \
  --exclude='checkpoints/*.pt' \
  --exclude='results/.baseline_cache/' \
  --exclude='results/*.png' \
  --exclude='third_party/*/.git/' \
  --exclude='third_party/**/logs/' \
  --exclude='**/build/' \
  --exclude='*.egg-info/' \
  --exclude='*.so' \
  --exclude='*.npy' \
  "${CV_PROJECT}/" "${STAGING}/"

# Ensure placeholders exist if folders were empty after exclusions
mkdir -p "${STAGING}/checkpoints"
touch "${STAGING}/checkpoints/.gitkeep"
echo "# Place downloaded *.pth here or pass paths via CLI / env. Not distributed." > "${STAGING}/checkpoints/README.txt"

(
  cd "${TMP}"
  zip -r -q "${OUT_ZIP}" presentation_code
)

rm -rf "${TMP}"
echo "[done] ${OUT_ZIP}"
ls -la "${OUT_ZIP}"
