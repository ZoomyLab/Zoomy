#!/bin/bash
#
# Run all Basilisk benchmark cases via Docker.
#
# Usage:
#   1. Build the image (once):
#        docker build -t basilisk_runner .
#   2. Run all benchmarks:
#        bash run_benchmarks.sh
#
# Output structure:
#   artifacts/basilisk_benchmarks/
#     ├── swe_bump1d/          (VTK series + log)
#     ├── swe_transcritical/
#     ├── swe_dambreak/
#     ├── swe_parabola/        (VTK series + analytical CSV)
#     ├── ml_lock_exchange/    (CSV profiles + VTK)
#     ├── ml_wind_driven/      (CSV profiles + VTK)
#     ├── gn_soliton/          (VTK series + error CSV)
#     └── gn_beach/            (VTK series)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="basilisk_runner"
OUTPUT_ROOT="${SCRIPT_DIR}/../../artifacts/basilisk_benchmarks"

# List of benchmark cases (filename without .c)
CASES=(
  swe_bump1d
  swe_transcritical
  swe_dambreak
  swe_parabola
  ml_lock_exchange
  ml_wind_driven
  gn_soliton
  gn_beach
)

echo "============================================"
echo " Basilisk Benchmark Runner"
echo "============================================"
echo ""

# Check Docker image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "[!] Docker image '$IMAGE_NAME' not found. Building..."
  docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
  echo ""
fi

# Clean and create output root
mkdir -p "$OUTPUT_ROOT"

FAILED=()
PASSED=()

for CASE in "${CASES[@]}"; do
  CASE_DIR="${OUTPUT_ROOT}/${CASE}"
  CASE_FILE="benchmarks/${CASE}.c"

  echo "--------------------------------------------"
  echo " Running: ${CASE}"
  echo "--------------------------------------------"

  # Verify source exists
  if [ ! -f "${SCRIPT_DIR}/${CASE_FILE}" ]; then
    echo "  [SKIP] Source file not found: ${CASE_FILE}"
    FAILED+=("$CASE (missing source)")
    continue
  fi

  # Create clean output directory
  rm -rf "$CASE_DIR"
  mkdir -p "$CASE_DIR"

  # Copy source into output dir for reference
  cp "${SCRIPT_DIR}/${CASE_FILE}" "${CASE_DIR}/"

  # Run in Docker: mount the case output dir as /workspace
  if docker run --rm \
    -v "${CASE_DIR}:/workspace" \
    "$IMAGE_NAME" "${CASE}.c" 2>&1 | tee "${CASE_DIR}/docker.log"; then
    echo "  [OK] ${CASE} completed successfully."
    PASSED+=("$CASE")
  else
    echo "  [FAIL] ${CASE} failed. See ${CASE_DIR}/docker.log"
    FAILED+=("$CASE")
  fi

  echo ""
done

echo "============================================"
echo " Summary"
echo "============================================"
echo " Passed: ${#PASSED[@]}/${#CASES[@]}"
for c in "${PASSED[@]}"; do echo "   [OK]   $c"; done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo " Failed: ${#FAILED[@]}/${#CASES[@]}"
  for c in "${FAILED[@]}"; do echo "   [FAIL] $c"; done
fi
echo ""
echo " Output: ${OUTPUT_ROOT}"
echo "============================================"
