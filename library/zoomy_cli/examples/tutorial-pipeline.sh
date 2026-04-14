#!/usr/bin/env bash
# tutorial-pipeline.sh — Full Zoomy CLI workflow: init → configure → run → results
#
# This script mirrors the GUI workflow from the command line:
#   1. Initialize a project
#   2. Select model, mesh, and solver cards
#   3. Inspect the case configuration
#   4. Run the simulation locally
#
# Usage:
#   bash library/zoomy_cli/examples/tutorial-pipeline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZOOMY="node ${SCRIPT_DIR}/../cli.js"

echo "============================================"
echo " Zoomy CLI Tutorial Pipeline"
echo "============================================"
echo ""

echo "=== Step 1: Initialize project ==="
$ZOOMY start
echo ""

echo "=== Step 2: Available cards ==="
$ZOOMY overview
echo ""

echo "=== Step 3: Select model + mesh + solver ==="
$ZOOMY select model swe-sme-l0
$ZOOMY select mesh mesh-create-1d
$ZOOMY select solver solver-numpy
echo ""

echo "=== Step 4: Current selections ==="
$ZOOMY status
echo ""

echo "=== Step 5: Inspect case JSON ==="
$ZOOMY case
echo ""

echo "=== Step 6: Run simulation locally ==="
echo "(No backend server needed — runs via Python subprocess)"
echo ""
$ZOOMY run --local
echo ""

echo "============================================"
echo " Pipeline complete!"
echo ""
echo " To run with a remote backend instead:"
echo "   $ZOOMY connect http://localhost:8080"
echo "   $ZOOMY run --wait"
echo ""
echo " To save the project:"
echo "   $ZOOMY save my-project.zip"
echo "============================================"
