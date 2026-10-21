#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

PY="conda run -n zoomy python"

$PY library/zoomy_jax/zoomy_jax/gnn_blueprint/dataset_deltaq.py   --mesh-name channel_quad_2d --n-fields 3 --n-steps 50 --param-values 0.8 1.0 1.2 1.5

$PY library/zoomy_jax/zoomy_jax/gnn_blueprint/train_deltaq.py   --dataset outputs/gnn_blueprint/dataset_deltaq.npz   --out-dir outputs/gnn_blueprint/model_deltaq --epochs 120 --lr 0.02

$PY library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_gmres_guess.py   --model-dir outputs/gnn_blueprint/model_deltaq --n 200

echo "Full loop complete."
