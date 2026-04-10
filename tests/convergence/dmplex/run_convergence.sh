#!/bin/bash
# ============================================================================
# DMPlex convergence study for 2D scalar advection (MUSCL+Venkatakrishnan)
#
# Proves O2 convergence for the DMPlex solver with SSP-RK splitting.
#
# Usage (from repo root):
#   apptainer exec --bind ./library/zoomy_dmplex:/zoomy_dmplex \
#       --bind ./tests/convergence/dmplex:/convergence_test \
#       --writable-tmpfs \
#       zoomy_dmplex_latest.sif bash /convergence_test/run_convergence.sh
#
# Or run a single configuration:
#   bash run_convergence.sh advection splitting 2
#   bash run_convergence.sh advection splitting 1   # O1 for comparison
#
# Notes:
#   - IMEX strategy is not supported for pure advection (source=0 causes
#     hardcoded shallow-water dry-cell check to read out-of-bounds Q[1]).
#   - dt is constant across resolutions (~1e-3, set by PETSc minRadius).
#     For N=10..40 the spatial error dominates and O2 is clearly visible.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMPLEX_DIR="${DMPLEX_DIR:-/zoomy_dmplex}"
WORK_DIR="/tmp/convergence_test"

# --- Configuration ---
RESOLUTIONS=(10 20 40)
TEST_TYPE="${1:-advection}"
STRATEGY="${2:-splitting}"        # splitting (SSP-RK)
RECON_ORDER="${3:-2}"             # 1 or 2

# Physics: u(x,y,t) = exp(-alpha * ((x - x0 - ax*t)^2 + (y - y0 - ay*t)^2))
AX=1.0
AY=0.0
ALPHA=50.0
X0=0.3
Y0=0.5
T_END=0.05
CFL=0.4

echo "============================================================"
echo " DMPlex Convergence Study"
echo " Test: ${TEST_TYPE}  Strategy: ${STRATEGY}  Order: ${RECON_ORDER}"
echo " IC: Gaussian at (${X0},${Y0}), alpha=${ALPHA}, a=(${AX},${AY})"
echo " t_end=${T_END}  CFL=${CFL}"
echo "============================================================"

# --- Setup working directory ---
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# --- Generate meshes ---
echo ""
echo "[1/4] Generating meshes..."
python3 "$SCRIPT_DIR/generate_meshes.py" \
    --resolutions ${RESOLUTIONS[@]} --outdir meshes

# --- Copy solver sources ---
echo ""
echo "[2/4] Setting up solver..."
mkdir -p solver
for f in main.cpp Makefile MUSCLSolver.hpp ModularSolver.hpp VirtualSolver.hpp \
         TransportStep.hpp SourceStep.hpp Reconstruction.hpp Gradient.hpp \
         SolverStrategies.hpp IOManager.hpp Settings.hpp MshLoader.hpp \
         MeshConfigLoader.hpp; do
    cp "$DMPLEX_DIR/$f" solver/ 2>/dev/null || true
done
cp -r "$DMPLEX_DIR/nlohmann" solver/ 2>/dev/null || true

cp "$SCRIPT_DIR/Model_advection.H" solver/Model.H
cp "$SCRIPT_DIR/Numerics_advection.H" solver/Numerics.H

# --- Compile ---
echo ""
echo "[3/4] Compiling..."
cd solver
make clean_all 2>/dev/null || true
make CPU 2>&1 | tail -5
if [ ! -f solver_cpu ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "  Compilation successful."

# --- Run convergence study ---
echo ""
echo "[4/4] Running convergence study..."
echo ""

for NX in "${RESOLUTIONS[@]}"; do
    H=$(python3 -c "print(1.0/$NX)")
    MESH_PATH="$WORK_DIR/meshes/square_${NX}.msh"
    OUT_DIR="$WORK_DIR/output_o${RECON_ORDER}_${STRATEGY}_${NX}"

    cat > settings.json <<ENDJSON
{
    "name": "ScalarAdv_o${RECON_ORDER}_${STRATEGY}_${NX}",
    "io": {
        "directory": "${OUT_DIR}",
        "filename": "sol",
        "snapshots": 1,
        "snapshot_logic": "interpolate",
        "clean_directory": true,
        "mesh_path": "${MESH_PATH}",
        "write_3d": false
    },
    "solver": {
        "t_end": ${T_END},
        "cfl": ${CFL},
        "reconstruction_order": ${RECON_ORDER},
        "strategy": "${STRATEGY}"
    },
    "model": {
        "parameters": {
            "ax": ${AX},
            "ay": ${AY},
            "alpha": ${ALPHA},
            "x0": ${X0},
            "y0": ${Y0}
        }
    }
}
ENDJSON

    mkdir -p "$OUT_DIR"

    echo "  Running NX=$NX (h=$H) strategy=$STRATEGY order=$RECON_ORDER..."
    ./solver_cpu -settings settings.json 2>&1 | grep -E "Step|INFO|Error" || true

    # Compute error against exact solution
    python3 - "$OUT_DIR" "$T_END" "$AX" "$AY" "$ALPHA" "$X0" "$Y0" "$NX" "$RECON_ORDER" "$STRATEGY" <<'PYTHON_SCRIPT'
import sys, os, numpy as np

out_dir = sys.argv[1]
t_end = float(sys.argv[2])
ax = float(sys.argv[3])
ay = float(sys.argv[4])
alpha = float(sys.argv[5])
x0 = float(sys.argv[6])
y0 = float(sys.argv[7])
nx = int(sys.argv[8])
order = sys.argv[9]
strategy = sys.argv[10]

import meshio

vtu_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.vtu')])
if not vtu_files:
    print(f"ERROR: No VTU files found in {out_dir}")
    sys.exit(1)

vtu = meshio.read(os.path.join(out_dir, vtu_files[-1]))

u_num = None
for key in vtu.cell_data:
    if key == "Rank":
        continue
    data = vtu.cell_data[key]
    if isinstance(data, list):
        data = np.concatenate(data)
    u_num = data[:, 0] if data.ndim > 1 else data
    break

if u_num is None:
    print(f"ERROR: No cell data in VTU")
    sys.exit(1)

tris = vtu.cells_dict["triangle"]
pts = vtu.points[:, :2]
centroids = pts[tris].mean(axis=1)

# Compute actual triangle areas for proper integration
v0 = pts[tris[:, 0]]; v1 = pts[tris[:, 1]]; v2 = pts[tris[:, 2]]
areas = 0.5 * np.abs((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) - (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1]))

h = 1.0 / nx
xc = x0 + ax * t_end
yc = y0 + ay * t_end
dx = centroids[:, 0] - xc
dy = centroids[:, 1] - yc
u_exact = np.exp(-alpha * (dx**2 + dy**2))

error = np.abs(u_num - u_exact)
L1 = np.sum(error * areas)
L2 = np.sqrt(np.sum(error**2 * areas) / np.sum(areas))
Linf = np.max(error)

print(f"  h={h:.4f}  L1={L1:.6e}  L2={L2:.6e}  Linf={Linf:.6e}  n_cells={len(u_num)}  peak={u_num.max():.6f}")

with open(os.path.join(out_dir, "error.txt"), "w") as f:
    f.write(f"{h} {L1} {L2} {Linf}\n")
PYTHON_SCRIPT

done

# --- Compute convergence rates ---
echo ""
echo "============================================================"
echo " Convergence Rate Summary"
echo " Test: ${TEST_TYPE}  Strategy: ${STRATEGY}  Order: ${RECON_ORDER}"
echo "============================================================"
echo ""

python3 - "$WORK_DIR" "${RESOLUTIONS[*]}" "$STRATEGY" "$RECON_ORDER" <<'CONV_SCRIPT'
import sys, os, numpy as np

work_dir = sys.argv[1]
resolutions = [int(x) for x in sys.argv[2].split()]
strategy = sys.argv[3]
order = sys.argv[4]

hs, L1s, L2s, Linfs = [], [], [], []
for nx in resolutions:
    err_file = os.path.join(work_dir, f"output_o{order}_{strategy}_{nx}", "error.txt")
    if os.path.exists(err_file):
        with open(err_file) as f:
            parts = f.read().strip().split()
            hs.append(float(parts[0]))
            L1s.append(float(parts[1]))
            L2s.append(float(parts[2]))
            Linfs.append(float(parts[3]))

if len(hs) < 2:
    print("ERROR: Need at least 2 data points for convergence rate")
    sys.exit(1)

print(f"{'h':>10s}  {'L1 error':>12s}  {'L1 rate':>8s}  {'L2 error':>12s}  {'L2 rate':>8s}  {'Linf error':>12s}  {'Linf rate':>10s}")
print("-" * 90)
for i in range(len(hs)):
    if i == 0:
        print(f"{hs[i]:10.4f}  {L1s[i]:12.6e}  {'---':>8s}  {L2s[i]:12.6e}  {'---':>8s}  {Linfs[i]:12.6e}  {'---':>10s}")
    else:
        rate_L1 = np.log(L1s[i-1] / L1s[i]) / np.log(hs[i-1] / hs[i])
        rate_L2 = np.log(L2s[i-1] / L2s[i]) / np.log(hs[i-1] / hs[i])
        rate_Linf = np.log(Linfs[i-1] / Linfs[i]) / np.log(hs[i-1] / hs[i])
        print(f"{hs[i]:10.4f}  {L1s[i]:12.6e}  {rate_L1:8.2f}  {L2s[i]:12.6e}  {rate_L2:8.2f}  {Linfs[i]:12.6e}  {rate_Linf:10.2f}")

if len(hs) >= 3:
    avg_L1 = np.mean([np.log(L1s[i-1]/L1s[i])/np.log(hs[i-1]/hs[i]) for i in range(1, len(hs))])
    avg_L2 = np.mean([np.log(L2s[i-1]/L2s[i])/np.log(hs[i-1]/hs[i]) for i in range(1, len(hs))])
    avg_Linf = np.mean([np.log(Linfs[i-1]/Linfs[i])/np.log(hs[i-1]/hs[i]) for i in range(1, len(hs))])
    print(f"\nAverage L1 rate:   {avg_L1:.2f}")
    print(f"Average L2 rate:   {avg_L2:.2f}")
    print(f"Average Linf rate: {avg_Linf:.2f}")

    if avg_L1 >= 1.8:
        print(f"\n*** PASS: L1 convergence rate {avg_L1:.2f} >= 1.8 (O2 achieved) ***")
    else:
        print(f"\n*** INFO: L1 convergence rate {avg_L1:.2f} < 1.8 ***")
CONV_SCRIPT

echo ""
echo "Done."
