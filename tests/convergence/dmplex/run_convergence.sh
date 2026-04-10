#!/bin/bash
# ============================================================================
# DMPlex convergence study for scalar advection and advection-diffusion
#
# Usage (from repo root):
#   apptainer exec --bind ./library/zoomy_core:/zoomy_core \
#       --bind ./library/zoomy_dmplex:/zoomy_dmplex \
#       --writable-tmpfs \
#       zoomy_dmplex_latest.sif bash tests/convergence/dmplex/run_convergence.sh
#
# Or run individual tests:
#   bash run_convergence.sh advection splitting
#   bash run_convergence.sh advection splitting 2
#   bash run_convergence.sh advdiff imex 2
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMPLEX_DIR="${DMPLEX_DIR:-/zoomy_dmplex}"
WORK_DIR="/tmp/convergence_test"

# --- Configuration ---
RESOLUTIONS=(10 20 40 80)
TEST_TYPE="${1:-advection}"       # advection or advdiff
STRATEGY="${2:-splitting}"        # splitting, imex, implicit
RECON_ORDER="${3:-2}"             # 1 or 2

echo "============================================================"
echo " DMPlex Convergence Study"
echo " Test: ${TEST_TYPE}  Strategy: ${STRATEGY}  Order: ${RECON_ORDER}"
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
# Copy all HPP/CPP files from the dmplex solver
for f in main.cpp Makefile MUSCLSolver.hpp ModularSolver.hpp VirtualSolver.hpp \
         TransportStep.hpp SourceStep.hpp Reconstruction.hpp Gradient.hpp \
         SolverStrategies.hpp IOManager.hpp Settings.hpp MshLoader.hpp \
         MeshConfigLoader.hpp; do
    cp "$DMPLEX_DIR/$f" solver/ 2>/dev/null || true
done
cp -r "$DMPLEX_DIR/nlohmann" solver/ 2>/dev/null || true

# Copy the appropriate Model.H and Numerics.H
if [ "$TEST_TYPE" = "advdiff" ]; then
    cp "$SCRIPT_DIR/Model_advdiff.H" solver/Model.H
    cp "$SCRIPT_DIR/Numerics_advection.H" solver/Numerics.H
else
    cp "$SCRIPT_DIR/Model_advection.H" solver/Model.H
    cp "$SCRIPT_DIR/Numerics_advection.H" solver/Numerics.H
fi

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

# Parameters for the test
AX=1.0
AY=0.5
SIGMA=0.15

if [ "$TEST_TYPE" = "advdiff" ]; then
    KAPPA=0.01
    T_END=0.1   # Short time so solution stays in domain
    CFL=0.3     # Smaller CFL for stability with diffusion
else
    KAPPA=0.0
    T_END=0.1   # Short time so Gaussian stays well inside [0,1]^2
    CFL=0.4
fi

# Store results for convergence computation
declare -a H_VALUES
declare -a L1_ERRORS
declare -a LINF_ERRORS

for NX in "${RESOLUTIONS[@]}"; do
    H=$(python3 -c "print(1.0/$NX)")
    MESH_PATH="$WORK_DIR/meshes/square_${NX}.msh"
    OUT_DIR="$WORK_DIR/output_${NX}"

    # Write settings.json
    if [ "$TEST_TYPE" = "advdiff" ]; then
        cat > settings.json <<ENDJSON
{
    "name": "ScalarAdvDiff_${NX}",
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
            "sigma": ${SIGMA},
            "kappa": ${KAPPA}
        }
    }
}
ENDJSON
    else
        cat > settings.json <<ENDJSON
{
    "name": "ScalarAdv_${NX}",
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
            "sigma": ${SIGMA}
        }
    }
}
ENDJSON
    fi

    mkdir -p "$OUT_DIR"

    echo "  Running NX=$NX (h=$H)..."
    ./solver_cpu -settings settings.json 2>&1 | grep -E "Step|INFO|Error" || true

    # Compute error against exact solution using PETSc + Python
    python3 - "$MESH_PATH" "$OUT_DIR" "$T_END" "$AX" "$AY" "$SIGMA" "$KAPPA" "$NX" <<'PYTHON_SCRIPT'
import sys
import numpy as np

mesh_path = sys.argv[1]
out_dir = sys.argv[2]
t_end = float(sys.argv[3])
ax = float(sys.argv[4])
ay = float(sys.argv[5])
sigma = float(sys.argv[6])
kappa = float(sys.argv[7])
nx = int(sys.argv[8])

# Read solution from VTK
import os
vtu_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.vtu')])
if not vtu_files:
    print(f"ERROR: No VTU files found in {out_dir}")
    sys.exit(1)

# Use the last snapshot
last_vtu = os.path.join(out_dir, vtu_files[-1])

# Parse VTU (XML format) to extract cell data
# Use meshio if available, otherwise manual parse
try:
    import meshio
    mesh = meshio.read(last_vtu)
    # Cell data is in mesh.cell_data or mesh.point_data
    # PETSc VTK writes cell data
    if 'State' in mesh.cell_data:
        u_num = np.concatenate([d[:, 0] if d.ndim > 1 else d for d in mesh.cell_data['State']])
    elif mesh.cell_data:
        key = list(mesh.cell_data.keys())[0]
        u_num = np.concatenate([d[:, 0] if d.ndim > 1 else d for d in mesh.cell_data[key]])
    else:
        # Try point_data
        key = list(mesh.point_data.keys())[0]
        u_num = mesh.point_data[key][:, 0] if mesh.point_data[key].ndim > 1 else mesh.point_data[key]
except Exception as e:
    print(f"  Warning: Could not read VTU with meshio: {e}")
    print(f"  Attempting PETSc direct read...")
    # Fallback: read solution via PETSc
    try:
        import petsc4py
        petsc4py.init([sys.argv[0]])
        from petsc4py import PETSc

        dm = PETSc.DMPlex().createFromFile(mesh_path)
        dm.distribute()

        cStart, cEnd = dm.getHeightStratum(0)
        n_cells = cEnd - cStart

        # Compute centroids and exact solution
        centroids = []
        for c in range(cStart, cEnd):
            geo = dm.computeCellGeometryFVM(c)
            centroids.append(geo[1][:2])
        centroids = np.array(centroids)

        # Exact solution at t_end
        sigma_t = np.sqrt(sigma**2 + 2 * kappa * t_end) if kappa > 0 else sigma
        amp = (sigma**2 / sigma_t**2) if kappa > 0 else 1.0
        x_c = 0.25 + ax * t_end
        y_c = 0.25 + ay * t_end
        dx = centroids[:, 0] - x_c
        dy = centroids[:, 1] - y_c
        u_exact = amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma_t**2))

        # Cannot get numerical solution without reading VTU, so use PETSc viewer
        print(f"  ERROR: Need meshio to read VTU. Install: pip install meshio")
        print(f"  L1_ERROR={nx} UNKNOWN")
        sys.exit(0)
    except:
        print(f"  ERROR: Cannot compute error")
        sys.exit(0)

# Compute centroids from mesh
try:
    mesh_data = meshio.read(mesh_path)
    tris = mesh_data.cells_dict.get('triangle')
    if tris is not None:
        pts = mesh_data.points[:, :2]
        centroids = pts[tris].mean(axis=1)
    else:
        # For PETSc VTU output, use the cell centers from the VTU itself
        if hasattr(mesh, 'points'):
            # VTU from PETSc: points are cell centers for FV data
            centroids = mesh.points[:, :2]
        else:
            print(f"  ERROR: Cannot determine centroids")
            sys.exit(0)
except:
    # Use points from VTU as centroids
    centroids = mesh.points[:, :2]

n_cells = len(u_num)

# Exact solution at t_end
sigma_t = np.sqrt(sigma**2 + 2 * kappa * t_end) if kappa > 0 else sigma
amp = (sigma**2 / sigma_t**2) if kappa > 0 else 1.0
x_c = 0.25 + ax * t_end
y_c = 0.25 + ay * t_end

if len(centroids) != n_cells:
    # If mismatch, use mesh centroids from .msh
    mesh_data = meshio.read(mesh_path)
    tris = mesh_data.cells_dict.get('triangle')
    pts = mesh_data.points[:, :2]
    centroids = pts[tris].mean(axis=1)

dx_arr = centroids[:, 0] - x_c
dy_arr = centroids[:, 1] - y_c
u_exact = amp * np.exp(-(dx_arr**2 + dy_arr**2) / (2.0 * sigma_t**2))

# Compute errors
h = 1.0 / nx
cell_area = h * h / 2.0  # Each triangle has area h^2/2 for structured mesh

error = np.abs(u_num[:len(u_exact)] - u_exact[:len(u_num)])
L1 = np.sum(error * cell_area)
Linf = np.max(error)

print(f"  h={h:.4f}  L1={L1:.6e}  Linf={Linf:.6e}  n_cells={n_cells}")
# Write to file for post-processing
with open(os.path.join(out_dir, "error.txt"), "w") as f:
    f.write(f"{h} {L1} {Linf}\n")
PYTHON_SCRIPT

done

# --- Compute convergence rates ---
echo ""
echo "============================================================"
echo " Convergence Rate Summary"
echo " Test: ${TEST_TYPE}  Strategy: ${STRATEGY}  Order: ${RECON_ORDER}"
echo "============================================================"
echo ""

python3 - "$WORK_DIR" "${RESOLUTIONS[*]}" <<'CONV_SCRIPT'
import sys, os, numpy as np

work_dir = sys.argv[1]
resolutions = [int(x) for x in sys.argv[2].split()]

hs, L1s, Linfs = [], [], []
for nx in resolutions:
    err_file = os.path.join(work_dir, f"output_{nx}", "error.txt")
    if os.path.exists(err_file):
        with open(err_file) as f:
            parts = f.read().strip().split()
            hs.append(float(parts[0]))
            L1s.append(float(parts[1]))
            Linfs.append(float(parts[2]))

if len(hs) < 2:
    print("ERROR: Need at least 2 data points for convergence rate")
    sys.exit(1)

print(f"{'h':>10s}  {'L1 error':>12s}  {'L1 rate':>8s}  {'Linf error':>12s}  {'Linf rate':>10s}")
print("-" * 60)
for i in range(len(hs)):
    if i == 0:
        print(f"{hs[i]:10.4f}  {L1s[i]:12.6e}  {'---':>8s}  {Linfs[i]:12.6e}  {'---':>10s}")
    else:
        rate_L1 = np.log(L1s[i-1] / L1s[i]) / np.log(hs[i-1] / hs[i])
        rate_Linf = np.log(Linfs[i-1] / Linfs[i]) / np.log(hs[i-1] / hs[i])
        print(f"{hs[i]:10.4f}  {L1s[i]:12.6e}  {rate_L1:8.2f}  {Linfs[i]:12.6e}  {rate_Linf:10.2f}")

# Final summary
if len(hs) >= 3:
    avg_L1 = np.mean([np.log(L1s[i-1]/L1s[i])/np.log(hs[i-1]/hs[i]) for i in range(1, len(hs))])
    avg_Linf = np.mean([np.log(Linfs[i-1]/Linfs[i])/np.log(hs[i-1]/hs[i]) for i in range(1, len(hs))])
    print(f"\nAverage L1 rate:   {avg_L1:.2f}")
    print(f"Average Linf rate: {avg_Linf:.2f}")
CONV_SCRIPT

echo ""
echo "Done."
