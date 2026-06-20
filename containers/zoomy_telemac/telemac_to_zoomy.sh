#!/usr/bin/env bash
# Convert a TELEMAC-2D SELAFIN result into a Zoomy HDF5 store (the layout the
# thesis/cases/malpasset_jax postprocessing reads with zoomy_plotting.read_hdf5).
#
# Phase 1 (dump) runs inside the apptainer (TelemacFile, no h5py); phase 2
# (build) runs on the host / zoomy env (h5py) and reuses a reference Zoomy
# store's mesh so the cell ordering matches the jax runs exactly.
#
# Usage:
#   ./telemac_to_zoomy.sh <result.slf> <out_dir> [ref_store.h5]
#     <result.slf>   a TELEMAC r2d_*.slf (e.g. from ./run_malpasset.sh)
#     <out_dir>      where to write telemac.h5 / settings.h5 / *.ckpt.json
#     [ref_store.h5] a Zoomy store to borrow the mesh from
#                    (default: malpasset_jax order-1 / telemac stand-in store)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$HERE/zoomy_telemac.sif"
SLF="${1:?need <result.slf>}"
OUT="${2:?need <out_dir>}"
REF="${3:-$HERE/../../thesis/cases/malpasset_jax/output/telemac/telemac.h5}"

[ -f "$SIF" ] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[ -f "$SLF" ] || { echo "ERROR: SELAFIN not found: $SLF"; exit 1; }
[ -f "$REF" ] || { echo "ERROR: reference Zoomy store (for mesh) not found: $REF"; exit 1; }

NPZ="$(mktemp /tmp/telemac_nodes.XXXXXX.npz)"
echo ">> phase 1/2: dump SELAFIN -> npz (in container)"
apptainer exec "$SIF" python3 "$HERE/selafin_to_zoomy.py" dump "$SLF" "$NPZ"
echo ">> phase 2/2: build Zoomy store (host h5py)"
python3 "$HERE/selafin_to_zoomy.py" build "$NPZ" "$REF" "$OUT"
rm -f "$NPZ"
echo ">> done: $OUT/telemac.h5"
