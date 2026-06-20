#!/usr/bin/env bash
# Run the bundled TELEMAC-2D Malpasset verification case in parallel.
#
# The SIF is read-only, so the case is copied to a writable workdir first; the
# TELEMAC driver (telemac2d.py) partitions with PARTEL+METIS, runs the solve
# under mpiexec, and recombines with GRETEL.
#
# Usage:
#   ./run_malpasset.sh [-n NCSIZE] [-c CAS] [-s SIF] [-w WORKDIR]
#     -n  number of MPI ranks (domains)          [default 8]
#     -c  steering file (one of t2d_malpasset-*)  [default t2d_malpasset-char.cas]
#     -s  path to the .sif                        [default ./zoomy_telemac.sif]
#     -w  writable workdir                        [default a fresh mktemp dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$HERE/zoomy_telemac.sif"
NCSIZE=8
CAS="t2d_malpasset-char.cas"
WORKDIR=""

while getopts "n:c:s:w:h" opt; do
  case "$opt" in
    n) NCSIZE="$OPTARG" ;;
    c) CAS="$OPTARG" ;;
    s) SIF="$OPTARG" ;;
    w) WORKDIR="$OPTARG" ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "bad option"; exit 2 ;;
  esac
done

[ -f "$SIF" ] || { echo "ERROR: SIF not found: $SIF (build it first)"; exit 1; }
[ -n "$WORKDIR" ] || WORKDIR="$(mktemp -d /tmp/malpasset.XXXXXX)"
mkdir -p "$WORKDIR"

echo ">> SIF      : $SIF"
echo ">> case     : $CAS"
echo ">> ncsize   : $NCSIZE"
echo ">> workdir  : $WORKDIR"

# Stage the example out of the read-only SIF into the writable workdir.
WORKDIR="$WORKDIR" apptainer exec "$SIF" bash -c \
  'cp -r "$HOMETEL/examples/telemac2d/malpasset/." "$WORKDIR"/'

cd "$WORKDIR"
echo ">> running…"
time apptainer exec "$SIF" telemac2d.py "$CAS" --ncsize="$NCSIZE"
echo ">> done. outputs in: $WORKDIR"
