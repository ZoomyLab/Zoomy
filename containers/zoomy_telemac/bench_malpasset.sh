#!/usr/bin/env bash
# Strong-scaling benchmark of TELEMAC-2D on the Malpasset case.
#
# Runs the same case at a sweep of MPI rank counts and records the wall-clock
# of the solve, so the TELEMAC timing can be compared against the Zoomy
# backends on the same mesh. Each run is a fresh writable workdir.
#
# Usage:
#   ./bench_malpasset.sh [-c CAS] [-s SIF] [-r "1 2 4 8 16 32"]
#     -c  steering file            [default t2d_malpasset-char.cas]
#     -s  path to the .sif         [default ./zoomy_telemac.sif]
#     -r  space-separated ncsizes  [default "1 2 4 8 16 32"]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$HERE/zoomy_telemac.sif"
CAS="t2d_malpasset-char.cas"
RANKS="1 2 4 8 16 32"

while getopts "c:s:r:h" opt; do
  case "$opt" in
    c) CAS="$OPTARG" ;;
    s) SIF="$OPTARG" ;;
    r) RANKS="$OPTARG" ;;
    h) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "bad option"; exit 2 ;;
  esac
done
[ -f "$SIF" ] || { echo "ERROR: SIF not found: $SIF"; exit 1; }

RESULTS="$HERE/bench_malpasset_results.csv"
echo "ncsize,wall_seconds" > "$RESULTS"
echo "== Malpasset strong-scaling: $CAS =="
printf "%-8s %-12s\n" "ncsize" "wall[s]"

for N in $RANKS; do
  WD="$(mktemp -d /tmp/malpasset_bench.XXXXXX)"
  WORKDIR="$WD" apptainer exec "$SIF" bash -c \
    'cp -r "$HOMETEL/examples/telemac2d/malpasset/." "$WORKDIR"/'
  START=$(date +%s.%N)
  ( cd "$WD" && apptainer exec "$SIF" telemac2d.py "$CAS" --ncsize="$N" \
      > run.log 2>&1 ) || { echo "  ncsize=$N FAILED (see $WD/run.log)"; continue; }
  END=$(date +%s.%N)
  SECS=$(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.1f", b-a}')
  printf "%-8s %-12s\n" "$N" "$SECS"
  echo "$N,$SECS" >> "$RESULTS"
  rm -rf "$WD"
done
echo "== results written to $RESULTS =="
