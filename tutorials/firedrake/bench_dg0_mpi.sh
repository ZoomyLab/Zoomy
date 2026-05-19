#!/bin/bash
# MPI sweep for DG(0) Malpasset.  Tries the four configs that
# *could* scale under MPI:
#   A — baseline-LU (serial-style LU; PETSc errors if no MPI matrix solver)
#   B — MUMPS direct (parallel sparse direct factorisation)
#   C — GAMG (algebraic multigrid iterative)
#   D — BJacobi + sub-LU iterative
#
# Usage:  bench_dg0_mpi.sh <N_procs> [time_end]
set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1

N="${1:-2}"
TIME_END="${2:-8.0}"
export ZOOMY_DIR=/work
export BENCH_DG_DEGREE=0
export BENCH_LIMITER=none
LOGDIR=/work/outputs/bench_logs_dg0_mpi
mkdir -p $LOGDIR

# Smoke test: just run the baseline at -n $N first.  If this dies with
# a halo SF error the patch failed and the rest of the sweep is moot.
echo "============================================================"
echo "  SMOKE: mpirun -n $N  (cfg 0 / baseline-LU)  time_end=${TIME_END}s"
echo "============================================================"
mpirun -n $N --allow-run-as-root --oversubscribe \
    python3 -u tutorials/firedrake/bench_malpasset_dg1.py 0 $TIME_END _mpiN${N}_smoke \
    2>&1 | tee $LOGDIR/n${N}_cfg0_smoke.log || true

# If smoke failed, bail.
if ! grep -q "DONE" $LOGDIR/n${N}_cfg0_smoke.log; then
    echo ""
    echo "[bench] smoke at -n $N failed — see $LOGDIR/n${N}_cfg0_smoke.log"
    exit 1
fi

# Configs to test under MPI.  Indices into bench_malpasset_dg1.CONFIGS.
#   4  MUMPS + basic LS + relaxed KSP (serial winner)
#   6  src-BJacobi+LU (was the only one that gained under MPI N=2)
#   8  combined-best-DG0 (serial best)
#   9  GMRES + ASM + sub-LU (MPI-targeted iterative)
#  10  GMRES + GAMG (MPI-targeted multigrid)
#  11  MPI-MUMPS-direct (parallel direct)
for CFG in 4 6 8 9 10 11; do
  echo ""
  echo "============================================================"
  echo "  MPI sweep:  -n $N  cfg=$CFG  time_end=${TIME_END}s"
  echo "============================================================"
  mpirun -n $N --allow-run-as-root --oversubscribe \
      python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG $TIME_END _mpiN${N} \
      2>&1 | tee $LOGDIR/n${N}_cfg${CFG}.log || true
done

# Summary
echo ""
echo "============================================================"
echo "  MPI(${N}) SUMMARY  (time_end=${TIME_END}s)"
echo "============================================================"
printf "  %-4s  %-32s  %10s  %8s  %10s\n" "cfg" "label" "wall(s)" "iters" "ΔV/V0"
printf "  %-4s  %-32s  %10s  %8s  %10s\n" "---" "-----" "-------" "-----" "-----"
for c in 0 4 6 8 9 10 11; do
  if [ "$c" = "0" ]; then
    f="$LOGDIR/n${N}_cfg0_smoke.log"
  else
    f="$LOGDIR/n${N}_cfg${c}.log"
  fi
  [ -f "$f" ] || continue
  label=$(grep -oE "label='[^']*'" "$f" | head -1 | sed "s/label=//; s/'//g")
  wall=$(grep -oE "wall=[0-9.]+s  iters=" "$f" | head -1 | sed "s/wall=//; s/s  iters=//")
  iters=$(grep -oE "iters=[0-9]+" "$f" | head -1 | sed "s/iters=//")
  mass=$(grep -oE "ΔV/V0=[+\-]?[0-9.e+\-]+" "$f" | head -1 | sed "s/ΔV\/V0=//")
  if [ -z "$wall" ]; then
    printf "  %-4s  %-32s  %10s  %8s  %10s\n" "$c" "${label:-?}" "FAIL" "-" "-"
  else
    printf "  %-4s  %-32s  %10s  %8s  %10s\n" "$c" "$label" "${wall}" "$iters" "$mass"
  fi
done
echo ""
echo "DONE mpi-${N} sweep."
