#!/bin/bash
# MPI sweep for the *iterative* DG(0) configs only — skips LU-based
# variants that don't parallelize.  Use to test scaling at higher N.
set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1
N="${1:-8}"
TIME_END="${2:-8.0}"
export ZOOMY_DIR=/work
export BENCH_DG_DEGREE=0
export BENCH_LIMITER=none
LOGDIR=/work/outputs/bench_logs_dg0_mpi
mkdir -p $LOGDIR

# Only iteratives + scaling-capable direct:
#   9  GMRES + ASM + sub-LU
#  10  GMRES + GAMG
for CFG in 9 10; do
  echo ""
  echo "============================================================"
  echo "  MPI iter:  -n $N  cfg=$CFG  time_end=${TIME_END}s"
  echo "============================================================"
  mpirun -n $N --allow-run-as-root --oversubscribe \
      python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG $TIME_END _mpiN${N}_iter \
      2>&1 | tee $LOGDIR/n${N}_cfg${CFG}.log || true
done

echo ""
echo "============================================================"
echo "  MPI(${N}) ITERATIVE SUMMARY  (time_end=${TIME_END}s)"
echo "============================================================"
printf "  %-4s  %-32s  %10s  %8s  %10s\n" "cfg" "label" "wall(s)" "iters" "ΔV/V0"
printf "  %-4s  %-32s  %10s  %8s  %10s\n" "---" "-----" "-------" "-----" "-----"
for c in 9 10; do
  f="$LOGDIR/n${N}_cfg${c}.log"
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
echo "DONE mpi-${N} iterative sweep."
