#!/bin/bash
# DG(0) optimization sweep — small test case (8 s physical ≈ 3 min wall
# per config in the baseline LU configuration).  Runs all 9 configs
# back-to-back, then prints a one-line summary per config.
set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1

export ZOOMY_DIR=/work
export BENCH_DG_DEGREE=0
export BENCH_LIMITER=none

LOGDIR=/work/outputs/bench_logs_dg0
mkdir -p $LOGDIR

TIME_END="${1:-8.0}"

for CFG in 0 1 2 3 4 5 6 7 8; do
  echo ""
  echo "============================================================"
  echo "  CFG $CFG  DG(0)  time_end=${TIME_END}s"
  echo "============================================================"
  python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG $TIME_END _dg0_${TIME_END}s 2>&1 \
      | tee $LOGDIR/cfg${CFG}_${TIME_END}s.log || true
done

echo ""
echo "============================================================"
echo "  DG(0) SWEEP SUMMARY  (time_end=${TIME_END}s)"
echo "============================================================"
printf "  %-32s  %10s  %8s  %10s  %10s\n" "config" "wall(s)" "iters" "src%" "mass-err"
printf "  %-32s  %10s  %8s  %10s  %10s\n" "------" "-------" "-----" "----" "--------"
for c in 0 1 2 3 4 5 6 7 8; do
  f="$LOGDIR/cfg${c}_${TIME_END}s.log"
  [ -f "$f" ] || continue
  label=$(grep -oE "label='[^']*'" $f | head -1 | sed "s/label=//; s/'//g")
  wall=$(grep -oE "wall=[0-9.]+s  iters=" $f | head -1 | sed "s/wall=//; s/s  iters=//")
  iters=$(grep -oE "iters=[0-9]+" $f | head -1 | sed "s/iters=//")
  srcpct=$(grep -oE "src=[0-9.]+s \([0-9.]+%\)" $f | tail -1 | sed "s/.*(\([0-9.]*\)%).*/\1/")
  mass=$(grep -oE "ΔV/V0=[+\-]?[0-9.e+\-]+" $f | head -1 | sed "s/ΔV\/V0=//")
  printf "  %-32s  %10s  %8s  %10s  %10s\n" "$label" "$wall" "$iters" "${srcpct}%" "$mass"
done

echo ""
echo "DONE DG(0) sweep."
