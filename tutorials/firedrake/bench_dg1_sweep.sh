#!/bin/bash
# Lean DG(1) sweep: configs 1–5 at t=5s for apples-to-apples
# comparison against the 0–5s baseline window already captured in
# cfg0_20s.log.
set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1
export ZOOMY_DIR=/work
LOGDIR=/work/outputs/bench_logs
mkdir -p $LOGDIR

for CFG in 1 2 3 4 5; do
  echo ""
  echo "============================================================"
  echo "  CFG $CFG  DG(1)  time_end=5s"
  echo "============================================================"
  python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG 5.0 _5s 2>&1 | tee $LOGDIR/cfg${CFG}_5s.log || true
done

echo ""
echo "============================================================"
echo "  SWEEP SUMMARY (configs 1–5 at 5s)"
echo "============================================================"
for c in 1 2 3 4 5; do
  f="$LOGDIR/cfg${c}_5s.log"
  if [ -f "$f" ]; then
    echo "--- cfg $c ---"
    grep -E "DONE|mass:|FAIL|per-stage|^\[bench\]   " $f | head -12
  fi
done
echo ""
echo "DONE sweep."
