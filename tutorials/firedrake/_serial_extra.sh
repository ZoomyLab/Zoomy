#!/bin/bash
set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1
export ZOOMY_DIR=/work BENCH_DG_DEGREE=0 BENCH_LIMITER=none
LOGDIR=/work/outputs/bench_logs_dg0
for CFG in 9 10 11; do
  python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG 8.0 _serial 2>&1 \
      | tee $LOGDIR/cfg${CFG}_8.0s.log
done
echo "---"
for c in 9 10 11; do
  f="$LOGDIR/cfg${c}_8.0s.log"
  wall=$(grep -oE "wall=[0-9.]+s  iters" "$f" | head -1 | sed "s/wall=//; s/s  iters//")
  label=$(grep -oE "label='[^']*'" "$f" | head -1 | sed "s/label=//; s/'//g")
  echo "cfg $c  $label  wall=${wall}s"
done
