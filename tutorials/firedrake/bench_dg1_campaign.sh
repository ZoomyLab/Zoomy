set -e
cd /work
pip install -q -e library/zoomy_core --no-deps 2>&1 | tail -1
pip install -q --no-deps -e library/zoomy_firedrake 2>&1 | tail -1

export ZOOMY_DIR=/work
mkdir -p /work/outputs/bench_logs
LOGDIR=/work/outputs/bench_logs

# Baseline (config 0) at 20s — the apples-to-apples reference.
echo "============================================================"
echo "  BASELINE  (config 0)  DG(1)  time_end=20s"
echo "============================================================"
python3 -u tutorials/firedrake/bench_malpasset_dg1.py 0 20.0 _20s 2>&1 | tee $LOGDIR/cfg0_20s.log

# Sweep configs 1-5 at 5s each (cheap exploration).
for CFG in 1 2 3 4 5; do
  echo ""
  echo "============================================================"
  echo "  VARIATION  (config $CFG)  DG(1)  time_end=5s"
  echo "============================================================"
  python3 -u tutorials/firedrake/bench_malpasset_dg1.py $CFG 5.0 _5s 2>&1 | tee $LOGDIR/cfg${CFG}_5s.log || true
done

# Summary table
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
for f in $LOGDIR/cfg0_20s.log $LOGDIR/cfg1_5s.log $LOGDIR/cfg2_5s.log $LOGDIR/cfg3_5s.log $LOGDIR/cfg4_5s.log $LOGDIR/cfg5_5s.log; do
  echo "--- $f ---"
  grep -E "DONE|mass:|per-stage|FAIL" $f | head -12 || true
done

# Pick best of configs 1-5 (least wall time on 5s) and rerun at 20s.
BEST_CFG=$(python3 -c "
import re, os
best = (1e30, 0)
for c in [1,2,3,4,5]:
    fn = '$LOGDIR'+f'/cfg{c}_5s.log'
    if not os.path.exists(fn): continue
    txt = open(fn).read()
    m = re.search(r'DONE.*wall=([0-9.]+)s', txt)
    if m and float(m.group(1)) < best[0]:
        best = (float(m.group(1)), c)
print(best[1])
")
echo ""
echo "============================================================"
echo "  BEST 5s CONFIG = $BEST_CFG  →  rerun at 20s"
echo "============================================================"
python3 -u tutorials/firedrake/bench_malpasset_dg1.py $BEST_CFG 20.0 _20s_best 2>&1 | tee $LOGDIR/cfg${BEST_CFG}_20s_best.log

echo ""
echo "DONE all configs."
