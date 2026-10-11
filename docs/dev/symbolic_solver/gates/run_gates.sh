#!/bin/bash
# All symbolic-solver gates, sequentially; one summary line per check.
#   bash run_gates.sh [gate.py ...]   (default: all)
#   "mpi:N gate.py ..." runs the gate on N MPI ranks
cd "$(dirname "$0")"
W=$(cd ../../../.. && pwd)
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export PYTHONPATH=$W/library/zoomy_core:$W/library/zoomy_jax:$W/library/zoomy_plotting:$W/library/zoomy_foam
GATES=("$@")
[ ${#GATES[@]} -eq 0 ] && GATES=(
  "gate1_vs_solver_jax.py" "gate4_mood_output.py" "gate6_imex.py" "gate7_chorin.py"
  "gate2_ad.py" "gate5_mesh_ad.py" "gate8_chorin_ad.py" "gate13_pressure_newton.py"
  "gate3_precice.py" "gate3b_subcycling.py"
  "gate9_foam.py 0:1 1:2 2d:2 eta:2 dry:2 smooth:2" "gate11_foam_imex.py" "gate12_foam_chorin.py"
  "gate23_foam_parallel.py bump1 2 4" "gate23_foam_parallel.py bump2 2 4" "gate23_foam_parallel.py tri2 2 4"
  "gate23_foam_parallel.py dry 2 4" "gate23_foam_parallel.py chorin1 2 4" "gate23_foam_parallel.py chorin2 2 4"
  "gate24_foam_chorin2d.py ref" "gate24_foam_chorin2d.py foam 1 2 4" "gate25_foam_newton.py 2 4"
  "gate26_foam_couple_parallel.py 2 2"
  "gate10_xcouple.py jax" "gate10_xcouple.py foam" "probe_audusse_vars.py"
  "mpi:4 gate16_parallel.py" "mpi:2 gate17_parallel_ad.py o1"
  "mpi:3 gate17_parallel_ad.py o2" "mpi:3 gate18_parallel_output.py" "gate19_mesh_builders.py" "mpi:3 gate20_parallel_write.py" "mpi:3 gate22_parallel_chorin_ad.py" "mpi:3 gate29_parallel_imex.py" "mpi:3 gate32_parallel_periodic.py" "gate31_parallel_implicit_diffusion.py" "mpi:3 gate31_parallel_implicit_diffusion.py par")
for g in "${GATES[@]}"; do
  echo "== $g"
  run="python"
  [[ $g == mpi:* ]] && { run="mpiexec -n ${g%% *}"; run="${run/mpi:/} python"; g=${g#* }; }
  timeout 3000 micromamba run -n zoomy $run $g 2>&1 \
    | grep -v "Warning\|warn\|cpu_aot\|pjrt\|^\s*$" \
    | grep -E "max|MOOD|io.write|AD|Error|error|Traceback|File \"|t foam|mass|steps|dJ|checks|O[12] |^[A-C] |ranks|Mesh" | head -40
done
echo GATES_DONE
