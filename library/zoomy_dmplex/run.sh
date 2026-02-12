#!/bin/bash
# mpiexec -n 1 ./solver_cpu -snes_monitor -snes_linesearch_type basic -pc_type lu
mpiexec -n 2 ./solver_cpu -snes_monitor
# mpiexec -n 4 ./solver_cpu -ts_adapt_monitor 
# mpiexec -n 4 ./solver_cpu 
