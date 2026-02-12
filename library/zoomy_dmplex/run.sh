#!/bin/bash
mpiexec -n 1 ./solver_cpu -snes_monitor
# mpiexec -n 4 ./solver_cpu -ts_adapt_monitor 
# mpiexec -n 4 ./solver_cpu 
