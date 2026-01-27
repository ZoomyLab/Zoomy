#!/bin/bash
# mpiexec -n 1 ./solver_cpu -snes_monitor
mpiexec -n 4 ./solver_cpu -order 2
