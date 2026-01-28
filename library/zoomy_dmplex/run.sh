#!/bin/bash
# mpiexec -n 1 ./solver_cpu -snes_monitor
mpiexec -n 2 ./solver_cpu -order 2
