#!/bin/bash
mpiexec -n 3 ./solver_cpu \
    -dm_plex_filename mesh_coarse.msh \
    -ts_max_steps 100 \
    -ufv_cfl 0.4 \
    -ts_monitor \
    -order 2 \
