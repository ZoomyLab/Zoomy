#!/bin/bash

mpiexec -n 1 ./solver \
    -dm_plex_filename mesh_coarse.msh \
    -dm_distribute \
    -petscfv_type leastsquares \
    -ts_monitor \
    -ts_max_steps 100 \
    -ts_monitor_solution_vtk 'output-%03d.vtu' \
    -ufv_cfl 0.4

