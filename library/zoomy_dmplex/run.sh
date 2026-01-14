#!/bin/bash

# Use -n 2 or more to test the parallel fix
mpiexec -n 1 ./solver \
    -dm_plex_filename mesh_coarse.msh \
    -petscfv_type upwind \
    -ts_monitor \
    -ts_max_steps 100 \
    -ts_monitor_solution_vtk 'output-%03d.vtu' \
    -ufv_cfl 0.4