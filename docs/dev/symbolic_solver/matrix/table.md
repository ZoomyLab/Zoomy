| case | check | backend | serial | mpi2 | mpi4 | gpu1 | gpu-mpi | threshold (source) |
|---|---|---|---|---|---|---|---|---|
| C1 | O1 max rel vs JAX | amrex | 8.60e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | amrex | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 max rel vs JAX | amrex | 2.84e-13 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O2 order 40->80 | amrex | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C1 | O1 order 40->80 | jax | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 order 40->80 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C2 | L1 order 40->80 | amrex | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C2 | max rel vs JAX | amrex | 1.80e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C2 | L1 order 40->80 | jax | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C3 | O1 max|u| after 100 steps | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | amrex | 5.36e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | max rel vs JAX | amrex | 5.80e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C3 | O1 max|u| after 100 steps | jax | 2.60e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | jax | 7.39e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 max rel vs JAX | amrex | 1.75e-04 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | amrex | 2.91e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | amrex | 5.76e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max rel vs JAX | amrex | 1.10e-04 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | amrex | 5.27e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | amrex | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | amrex | 1.73e-03 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | amrex | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | amrex | 1.21e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | amrex | 7.04e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max rel vs JAX | amrex | 2.94e-04 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O2 max|q/q_in - 1| n100 | amrex | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | amrex | 3.47e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | amrex | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | jax | 5.94e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | jax | 5.82e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | jax | 6.34e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | jax | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max|q/q_in - 1| n100 | jax | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | jax | 2.26e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | jax | 7.06e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max|q/q_in - 1| n100 | jax | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | jax | 3.08e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | jax | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C5 | h_min | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | >= 0 (theory (positivity)) |
| C5 | max rel vs JAX | amrex | 4.54e-15 PASS | - | - | - | - | <= 1e-09 (observed 6e-10 (wet/dry eigenvalue cluster)) |
| C5 | h_min | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C6 | C1 mass change (rel) | amrex | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | amrex | 1.73e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | amrex | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | amrex | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | jax | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | jax | 3.47e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | jax | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | jax | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | jax | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | jax | 1.95e-16 PASS | 1.95e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | amrex | - | 6.73e-15 PASS | 6.44e-15 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | jax | - | 1.41e-13 PASS | 1.41e-13 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | jax | - | 1.93e-15 PASS | 1.24e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | jax | - | 4.54e-16 PASS | 4.54e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | jax | - | 3.18e-16 PASS | 3.18e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | jax | - | 4.38e-16 PASS | 4.38e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | jax | - | 7.99e-16 PASS | 5.33e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | jax | - | 6.62e-16 PASS | 6.11e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | jax | - | 5.25e-15 PASS | 5.61e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | jax | - | 1.19e-10 PASS | 1.08e-10 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | amrex | 8.11e-11 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam max rel vs JAX | amrex | 9.79e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_newton max rel vs JAX | amrex | 1.15e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_t2 max rel vs JAX | amrex | 4.26e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | jax | 3.88e-11 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 1.45e-13 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 4.74e-16 PASS | 4.41e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | jax | - | - | - | 6.62e-16 PASS | 7.63e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | jax | - | - | - | 5.37e-15 PASS | 7.13e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | jax | - | - | - | 1.05e-10 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C10 | space O1 time O1 order (Richardson) | amrex | 1.88e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | amrex | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | amrex | 2.02e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | jax | 1.94e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | jax | 1.98e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | jax | 2.03e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C11 | smooth O1 max rel vs JAX | amrex | 1.64e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | amrex | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 max rel vs JAX | amrex | 4.99e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O2 order 80->160 | amrex | 1.70e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | amrex | 1.39e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | amrex | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max rel vs JAX | amrex | 4.52e-06 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | amrex | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C11 | smooth O1 order 80->160 | jax | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 order 80->160 | jax | 1.70e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | jax | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | jax | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C12 | runs | amrex | n/a | - | - | - | - | information |
| C12 | bump O1 order 20->40 | jax | 6.52e-01 PASS | - | - | - | - | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 order 20->40 | jax | 1.69e+00 PASS | - | - | - | - | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 max|q/q_in - 1| n100 | jax | 9.04e-03 | - | - | - | - | information |
| C12 | sub O1 max|q_y| n100 | jax | 5.10e-03 | - | - | - | - | information |
| C12 | sub O1 order 50->100 | jax | 7.80e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 max|q/q_in - 1| n100 | jax | 6.35e-04 | - | - | - | - | information |
| C12 | sub O2 max|q_y| n100 | jax | 1.56e-03 | - | - | - | - | information |
| C12 | sub O2 order 50->100 | jax | 2.55e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C13 | O1 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | max rel vs JAX | amrex | 3.91e-03 **FAIL** | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | jax | 2.66e-16 PASS | 2.66e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | jax | 2.54e-15 PASS | 9.53e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C14 | O1 max rel vs JAX | amrex | 1.52e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | amrex | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 max rel vs JAX | amrex | 1.71e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O2 order 40->80 | amrex | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 order 40->80 | jax | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 order 40->80 | jax | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C15 | runs | amrex | n/a | - | - | - | - | information |
| C15 | runs | foam | n/a | - | - | n/a | n/a | information |
| C15 | chorin jvp vs serial (rel) | jax | - | 8.95e-12 PASS | - | nan **FAIL** | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C15 | chorin |FD - jvp| (rel) | jax | 3.80e-09 PASS | 1.38e-08 PASS | - | nan **FAIL** | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | chorin |rev - jvp| (rel) | jax | 1.55e-13 PASS | 2.60e-11 PASS | - | - | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | hyperbolic jvp vs serial (rel) | jax | - | 1.48e-15 PASS | - | 1.11e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | hyperbolic |FD - jvp| (rel) | jax | 7.88e-09 PASS | 1.39e-08 PASS | - | 7.88e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | hyperbolic |rev - jvp| (rel) | jax | 6.66e-15 PASS | 1.42e-14 PASS | - | 1.85e-16 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | imex jvp vs serial (rel) | jax | - | 5.26e-15 PASS | - | 2.10e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | imex |FD - jvp| (rel) | jax | 3.66e-07 PASS | 1.93e-07 PASS | - | 1.36e-07 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | imex |rev - jvp| (rel) | jax | 1.30e-14 PASS | 4.21e-15 PASS | - | 4.56e-15 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C16 | amrex_amrex coupled vs single (max) | amrex | 4.05e-15 PASS | 4.05e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_amrex mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax coupled vs single (max) | amrex | 3.28e-15 PASS | 3.28e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam coupled vs single (max) | jax | 5.27e-15 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam mass drift | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam runs | jax | - | unsupported | - | - | - | information |
| C16 | jax_jax coupled vs single (max) | jax | 3.44e-15 PASS | 3.44e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_jax mass drift | jax | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C17 | box energy drift | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | box mass drift | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | boxnr energy drift | amrex | 1.44e-03 | - | - | - | - | information |
| C17 | boxnr mass drift | amrex | 5.91e-04 | - | - | - | - | information |
| C17 | cover_dry fine level vs fine single level (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | cover_o2 fine level vs fine single level (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | lake max|u| | amrex | 1.29e-15 PASS | - | - | - | - | <= 1e-12 (theory (well-balanced)) |
| C17 | wave order 16->32 (composite) | amrex | 2.53e+00 PASS | - | - | - | - | >= 1.5 (observed (O2, static patch; theory 2)) |
| C17 | wave16 L1(rho - exact) | amrex | 5.27e-03 | - | - | - | - | information |
| C17 | wave32 L1(rho - exact) | amrex | 9.15e-04 | - | - | - | - | information |
| C17 | runs | foam | n/a | - | - | n/a | n/a | information |
| C17 | runs | jax | n/a | - | - | - | - | information |
| C18 | Sod O1 L1(rho) n100 | jax | 2.09e-02 PASS | - | - | - | - | <= 0.042 (observed 2.09e-2 (3-D worker)) |
| C18 | Sod O1 max|u_perp| | jax | 4.66e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 order 50->100 | jax | 5.36e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C18 | Sod O1 x / y / z profiles (max) | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 L1(rho) n100 | jax | 5.13e-03 PASS | - | - | - | - | <= 0.01 (observed 5.13e-3) |
| C18 | Sod O2 max|u_perp| | jax | 9.55e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 order 50->100 | jax | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C18 | Sod O2 x / y / z profiles (max) | jax | 6.72e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 max|u - u0|, |p - 1| | jax | 8.88e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 order 16->24 | jax | 8.53e-01 PASS | - | - | - | - | >= 0.75 (observed 0.85, pre-asymptotic) |
| C19 | wave o2minmod max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2minmod order 16->24 | jax | 2.79e+00 PASS | - | - | - | - | >= 1.8 (theory 2 (observed 2.53)) |
| C19 | wave o2none max|u - u0|, |p - 1| | jax | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2none order 16->24 | jax | 2.77e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C20 | O1 energy change (rel) | jax | 1.72e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 mirror / permutation symmetry (max rel) | jax | 3.23e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 energy change (rel) | jax | 1.72e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 mirror / permutation symmetry (max rel) | jax | 5.55e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | runs | amrex | n/a | - | - | - | - | information |
| C22 | runs | foam | n/a | - | - | n/a | n/a | information |
| C22 | o1_adaptive |FD - jvp| (rel) | jax | 8.81e-09 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_adaptive |rev - jvp| (rel) | jax | 7.47e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o1_dt |FD - jvp| (rel) | jax | 2.05e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_dt |rev - jvp| (rel) | jax | 4.38e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o2_dt |FD - jvp| (rel) | jax | 4.44e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o2_dt |rev - jvp| (rel) | jax | 1.98e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 L1(rho) n100 | jax | 2.05e-02 | - | - | - | - | information |
| C24 | Sod O1 max|u_perp| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 order 50->100 | jax | 5.39e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C24 | Sod O1 x / y / z profiles (max) | jax | 8.88e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 L1(rho) n100 | jax | 5.14e-03 | - | - | - | - | information |
| C24 | Sod O2 max|u_perp| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 order 50->100 | jax | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C24 | Sod O2 x / y / z profiles (max) | jax | 2.22e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 max|u - u0|, |p - 1| | jax | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 order 32->64 | jax | 9.59e-01 PASS | - | - | - | - | >= 0.85 (observed 0.96 (theory 1)) |
| C24 | wave o2minmod max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2minmod order 32->64 | jax | 1.82e+00 PASS | - | - | - | - | >= 1.7 (observed 1.83) |
| C24 | wave o2none max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2none order 32->64 | jax | 2.32e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |

## Large tier

| level | run | result |
|---|---|---|

Failures:
- C4 sub O1 max rel vs JAX, amrex serial: 1.749e-04 (need <= 1e-12) 
- C4 sub O2 max rel vs JAX, amrex serial: 1.098e-04 (need <= 1e-12) 
- C4 trans O1 max rel vs JAX, amrex serial: 1.730e-03 (need <= 1e-12) 
- C4 trans O2 max rel vs JAX, amrex serial: 2.942e-04 (need <= 1e-12) 
- C11 sub O2 max rel vs JAX, amrex serial: 4.524e-06 (need <= 1e-12) 
- C13 max rel vs JAX, amrex serial: 3.913e-03 (need <= 1e-12) 
- C15 chorin jvp vs serial (rel), jax gpu1: nan (need <= 1e-09) 
- C15 chorin |FD - jvp| (rel), jax gpu1: nan (need <= 1e-05) 

Notes:
- C16 jax_foam runs, jax mpi2: jax/mpi2/C16/jax_foam.unsupported: ...
- C1 O1 order 40->80, jax serial: L1(h) 1.78e-03 / 1.18e-03 / 7.19e-04
- C12 bump O1 order 20->40, jax serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, jax serial: L1(h) 8.10e-04 / 4.15e-04 / 2.05e-04
- C14 O1 order 40->80, jax serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, jax serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C12 bump O2 order 20->40, jax serial: L1(h) 1.33e-03 / 7.14e-04 / 2.21e-04
- C11 smooth O2 order 80->160, jax serial: L1(h) 1.36e-03 / 4.01e-04 / 1.23e-04
- C14 O2 order 40->80, jax serial: L1(h) 7.69e-04 / 7.64e-04 / 2.67e-04
- C2 L1 order 40->80, jax serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, jax serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, jax serial: L1(h) 8.64e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, jax serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, jax serial: L1(h) 8.81e-03 / 2.41e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, jax serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, jax serial: L1(h) 8.31e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 order 50->100, jax serial: L1(h) 1.22e-02 / 8.13e-03 / 4.73e-03
- C12 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 order 50->100, jax serial: L1(h) 6.89e-03 / 7.65e-04 / 1.31e-04
- C12 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), jax serial: |h30-h60| 2.80e-04, |h60-h120| 7.28e-05
- C10 space O2 time O1 order (Richardson), jax serial: |h30-h60| 2.56e-04, |h60-h120| 6.46e-05
- C10 space O2 time O2 order (Richardson), jax serial: |h30-h60| 2.56e-04, |h60-h120| 6.29e-05
- C18 Sod O1 order 50->100, jax serial: L1(rho) 3.03e-02 / 2.09e-02
- C18 Sod O2 order 50->100, jax serial: L1(rho) 9.59e-03 / 5.13e-03
- C19 wave o1 order 16->24, jax serial: L1(rho) 6.99e-02 / 4.38e-02 / 3.10e-02
- C19 wave o1 max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C19 wave o2none order 16->24, jax serial: L1(rho) 3.88e-02 / 7.25e-03 / 2.35e-03
- C19 wave o2none max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C19 wave o2minmod order 16->24, jax serial: L1(rho) 4.02e-02 / 7.45e-03 / 2.41e-03
- C19 wave o2minmod max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C24 Sod O1 order 50->100, jax serial: L1(rho) 2.98e-02 / 2.05e-02
- C24 Sod O2 order 50->100, jax serial: L1(rho) 9.62e-03 / 5.14e-03
- C24 wave o1 order 32->64, jax serial: L1(rho) 3.03e-02 / 1.60e-02 / 8.25e-03
- C24 wave o1 max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C24 wave o2none order 32->64, jax serial: L1(rho) 3.41e-03 / 5.59e-04 / 1.12e-04
- C24 wave o2none max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C24 wave o2minmod order 32->64, jax serial: L1(rho) 3.60e-03 / 8.58e-04 / 2.43e-04
- C24 wave o2minmod max|u - u0|, |p - 1|, jax serial: contact: velocity and pressure constant
- C6 C1 mass change (rel), jax serial: max over 6 runs
- C6 C1 mass change (rel), jax mpi2: max over 1 runs
- C6 C1 mass change (rel), jax mpi4: max over 1 runs
- C6 C1 mass change (rel), jax gpu1: max over 1 runs
- C6 C2 mass change (rel), jax serial: max over 2 runs
- C6 C3 mass change (rel), jax serial: max over 2 runs
- C6 C5 mass change (rel), jax serial: max over 1 runs
- C6 C5 mass change (rel), jax mpi2: max over 1 runs
- C6 C5 mass change (rel), jax mpi4: max over 1 runs
- C6 C5 mass change (rel), jax gpu1: max over 1 runs
- C6 C5 mass change (rel), jax gpu2: max over 1 runs
- C6 C11 mass change (rel), jax serial: max over 7 runs
- C6 C12 mass change (rel), jax serial: max over 6 runs
- C6 C13 mass change (rel), jax serial: max over 4 runs
- C6 C13 mass change (rel), jax mpi2: max over 4 runs
- C6 C13 mass change (rel), jax mpi4: max over 4 runs
- C6 C14 mass change (rel), jax serial: max over 7 runs
- C6 C14 mass change (rel), jax mpi2: max over 1 runs
- C6 C14 mass change (rel), jax mpi4: max over 1 runs
- C6 C14 mass change (rel), jax gpu1: max over 1 runs
- C6 C14 mass change (rel), jax gpu2: max over 1 runs
- C6 C20 mass change (rel), jax serial: max over 2 runs
- C6 C23 mass change (rel), jax serial: max over 6 runs
- C1 O1 order 40->80, amrex serial: L1(h) 1.78e-03 / 1.18e-03 / 7.19e-04
- C11 smooth O1 order 80->160, amrex serial: L1(h) 8.10e-04 / 4.15e-04 / 2.05e-04
- C14 O1 order 40->80, amrex serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, amrex serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C11 smooth O2 order 80->160, amrex serial: L1(h) 1.36e-03 / 4.01e-04 / 1.23e-04
- C14 O2 order 40->80, amrex serial: L1(h) 7.69e-04 / 7.64e-04 / 2.67e-04
- C2 L1 order 40->80, amrex serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, amrex serial: L1(h) 1.57e-02 / 1.13e-02 / 7.57e-03
- C4 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, amrex serial: L1(h) 8.64e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, amrex serial: L1(h) 5.66e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, amrex serial: L1(h) 8.80e-03 / 2.41e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, amrex serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, amrex serial: L1(h) 8.31e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), amrex serial: |h30-h60| 2.75e-04, |h60-h120| 7.49e-05
- C10 space O2 time O1 order (Richardson), amrex serial: |h30-h60| 2.55e-04, |h60-h120| 6.39e-05
- C10 space O2 time O2 order (Richardson), amrex serial: |h30-h60| 2.56e-04, |h60-h120| 6.28e-05
- C17 boxnr mass drift, amrex serial: no reflux: not conservative
- C17 boxnr energy drift, amrex serial: no reflux: not conservative
- C6 C1 mass change (rel), amrex serial: max over 6 runs
- C6 C1 mass change (rel), amrex mpi2: max over 1 runs
- C6 C1 mass change (rel), amrex mpi4: max over 1 runs
- C6 C2 mass change (rel), amrex serial: max over 2 runs
- C6 C3 mass change (rel), amrex serial: max over 2 runs
- C6 C5 mass change (rel), amrex serial: max over 1 runs
- C6 C5 mass change (rel), amrex mpi2: max over 1 runs
- C6 C5 mass change (rel), amrex mpi4: max over 1 runs
- C6 C11 mass change (rel), amrex serial: max over 6 runs
- C6 C13 mass change (rel), amrex serial: max over 4 runs
- C6 C13 mass change (rel), amrex mpi2: max over 4 runs
- C6 C13 mass change (rel), amrex mpi4: max over 4 runs
- C6 C14 mass change (rel), amrex serial: max over 6 runs
- C6 C14 mass change (rel), amrex mpi2: max over 1 runs
- C6 C14 mass change (rel), amrex mpi4: max over 1 runs

Summed run wall time (s, without process start-up): amrex 3574, jax 6751
