| case | check | backend | serial | mpi2 | mpi4 | gpu1 | gpu-mpi | threshold (source) |
|---|---|---|---|---|---|---|---|---|
| C1 | O1 max rel vs JAX | amrex | 8.60e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | amrex | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 max rel vs JAX | amrex | 9.14e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O2 order 40->80 | amrex | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C1 | O1 max rel vs JAX | foam | 7.78e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | foam | 7.15e-01 PASS | - | - | n/a | n/a | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 foam-JAX L1 diff order 40->80 | foam | 2.70e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C1 | O2 order 40->80 | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C1 | O1 order 40->80 | jax | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 order 40->80 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C2 | L1 order 40->80 | amrex | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C2 | max rel vs JAX | amrex | 1.55e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C2 | L1 order 40->80 | foam | 8.94e-01 PASS | - | - | n/a | n/a | >= 0.6 (spec) |
| C2 | L1 vs JAX n80 (rel) | foam | 2.58e-04 PASS | - | - | n/a | n/a | <= 0.0015 (observed (foam 7.1e-4)) |
| C2 | L1 order 40->80 | jax | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C3 | O1 max|u| after 100 steps | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | amrex | 5.36e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | max rel vs JAX | amrex | 5.80e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C3 | O1 max|u| after 100 steps | foam | 5.11e-16 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | foam | 1.27e-15 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O1 max|u| after 100 steps | jax | 2.60e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | jax | 7.39e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 max rel vs JAX | amrex | 1.19e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | amrex | 3.72e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | amrex | 5.82e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max rel vs JAX | amrex | 8.07e-14 PASS | - | - | - | - | <= 3.24985e-12 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 3.2e-13)) |
| C4 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | amrex | 7.60e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | amrex | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | amrex | 8.28e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | amrex | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | amrex | 5.68e-17 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | amrex | 7.06e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max rel vs JAX | amrex | 3.67e-12 PASS | - | - | - | - | <= 1.33656e-10 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 1.3e-11)) |
| C4 | trans O2 max|q/q_in - 1| n100 | amrex | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | amrex | 6.81e-15 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | amrex | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | sub O1 max rel vs JAX | foam | 1.19e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | foam | 1.37e-02 | - | - | n/a | n/a | information |
| C4 | sub O1 max|q_y| n100 | foam | 5.95e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | foam | 5.82e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 foam-JAX L1 diff order 50->100 | foam | 4.37e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C4 | sub O2 max|q/q_in - 1| n100 | foam | 2.96e-03 | - | - | n/a | n/a | information |
| C4 | sub O2 max|q_y| n100 | foam | 5.99e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | foam | 2.22e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | foam | 6.67e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | foam | 5.13e-02 | - | - | n/a | n/a | information |
| C4 | trans O1 max|q_y| n100 | foam | 2.46e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | foam | 7.06e-01 PASS | - | - | n/a | n/a | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 foam-JAX L1 diff n100 | foam | 2.14e-13 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C4 | trans O2 max|q/q_in - 1| n100 | foam | 1.91e-03 | - | - | n/a | n/a | information |
| C4 | trans O2 max|q_y| n100 | foam | 3.65e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | foam | 2.85e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | jax | 4.66e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | jax | 5.82e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | jax | 9.06e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | jax | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max|q/q_in - 1| n100 | jax | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | jax | 1.02e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | jax | 7.06e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max|q/q_in - 1| n100 | jax | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | jax | 3.61e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | jax | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C5 | h_min | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C5 | max rel vs JAX | amrex | 4.54e-15 PASS | - | - | - | - | <= 1e-09 (observed 6e-10 (wet/dry eigenvalue cluster)) |
| C5 | L1 rel vs JAX | foam | 1.69e-02 PASS | - | - | n/a | n/a | <= 0.035 (observed (foam 1.7e-2)) |
| C5 | h_min | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | >= 0 (theory (positivity)) |
| C5 | h_min | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C6 | C1 mass change (rel) | amrex | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | amrex | 1.73e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | amrex | 1.73e-16 PASS | 1.73e-16 PASS | 1.73e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | amrex | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | amrex | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | amrex | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | foam | 2.82e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | foam | 3.47e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | foam | 1.41e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | foam | 1.41e-16 PASS | 0.00e+00 PASS | 1.41e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | foam | 1.41e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | foam | 2.20e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | foam | 2.22e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | foam | 0.00e+00 PASS | 1.95e-16 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | jax | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | jax | 1.73e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | jax | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | jax | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | jax | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | jax | 1.95e-16 PASS | 1.95e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 1.95e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | amrex | - | 7.15e-15 PASS | 6.48e-15 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | foam | - | 1.23e-15 PASS | 1.19e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | foam | - | 8.47e-16 PASS | 7.42e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | foam | - | 1.04e-14 PASS | 1.06e-14 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | foam | - | 3.18e-16 PASS | 3.18e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | foam | - | 4.31e-16 PASS | 5.74e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | foam | - | 7.99e-16 PASS | 9.33e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | foam | - | 5.91e-14 PASS | 5.93e-14 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | foam | - | 1.80e-14 PASS | 1.27e-14 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | foam | - | 1.11e-14 PASS | 9.74e-15 PASS | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | jax | - | 1.03e-13 PASS | 1.03e-13 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | jax | - | 1.22e-15 PASS | 1.22e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | jax | - | 4.54e-16 PASS | 4.54e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | jax | - | 3.18e-16 PASS | 3.18e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | jax | - | 5.84e-16 PASS | 5.84e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | jax | - | 1.78e-15 PASS | 1.15e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | jax | - | 1.22e-15 PASS | 1.12e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | jax | - | 5.25e-15 PASS | 5.61e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | jax | - | 6.47e-11 PASS | 6.30e-11 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | amrex | 5.98e-11 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam max rel vs JAX | amrex | 7.65e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_newton max rel vs JAX | amrex | 3.14e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_t2 max rel vs JAX | amrex | 9.13e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | foam | 5.98e-11 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam L1 rel vs JAX | foam | 7.03e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_newton L1 rel vs JAX | foam | 2.13e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_t2 L1 rel vs JAX | foam | 5.49e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_tri L1 rel vs JAX | foam | 1.98e-10 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | Newton vs linear pressure (max rel) | jax | 6.57e-11 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 8.44e-15 PASS | 8.44e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | amrex | - | - | - | 4.56e-15 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 2.06e-13 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 4.15e-16 PASS | 4.80e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | jax | - | - | - | 8.33e-16 PASS | 1.01e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | jax | - | - | - | 5.13e-15 PASS | 7.55e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | jax | - | - | - | 4.16e-11 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C10 | space O1 time O1 order (Richardson) | amrex | 1.92e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | amrex | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | amrex | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | foam | 1.92e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | foam | 2.01e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | jax | 1.92e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | jax | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | jax | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C11 | smooth O1 max rel vs JAX | amrex | 1.64e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | amrex | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 max rel vs JAX | amrex | 7.82e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O2 order 80->160 | amrex | 1.69e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | amrex | 1.78e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | amrex | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max rel vs JAX | amrex | 4.52e-06 PASS | - | - | - | - | <= 0.000196356 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 2.0e-05)) |
| C11 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | amrex | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C11 | smooth O1 max rel vs JAX | foam | 1.49e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | foam | 1.02e+00 PASS | - | - | n/a | n/a | >= 0.9 (theory 1) |
| C11 | smooth O2 foam-JAX L1 diff order 80->160 | foam | 2.36e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | smooth O2 order 80->160 | foam | 1.67e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | foam | 1.78e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | foam | 1.37e-02 | - | - | n/a | n/a | information |
| C11 | sub O1 order 50->100 | foam | 5.89e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 foam-JAX L1 diff order 50->100 | foam | 1.50e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | sub O2 max|q/q_in - 1| n100 | foam | 2.96e-03 | - | - | n/a | n/a | information |
| C11 | sub O2 order 50->100 | foam | 2.28e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C11 | smooth O1 order 80->160 | jax | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 order 80->160 | jax | 1.69e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | jax | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | jax | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C12 | runs | amrex | n/a | - | - | - | - | information |
| C12 | bump O1 foam-JAX L1 diff n40 | foam | 2.31e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C12 | bump O1 order 20->40 | foam | 6.52e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 foam-JAX L1 diff order 20->40 | foam | 1.44e+00 PASS | - | - | n/a | n/a | >= 0.25 (observed 0.33-0.39 (O1 triangles: both converge slowly)) |
| C12 | bump O2 order 20->40 | foam | 1.92e+00 PASS | - | - | n/a | n/a | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 foam-JAX L1 diff n100 | foam | 3.18e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C12 | sub O1 max|q/q_in - 1| n100 | foam | 9.43e-03 | - | - | n/a | n/a | information |
| C12 | sub O1 max|q_y| n100 | foam | 6.70e-03 | - | - | n/a | n/a | information |
| C12 | sub O1 order 50->100 | foam | 6.94e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 foam-JAX L1 diff order 50->100 | foam | 2.08e+00 PASS | - | - | n/a | n/a | >= 0.25 (observed 0.33-0.39 (O1 triangles: both converge slowly)) |
| C12 | sub O2 max|q/q_in - 1| n100 | foam | 1.67e-03 | - | - | n/a | n/a | information |
| C12 | sub O2 max|q_y| n100 | foam | 4.37e-03 | - | - | n/a | n/a | information |
| C12 | sub O2 order 50->100 | foam | 2.76e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C12 | bump O1 order 20->40 | jax | 6.52e-01 PASS | - | - | - | - | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 order 20->40 | jax | 1.69e+00 PASS | - | - | - | - | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 max|q/q_in - 1| n100 | jax | 9.43e-03 | - | - | - | - | information |
| C12 | sub O1 max|q_y| n100 | jax | 6.70e-03 | - | - | - | - | information |
| C12 | sub O1 order 50->100 | jax | 6.94e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 max|q/q_in - 1| n100 | jax | 1.47e-03 | - | - | - | - | information |
| C12 | sub O2 max|q_y| n100 | jax | 3.24e-03 | - | - | - | - | information |
| C12 | sub O2 order 50->100 | jax | 2.42e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C13 | O1 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | max rel vs JAX | amrex | 1.32e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 max rel vs JAX | foam | 6.12e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | foam | 5.33e-16 PASS | 4.79e-16 PASS | 5.33e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | foam | 7.42e-16 PASS | 1.17e-15 PASS | 5.30e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | jax | 2.66e-16 PASS | 2.66e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | jax | 9.00e-16 PASS | 5.30e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C14 | O1 max rel vs JAX | amrex | 1.52e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | amrex | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 max rel vs JAX | amrex | 1.71e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O2 order 40->80 | amrex | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 max rel vs JAX | foam | 1.52e-12 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | foam | 6.31e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 foam-JAX L1 diff order 40->80 | foam | 1.20e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C14 | O2 order 40->80 | foam | 1.55e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 order 40->80 | jax | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 order 40->80 | jax | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C15 | runs | amrex | n/a | - | - | - | - | information |
| C15 | runs | foam | n/a | - | - | n/a | n/a | information |
| C15 | chorin jvp vs serial (rel) | jax | - | 5.36e-12 PASS | - | 5.54e-15 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C15 | chorin |FD - jvp| (rel) | jax | 6.39e-09 PASS | 6.40e-09 PASS | - | 6.39e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | chorin |rev - jvp| (rel) | jax | 8.87e-15 PASS | 4.07e-11 PASS | - | 4.62e-13 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | hyperbolic jvp vs serial (rel) | jax | - | 5.55e-16 PASS | - | 3.70e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | hyperbolic |FD - jvp| (rel) | jax | 7.88e-09 PASS | 1.39e-08 PASS | - | 7.88e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | hyperbolic |rev - jvp| (rel) | jax | 7.77e-15 PASS | 1.65e-14 PASS | - | 2.03e-15 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | imex jvp vs serial (rel) | jax | - | 1.75e-15 PASS | - | 3.51e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | imex |FD - jvp| (rel) | jax | 3.66e-07 PASS | 1.93e-07 PASS | - | 1.36e-07 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | imex |rev - jvp| (rel) | jax | 5.79e-15 PASS | 1.49e-14 PASS | - | 7.01e-16 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C16 | amrex_amrex coupled vs single (max) | amrex | 4.05e-15 PASS | 4.05e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_amrex mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax coupled vs single (max) | amrex | 3.28e-15 PASS | 3.28e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam coupled vs single (max) | foam | 4.94e-15 PASS | 4.94e-15 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam mass drift | foam | 0.00e+00 PASS | 0.00e+00 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam runs | jax | nan | unsupported | - | - | - | information |
| C16 | jax_jax coupled vs single (max) | jax | 3.44e-15 PASS | 3.44e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_jax mass drift | jax | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C17 | box energy drift | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | box mass drift | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | box vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | boxnr energy drift | amrex | 1.44e-03 | - | - | - | - | information |
| C17 | boxnr mass drift | amrex | 5.91e-04 | - | - | - | - | information |
| C17 | boxnr vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | cover_dry fine level vs fine single level (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | cover_dry vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | cover_o2 fine level vs fine single level (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C17 | cover_o2 vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | lake max|u| | amrex | 1.29e-15 PASS | - | - | - | - | <= 1e-12 (theory (well-balanced)) |
| C17 | wave order 16->32 (composite) | amrex | 2.53e+00 PASS | - | - | - | - | >= 1.5 (observed (O2, static patch; theory 2)) |
| C17 | wave16 L1(rho - exact) | amrex | 5.27e-03 | - | - | - | - | information |
| C17 | wave16 vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | wave32 L1(rho - exact) | amrex | 9.15e-04 | - | - | - | - | information |
| C17 | wave32 vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | runs | foam | n/a | - | - | n/a | n/a | information |
| C17 | runs | jax | n/a | - | - | - | - | information |
| C18 | Sod O1 L1(rho) n100 | amrex | 2.09e-02 PASS | - | - | - | - | <= 0.042 (observed 2.09e-2 (3-D worker)) |
| C18 | Sod O1 max|u_perp| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 order 50->100 | amrex | 5.36e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C18 | Sod O1 x / y / z profiles (max) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 L1(rho) n100 | amrex | 5.13e-03 PASS | - | - | - | - | <= 0.01 (observed 5.13e-3) |
| C18 | Sod O2 max|u_perp| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 order 50->100 | amrex | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C18 | Sod O2 x / y / z profiles (max) | amrex | 1.89e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 L1(rho) n100 | foam | 2.09e-02 PASS | - | - | n/a | n/a | <= 0.042 (observed 2.09e-2 (3-D worker)) |
| C18 | Sod O1 max|u_perp| | foam | 1.02e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 order 50->100 | foam | 5.36e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C18 | Sod O1 x / y / z profiles (max) | foam | 4.88e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 L1(rho) n100 | foam | 5.13e-03 PASS | - | - | n/a | n/a | <= 0.01 (observed 5.13e-3) |
| C18 | Sod O2 max|u_perp| | foam | 6.06e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 order 50->100 | foam | 9.02e-01 PASS | - | - | n/a | n/a | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C18 | Sod O2 x / y / z profiles (max) | foam | 3.71e-14 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 L1(rho) n100 | jax | 2.09e-02 PASS | - | - | - | - | <= 0.042 (observed 2.09e-2 (3-D worker)) |
| C18 | Sod O1 max|u_perp| | jax | 4.66e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 order 50->100 | jax | 5.36e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C18 | Sod O1 x / y / z profiles (max) | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 L1(rho) n100 | jax | 5.13e-03 PASS | - | - | - | - | <= 0.01 (observed 5.13e-3) |
| C18 | Sod O2 max|u_perp| | jax | 1.36e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 order 50->100 | jax | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C18 | Sod O2 x / y / z profiles (max) | jax | 7.22e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 max|u - u0|, |p - 1| | amrex | 6.66e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 order 16->24 | amrex | 8.53e-01 PASS | - | - | - | - | >= 0.75 (observed 0.85, pre-asymptotic) |
| C19 | wave o2minmod max|u - u0|, |p - 1| | amrex | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2minmod order 16->24 | amrex | 2.79e+00 PASS | - | - | - | - | >= 1.8 (theory 2 (observed 2.53)) |
| C19 | wave o2none max|u - u0|, |p - 1| | amrex | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2none order 16->24 | amrex | 2.77e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C19 | wave o1 max|u - u0|, |p - 1| | foam | 1.11e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 order 16->24 | foam | 8.53e-01 PASS | - | - | n/a | n/a | >= 0.75 (observed 0.85, pre-asymptotic) |
| C19 | wave o2minmod max|u - u0|, |p - 1| | foam | 1.55e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2minmod order 16->24 | foam | 1.87e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2 (observed 2.53)) |
| C19 | wave o2none max|u - u0|, |p - 1| | foam | 1.33e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2none order 16->24 | foam | 2.41e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C19 | wave o1 max|u - u0|, |p - 1| | jax | 8.88e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 order 16->24 | jax | 8.53e-01 PASS | - | - | - | - | >= 0.75 (observed 0.85, pre-asymptotic) |
| C19 | wave o2minmod max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2minmod order 16->24 | jax | 2.79e+00 PASS | - | - | - | - | >= 1.8 (theory 2 (observed 2.53)) |
| C19 | wave o2none max|u - u0|, |p - 1| | jax | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2none order 16->24 | jax | 2.77e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C20 | O1 energy change (rel) | amrex | 1.72e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 mirror / permutation symmetry (max rel) | amrex | 3.23e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 energy change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 mirror / permutation symmetry (max rel) | amrex | 4.16e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 energy change (rel) | foam | 1.72e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 mirror / permutation symmetry (max rel) | foam | 3.23e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 energy change (rel) | foam | 3.44e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 mirror / permutation symmetry (max rel) | foam | 8.00e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 energy change (rel) | jax | 1.72e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 mirror / permutation symmetry (max rel) | jax | 3.23e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 energy change (rel) | jax | 1.72e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 mirror / permutation symmetry (max rel) | jax | 6.94e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | runs | amrex | n/a | - | - | - | - | information |
| C22 | runs | foam | n/a | - | - | n/a | n/a | information |
| C22 | o1_adaptive |FD - jvp| (rel) | jax | 8.81e-09 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_adaptive |rev - jvp| (rel) | jax | 7.47e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o1_dt |FD - jvp| (rel) | jax | 2.05e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_dt |rev - jvp| (rel) | jax | 2.12e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o2_dt |FD - jvp| (rel) | jax | 4.44e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o2_dt |rev - jvp| (rel) | jax | 1.58e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n16 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n24 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n8 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O2 max rel vs JAX | amrex | 3.08e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n16 max rel vs JAX | foam | 3.54e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n24 max rel vs JAX | foam | 5.31e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n8 max rel vs JAX | foam | 3.54e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O2 L1|backend - JAX| order 16->24 | foam | 1.98e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed 2.11 (different stencils, converging)) |
| C24 | Sod O1 L1(rho) n100 | amrex | 2.05e-02 | - | - | - | - | information |
| C24 | Sod O1 max|u_perp| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 order 50->100 | amrex | 5.39e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C24 | Sod O1 x / y / z profiles (max) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 L1(rho) n100 | amrex | 5.14e-03 | - | - | - | - | information |
| C24 | Sod O2 max|u_perp| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 order 50->100 | amrex | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C24 | Sod O2 x / y / z profiles (max) | amrex | 4.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 max|u - u0|, |p - 1| | amrex | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 order 32->64 | amrex | 9.59e-01 PASS | - | - | - | - | >= 0.85 (observed 0.96 (theory 1)) |
| C24 | wave o2minmod max|u - u0|, |p - 1| | amrex | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2minmod order 32->64 | amrex | 1.82e+00 PASS | - | - | - | - | >= 1.7 (observed 1.83) |
| C24 | wave o2none max|u - u0|, |p - 1| | amrex | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2none order 32->64 | amrex | 2.32e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C24 | Sod O1 L1(rho) n100 | foam | 2.05e-02 | - | - | n/a | n/a | information |
| C24 | Sod O1 max|u_perp| | foam | 5.28e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 order 50->100 | foam | 5.39e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C24 | Sod O1 x / y / z profiles (max) | foam | 2.66e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 L1(rho) n100 | foam | 5.14e-03 | - | - | n/a | n/a | information |
| C24 | Sod O2 max|u_perp| | foam | 4.73e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 order 50->100 | foam | 9.02e-01 PASS | - | - | n/a | n/a | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C24 | Sod O2 x / y / z profiles (max) | foam | 7.55e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 max|u - u0|, |p - 1| | foam | 1.11e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 order 32->64 | foam | 9.59e-01 PASS | - | - | n/a | n/a | >= 0.85 (observed 0.96 (theory 1)) |
| C24 | wave o2minmod max|u - u0|, |p - 1| | foam | 1.78e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2minmod order 32->64 | foam | 1.89e+00 PASS | - | - | n/a | n/a | >= 1.7 (observed 1.83) |
| C24 | wave o2none max|u - u0|, |p - 1| | foam | 1.55e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2none order 32->64 | foam | 2.07e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C24 | Sod O1 L1(rho) n100 | jax | 2.05e-02 | - | - | - | - | information |
| C24 | Sod O1 max|u_perp| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 order 50->100 | jax | 5.39e-01 PASS | - | - | - | - | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C24 | Sod O1 x / y / z profiles (max) | jax | 8.88e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 L1(rho) n100 | jax | 5.14e-03 | - | - | - | - | information |
| C24 | Sod O2 max|u_perp| | jax | 9.27e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 order 50->100 | jax | 9.02e-01 PASS | - | - | - | - | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C24 | Sod O2 x / y / z profiles (max) | jax | 2.66e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 max|u - u0|, |p - 1| | jax | 1.11e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 order 32->64 | jax | 9.59e-01 PASS | - | - | - | - | >= 0.85 (observed 0.96 (theory 1)) |
| C24 | wave o2minmod max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2minmod order 32->64 | jax | 1.82e+00 PASS | - | - | - | - | >= 1.7 (observed 1.83) |
| C24 | wave o2none max|u - u0|, |p - 1| | jax | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2none order 32->64 | jax | 2.32e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C26 | max rel vs JAX | amrex | 6.72e-14 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C26 | L1 rel vs JAX | foam | 1.66e-03 | - | - | n/a | n/a | information |
| C27 | ODE time_order 1 order 32->64 steps | amrex | 1.01e+00 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C27 | ODE time_order 2 order 32->64 steps | amrex | 1.92e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C27 | ODE time_order 3 order 32->64 steps | amrex | 3.00e+00 PASS | - | - | - | - | >= 2.8 (theory 3, slack 0.2) |
| C27 | wave time_order 1 order dt/2->dt/4 | amrex | 1.26e+00 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C27 | wave time_order 2 order dt/2->dt/4 | amrex | 2.08e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C27 | wave time_order 3 order dt/2->dt/4 | amrex | 3.02e+00 PASS | - | - | - | - | >= 2.8 (theory 3, slack 0.2) |
| C27 | ODE time_order 1 order 32->64 steps | foam | 1.01e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C27 | ODE time_order 2 order 32->64 steps | foam | 1.92e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C27 | ODE time_order 3 order 32->64 steps | foam | 3.00e+00 PASS | - | - | n/a | n/a | >= 2.8 (theory 3, slack 0.2) |
| C27 | wave time_order 1 order dt/2->dt/4 | foam | 1.26e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C27 | wave time_order 2 order dt/2->dt/4 | foam | 2.08e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C27 | wave time_order 3 order dt/2->dt/4 | foam | 3.02e+00 PASS | - | - | n/a | n/a | >= 2.8 (theory 3, slack 0.2) |
| C27 | ODE control (RK3, wrong stage time) order | jax | 9.92e-01 PASS | - | - | - | - | <= 2.5 (gate 40 control: a wrong stage time loses the order) |
| C27 | ODE time_order 1 order 32->64 steps | jax | 1.01e+00 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C27 | ODE time_order 2 order 32->64 steps | jax | 1.92e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C27 | ODE time_order 3 order 32->64 steps | jax | 3.00e+00 PASS | - | - | - | - | >= 2.8 (theory 3, slack 0.2) |
| C27 | wave time_order 1 order dt/2->dt/4 | jax | 1.26e+00 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C27 | wave time_order 2 order dt/2->dt/4 | jax | 2.08e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C27 | wave time_order 3 order dt/2->dt/4 | jax | 3.02e+00 PASS | - | - | - | - | >= 2.8 (theory 3, slack 0.2) |
| C28 | 1d expl dir0 max rel vs JAX | amrex | 1.28e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl dirL max rel vs JAX | amrex | 1.04e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl neu max rel vs JAX | amrex | 1.46e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl per max rel vs JAX | amrex | 1.21e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max rel vs JAX | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max|u - L| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | 1d impl dir0 max rel vs JAX | amrex | 6.72e-11 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl dirL max rel vs JAX | amrex | 1.40e-11 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl neu max rel vs JAX | amrex | 4.48e-11 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl per max rel vs JAX | amrex | 2.52e-10 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max rel vs JAX | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max|u - L| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | quad expl dir0 max rel vs JAX | amrex | 1.20e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl dirL max rel vs JAX | amrex | 7.59e-16 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl neu max rel vs JAX | amrex | 1.20e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl per max rel vs JAX | amrex | 2.75e-15 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max rel vs JAX | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max|u - L| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | quad impl dir0 max rel vs JAX | amrex | 1.35e-15 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl dirL max rel vs JAX | amrex | 3.04e-16 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl neu max rel vs JAX | amrex | 6.00e-16 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl per max rel vs JAX | amrex | 2.16e-10 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max rel vs JAX | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max|u - L| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | tri max rel vs JAX (n/a: AMReX takes Cartesian grids only) | amrex | n/a | - | - | - | - | information |
| C28 | 1d expl dir0 max rel vs JAX | foam | 1.28e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl dirL max rel vs JAX | foam | 1.04e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl neu max rel vs JAX | foam | 1.46e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl per max rel vs JAX | foam | 1.21e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | 1d impl dir0 max rel vs JAX | foam | 5.06e-11 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl dirL max rel vs JAX | foam | 8.14e-12 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl neu max rel vs JAX | foam | 3.45e-11 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl per max rel vs JAX | foam | 1.95e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | quad expl dir0 max rel vs JAX | foam | 1.20e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl dirL max rel vs JAX | foam | 7.59e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl neu max rel vs JAX | foam | 1.20e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl per max rel vs JAX | foam | 2.75e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | quad impl dir0 max rel vs JAX | foam | 1.35e-15 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl dirL max rel vs JAX | foam | 3.04e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl neu max rel vs JAX | foam | 6.00e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl per max rel vs JAX | foam | 2.26e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri expl dir0 max rel vs JAX | foam | 4.48e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl dirL max rel vs JAX | foam | 3.01e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl neu max rel vs JAX | foam | 9.71e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl per max rel vs JAX | foam | 4.24e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri impl dir0 max rel vs JAX | foam | 6.72e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl dirL max rel vs JAX | foam | 1.51e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl neu max rel vs JAX | foam | 6.44e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl per max rel vs JAX | foam | 1.04e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | 1d expl dir0 L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl dirL L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl neu L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl per L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | 1d impl dir0 L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl dirL L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl neu L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl per L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | quad expl dir0 L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl dirL L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl neu L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl per L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | quad impl dir0 L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl dirL L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl neu L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl per L_inf order 32->64 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | tri expl dir0 L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl dirL L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl neu L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl per L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C28 | tri impl dir0 L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl dirL L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl neu L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl per L_inf order 32->64 | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl steady max|u - L| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C29 | runs (n/a: gate 44 calls the JAX runtime (derivative_aux)) | amrex | n/a | - | - | - | - | information |
| C29 | runs (n/a: gate 44 calls the JAX runtime (derivative_aux)) | foam | n/a | - | - | n/a | n/a | information |
| C29 | p2 dir d1 boundary order 24->48 | jax | 1.97e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 dir d1 interior order 24->48 | jax | 1.98e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 dir d2 boundary order 24->48 | jax | 9.29e-01 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 dir d2 interior order 24->48 | jax | 1.99e+00 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 neu d1 boundary order 24->48 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 neu d1 interior order 24->48 | jax | 1.98e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 neu d2 boundary order 24->48 | jax | 9.20e-01 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 neu d2 interior order 24->48 | jax | 1.99e+00 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 dir d1 boundary order 24->48 | jax | 2.94e+00 PASS | - | - | - | - | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 dir d1 interior order 24->48 | jax | 3.32e+00 PASS | - | - | - | - | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 dir d2 boundary order 24->48 | jax | 1.96e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 dir d2 interior order 24->48 | jax | 1.93e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 dir d3 boundary order 24->48 | jax | 9.34e-01 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 dir d3 interior order 24->48 | jax | 1.34e+00 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 neu d1 boundary order 24->48 | jax | 2.94e+00 PASS | - | - | - | - | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 neu d1 interior order 24->48 | jax | 3.32e+00 PASS | - | - | - | - | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 neu d2 boundary order 24->48 | jax | 1.96e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 neu d2 interior order 24->48 | jax | 1.93e+00 PASS | - | - | - | - | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 neu d3 boundary order 24->48 | jax | 8.27e-01 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 neu d3 interior order 24->48 | jax | 1.34e+00 PASS | - | - | - | - | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C30 | runs (n/a: gate 39 calls the JAX runtime (lsq_gradient_per_field)) | amrex | n/a | - | - | - | - | information |
| C30 | runs (n/a: gate 39 calls the JAX runtime (lsq_gradient_per_field)) | foam | n/a | - | - | n/a | n/a | information |
| C30 | gate 39 checks failing | jax | 0.00e+00 PASS | - | - | - | - | <= 0 (spec, gate 39: exact to 1e-09) |

## Large tier

| level | run | result |
|---|---|---|

Failures:
- none

Notes:
- C16 jax_foam runs, jax serial: jax/serial/C16/jax_foam.err: ...
- C16 jax_foam runs, jax mpi2: jax/mpi2/C16/jax_foam.unsupported: ...
- C1 O1 order 40->80, jax serial: L1(h) 1.78e-03 / 1.18e-03 / 7.19e-04
- C12 bump O1 order 20->40, jax serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, jax serial: L1(h) 8.10e-04 / 4.15e-04 / 2.05e-04
- C14 O1 order 40->80, jax serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, jax serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C12 bump O2 order 20->40, jax serial: L1(h) 1.33e-03 / 7.14e-04 / 2.21e-04
- C11 smooth O2 order 80->160, jax serial: L1(h) 1.34e-03 / 3.97e-04 / 1.23e-04
- C14 O2 order 40->80, jax serial: L1(h) 7.69e-04 / 7.64e-04 / 2.67e-04
- C2 L1 order 40->80, jax serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, jax serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, jax serial: L1(h) 8.67e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, jax serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, jax serial: L1(h) 8.81e-03 / 2.41e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, jax serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, jax serial: L1(h) 8.32e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 order 50->100, jax serial: L1(h) 1.16e-02 / 8.13e-03 / 5.02e-03
- C12 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 order 50->100, jax serial: L1(h) 3.29e-03 / 7.65e-04 / 1.43e-04
- C12 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), jax serial: |h30-h60| 2.79e-04, |h60-h120| 7.38e-05
- C10 space O2 time O1 order (Richardson), jax serial: |h30-h60| 2.59e-04, |h60-h120| 6.46e-05
- C10 space O2 time O2 order (Richardson), jax serial: |h30-h60| 2.59e-04, |h60-h120| 6.43e-05
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
- C27 ODE time_order 1 order 32->64 steps, jax serial: err 3.25e-02 / 1.60e-02 / 7.96e-03
- C27 ODE time_order 2 order 32->64 steps, jax serial: err 2.48e-04 / 6.95e-05 / 1.83e-05
- C27 ODE time_order 3 order 32->64 steps, jax serial: err 4.12e-05 / 5.15e-06 / 6.43e-07
- C27 wave time_order 1 order dt/2->dt/4, jax serial: err 1.10e-02 / 4.82e-03 / 2.01e-03
- C27 wave time_order 2 order dt/2->dt/4, jax serial: err 4.31e-04 / 1.05e-04 / 2.49e-05
- C27 wave time_order 3 order dt/2->dt/4, jax serial: err 1.86e-05 / 2.32e-06 / 2.86e-07
- C27 ODE control (RK3, wrong stage time) order, jax serial: err 2.49e-02 / 1.26e-02 / 6.33e-03
- C28 1d expl dir0 L_inf order 32->64, jax serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl neu L_inf order 32->64, jax serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl dirL L_inf order 32->64, jax serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl per L_inf order 32->64, jax serial: L_inf 1.40e-03 / 3.52e-04 / 8.81e-05
- C28 1d expl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C28 1d impl dir0 L_inf order 32->64, jax serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl neu L_inf order 32->64, jax serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl dirL L_inf order 32->64, jax serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl per L_inf order 32->64, jax serial: L_inf 3.35e-03 / 8.45e-04 / 2.12e-04
- C28 1d impl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C28 quad expl dir0 L_inf order 32->64, jax serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl neu L_inf order 32->64, jax serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl dirL L_inf order 32->64, jax serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl per L_inf order 32->64, jax serial: L_inf 3.84e-04 / 9.75e-05 / 2.44e-05
- C28 quad expl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C28 quad impl dir0 L_inf order 32->64, jax serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl neu L_inf order 32->64, jax serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl dirL L_inf order 32->64, jax serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl per L_inf order 32->64, jax serial: L_inf 8.16e-04 / 2.06e-04 / 5.17e-05
- C28 quad impl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C28 tri expl dir0 L_inf order 32->64, jax serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl neu L_inf order 32->64, jax serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl dirL L_inf order 32->64, jax serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl per L_inf order 32->64, jax serial: L_inf 4.72e-04 / 1.15e-04 / 2.86e-05
- C28 tri expl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C28 tri impl dir0 L_inf order 32->64, jax serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl neu L_inf order 32->64, jax serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl dirL L_inf order 32->64, jax serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl per L_inf order 32->64, jax serial: L_inf 5.20e-04 / 1.27e-04 / 3.14e-05
- C28 tri impl steady max|u - L|, jax serial: linear Dirichlet field kept to round-off
- C29 p2 dir d1 boundary order 24->48, jax serial: max err 2.01e-02 / 5.32e-03 / 1.35e-03; ghost as sample: 3.10e-02 / 1.21e-02 / 4.85e-03
- C29 p2 dir d2 boundary order 24->48, jax serial: max err 4.95e-02 / 2.42e-02 / 1.27e-02; ghost as sample: 6.45e-02 / 5.33e-02 / 4.96e-02
- C29 p2 dir d1 interior order 24->48, jax serial: max err 4.18e-02 / 1.10e-02 / 2.81e-03; ghost as sample: 4.18e-02 / 1.10e-02 / 2.81e-03
- C29 p2 dir d2 interior order 24->48, jax serial: max err 1.40e-02 / 3.65e-03 / 9.20e-04; ghost as sample: 1.40e-02 / 3.65e-03 / 9.20e-04
- C29 p2 neu d1 boundary order 24->48, jax serial: max err 2.19e-02 / 6.19e-03 / 1.61e-03; ghost as sample: 2.19e-02 / 6.19e-03 / 1.61e-03
- C29 p2 neu d2 boundary order 24->48, jax serial: max err 4.95e-02 / 2.04e-02 / 1.08e-02; ghost as sample: 4.95e-02 / 2.04e-02 / 1.08e-02
- C29 p2 neu d1 interior order 24->48, jax serial: max err 4.18e-02 / 1.10e-02 / 2.81e-03; ghost as sample: 4.18e-02 / 1.10e-02 / 2.81e-03
- C29 p2 neu d2 interior order 24->48, jax serial: max err 1.40e-02 / 3.65e-03 / 9.20e-04; ghost as sample: 1.40e-02 / 3.65e-03 / 9.20e-04
- C29 p3 dir d1 boundary order 24->48, jax serial: max err 1.04e-02 / 1.52e-03 / 1.98e-04; ghost as sample: 2.01e-02 / 7.59e-03 / 3.42e-03
- C29 p3 dir d2 boundary order 24->48, jax serial: max err 3.12e-02 / 8.71e-03 / 2.25e-03; ghost as sample: 4.33e-02 / 4.61e-02 / 4.71e-02
- C29 p3 dir d3 boundary order 24->48, jax serial: max err 6.32e-02 / 2.91e-02 / 1.53e-02; ghost as sample: 3.54e-02 / 9.11e-02 / 2.16e-01
- C29 p3 dir d1 interior order 24->48, jax serial: max err 2.12e-03 / 2.43e-04 / 2.43e-05; ghost as sample: 2.12e-03 / 2.43e-04 / 2.43e-05
- C29 p3 dir d2 interior order 24->48, jax serial: max err 2.07e-02 / 5.39e-03 / 1.42e-03; ghost as sample: 2.07e-02 / 5.39e-03 / 1.42e-03
- C29 p3 dir d3 interior order 24->48, jax serial: max err 1.91e-02 / 8.77e-03 / 3.47e-03; ghost as sample: 1.91e-02 / 8.77e-03 / 3.47e-03
- C29 p3 neu d1 boundary order 24->48, jax serial: max err 1.04e-02 / 1.52e-03 / 1.98e-04; ghost as sample: 1.04e-02 / 1.52e-03 / 1.98e-04
- C29 p3 neu d2 boundary order 24->48, jax serial: max err 3.12e-02 / 8.71e-03 / 2.25e-03; ghost as sample: 3.12e-02 / 8.71e-03 / 2.25e-03
- C29 p3 neu d3 boundary order 24->48, jax serial: max err 5.35e-02 / 2.54e-02 / 1.43e-02; ghost as sample: 5.71e-02 / 2.58e-02 / 1.46e-02
- C29 p3 neu d1 interior order 24->48, jax serial: max err 2.12e-03 / 2.43e-04 / 2.43e-05; ghost as sample: 2.12e-03 / 2.43e-04 / 2.43e-05
- C29 p3 neu d2 interior order 24->48, jax serial: max err 2.07e-02 / 5.39e-03 / 1.42e-03; ghost as sample: 2.07e-02 / 5.39e-03 / 1.42e-03
- C29 p3 neu d3 interior order 24->48, jax serial: max err 1.91e-02 / 8.77e-03 / 3.47e-03; ghost as sample: 1.91e-02 / 8.77e-03 / 3.47e-03
- C30 gate 39 checks failing, jax serial: PASS=72 FAIL=0 (tol 1e-09); the ghost as a plain sample is inexact in 15 cases (sensitivity)
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
- C1 O1 order 40->80, foam serial: L1(h) 1.78e-03 / 1.18e-03 / 7.19e-04
- C12 bump O1 order 20->40, foam serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, foam serial: L1(h) 8.10e-04 / 4.15e-04 / 2.05e-04
- C14 O1 order 40->80, foam serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O1 max rel vs JAX, foam serial: same stencil
- C11 smooth O1 max rel vs JAX, foam serial: same stencil
- C12 bump O1 foam-JAX L1 diff n40, foam serial: 6.7e-16 / 1.1e-15 / 2.3e-15
- C14 O1 max rel vs JAX, foam serial: same stencil
- C1 O2 order 40->80, foam serial: L1(h) 1.15e-03 / 4.45e-04 / 1.11e-04
- C12 bump O2 order 20->40, foam serial: L1(h) 1.11e-03 / 4.70e-04 / 1.24e-04
- C11 smooth O2 order 80->160, foam serial: L1(h) 1.28e-03 / 3.88e-04 / 1.22e-04
- C14 O2 order 40->80, foam serial: L1(h) 7.02e-04 / 7.80e-04 / 2.66e-04
- C1 O2 foam-JAX L1 diff order 40->80, foam serial: 1.8e-03 / 4.9e-04 / 7.6e-05
- C11 smooth O2 foam-JAX L1 diff order 80->160, foam serial: 5.0e-04 / 9.1e-05 / 1.8e-05
- C12 bump O2 foam-JAX L1 diff order 20->40, foam serial: 6.4e-03 / 3.0e-03 / 1.1e-03
- C14 O2 foam-JAX L1 diff order 40->80, foam serial: 7.2e-04 / 2.5e-04 / 1.1e-04
- C2 L1 order 40->80, foam serial: L1(h) 2.11e-03 / 1.14e-03
- C4 sub O1 order 50->100, foam serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O1 max rel vs JAX, foam serial: same stencil
- C4 sub O2 order 50->100, foam serial: L1(h) 8.67e-03 / 2.73e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 foam-JAX L1 diff order 50->100, foam serial: 8.8e-06 / 2.4e-06 / 1.2e-07
- C4 trans O1 order 50->100, foam serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 max rel vs JAX, foam serial: same stencil
- C4 trans O2 order 50->100, foam serial: L1(h) 8.81e-03 / 2.41e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 foam-JAX L1 diff n100, foam serial: 2.5e-06 / 6.0e-16 / 2.1e-13
- C11 sub O1 order 50->100, foam serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 max rel vs JAX, foam serial: same stencil
- C11 sub O2 order 50->100, foam serial: L1(h) 8.33e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 foam-JAX L1 diff order 50->100, foam serial: 1.2e-05 / 1.7e-06 / 6.0e-07
- C12 sub O1 order 50->100, foam serial: L1(h) 1.16e-02 / 8.13e-03 / 5.02e-03
- C12 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 foam-JAX L1 diff n100, foam serial: 1.0e-15 / 1.7e-15 / 3.2e-15
- C12 sub O2 order 50->100, foam serial: L1(h) 3.18e-03 / 7.85e-04 / 1.16e-04
- C12 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 foam-JAX L1 diff order 50->100, foam serial: 1.2e-03 / 5.2e-04 / 1.2e-04
- C10 space O1 time O1 order (Richardson), foam serial: |h30-h60| 2.79e-04, |h60-h120| 7.38e-05
- C10 space O2 time O1 order (Richardson), foam serial: |h30-h60| 2.60e-04, |h60-h120| 6.51e-05
- C10 space O2 time O2 order (Richardson), foam serial: |h30-h60| 2.60e-04, |h60-h120| 6.48e-05
- C13 O1 max rel vs JAX, foam serial: O1: the same stencil
- C18 Sod O1 order 50->100, foam serial: L1(rho) 3.03e-02 / 2.09e-02
- C18 Sod O2 order 50->100, foam serial: L1(rho) 9.59e-03 / 5.13e-03
- C19 wave o1 order 16->24, foam serial: L1(rho) 6.99e-02 / 4.38e-02 / 3.10e-02
- C19 wave o1 max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C19 wave o2none order 16->24, foam serial: L1(rho) 1.58e-02 / 2.68e-03 / 1.01e-03
- C19 wave o2none max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C19 wave o2minmod order 16->24, foam serial: L1(rho) 1.98e-02 / 3.88e-03 / 1.82e-03
- C19 wave o2minmod max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C24 Sod O1 order 50->100, foam serial: L1(rho) 2.98e-02 / 2.05e-02
- C24 Sod O2 order 50->100, foam serial: L1(rho) 9.62e-03 / 5.14e-03
- C24 wave o1 order 32->64, foam serial: L1(rho) 3.03e-02 / 1.60e-02 / 8.25e-03
- C24 wave o1 max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C24 wave o2none order 32->64, foam serial: L1(rho) 2.02e-03 / 4.31e-04 / 1.02e-04
- C24 wave o2none max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C24 wave o2minmod order 32->64, foam serial: L1(rho) 3.39e-03 / 9.15e-04 / 2.47e-04
- C24 wave o2minmod max|u - u0|, |p - 1|, foam serial: contact: velocity and pressure constant
- C23 O1 n8 max rel vs JAX, foam serial: same stencil
- C23 O1 n16 max rel vs JAX, foam serial: same stencil
- C23 O1 n24 max rel vs JAX, foam serial: same stencil
- C23 O2 L1|backend - JAX| order 16->24, foam serial: 7.44e-05 / 3.33e-05
- C27 ODE time_order 1 order 32->64 steps, foam serial: err 3.25e-02 / 1.60e-02 / 7.96e-03
- C27 ODE time_order 2 order 32->64 steps, foam serial: err 2.48e-04 / 6.95e-05 / 1.83e-05
- C27 ODE time_order 3 order 32->64 steps, foam serial: err 4.12e-05 / 5.15e-06 / 6.43e-07
- C27 wave time_order 1 order dt/2->dt/4, foam serial: err 1.10e-02 / 4.82e-03 / 2.01e-03
- C27 wave time_order 2 order dt/2->dt/4, foam serial: err 4.31e-04 / 1.05e-04 / 2.49e-05
- C27 wave time_order 3 order dt/2->dt/4, foam serial: err 1.86e-05 / 2.32e-06 / 2.86e-07
- C28 1d expl dir0 max rel vs JAX, foam serial: L_inf vs exact 3.86e-04
- C28 1d expl neu max rel vs JAX, foam serial: L_inf vs exact 3.86e-04
- C28 1d expl dirL max rel vs JAX, foam serial: L_inf vs exact 3.86e-04
- C28 1d expl per max rel vs JAX, foam serial: L_inf vs exact 1.40e-03
- C28 1d expl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 1d expl steady max|u - L|, foam serial: on the backend itself
- C28 1d impl dir0 max rel vs JAX, foam serial: L_inf vs exact 9.54e-04
- C28 1d impl neu max rel vs JAX, foam serial: L_inf vs exact 9.54e-04
- C28 1d impl dirL max rel vs JAX, foam serial: L_inf vs exact 9.54e-04
- C28 1d impl per max rel vs JAX, foam serial: L_inf vs exact 3.35e-03
- C28 1d impl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 1d impl steady max|u - L|, foam serial: on the backend itself
- C28 quad expl dir0 max rel vs JAX, foam serial: L_inf vs exact 4.70e-04
- C28 quad expl neu max rel vs JAX, foam serial: L_inf vs exact 4.70e-04
- C28 quad expl dirL max rel vs JAX, foam serial: L_inf vs exact 4.70e-04
- C28 quad expl per max rel vs JAX, foam serial: L_inf vs exact 3.84e-04
- C28 quad expl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 quad expl steady max|u - L|, foam serial: on the backend itself
- C28 quad impl dir0 max rel vs JAX, foam serial: L_inf vs exact 1.13e-03
- C28 quad impl neu max rel vs JAX, foam serial: L_inf vs exact 1.13e-03
- C28 quad impl dirL max rel vs JAX, foam serial: L_inf vs exact 1.13e-03
- C28 quad impl per max rel vs JAX, foam serial: L_inf vs exact 8.16e-04
- C28 quad impl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 quad impl steady max|u - L|, foam serial: on the backend itself
- C28 tri expl dir0 max rel vs JAX, foam serial: L_inf vs exact 8.15e-04
- C28 tri expl neu max rel vs JAX, foam serial: L_inf vs exact 8.15e-04
- C28 tri expl dirL max rel vs JAX, foam serial: L_inf vs exact 8.15e-04
- C28 tri expl per max rel vs JAX, foam serial: L_inf vs exact 4.72e-04
- C28 tri expl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 tri expl steady max|u - L|, foam serial: on the backend itself
- C28 tri impl dir0 max rel vs JAX, foam serial: L_inf vs exact 8.25e-04
- C28 tri impl neu max rel vs JAX, foam serial: L_inf vs exact 8.25e-04
- C28 tri impl dirL max rel vs JAX, foam serial: L_inf vs exact 8.25e-04
- C28 tri impl per max rel vs JAX, foam serial: L_inf vs exact 5.20e-04
- C28 tri impl steady max rel vs JAX, foam serial: L_inf vs exact 0.00e+00
- C28 tri impl steady max|u - L|, foam serial: on the backend itself
- C6 C1 mass change (rel), foam serial: max over 6 runs
- C6 C1 mass change (rel), foam mpi2: max over 1 runs
- C6 C1 mass change (rel), foam mpi4: max over 1 runs
- C6 C2 mass change (rel), foam serial: max over 2 runs
- C6 C3 mass change (rel), foam serial: max over 2 runs
- C6 C5 mass change (rel), foam serial: max over 1 runs
- C6 C5 mass change (rel), foam mpi2: max over 1 runs
- C6 C5 mass change (rel), foam mpi4: max over 1 runs
- C6 C11 mass change (rel), foam serial: max over 6 runs
- C6 C12 mass change (rel), foam serial: max over 6 runs
- C6 C13 mass change (rel), foam serial: max over 4 runs
- C6 C13 mass change (rel), foam mpi2: max over 4 runs
- C6 C13 mass change (rel), foam mpi4: max over 4 runs
- C6 C14 mass change (rel), foam serial: max over 6 runs
- C6 C14 mass change (rel), foam mpi2: max over 1 runs
- C6 C14 mass change (rel), foam mpi4: max over 1 runs
- C6 C20 mass change (rel), foam serial: max over 2 runs
- C6 C23 mass change (rel), foam serial: max over 6 runs
- C1 O1 order 40->80, amrex serial: L1(h) 1.78e-03 / 1.18e-03 / 7.19e-04
- C11 smooth O1 order 80->160, amrex serial: L1(h) 8.10e-04 / 4.15e-04 / 2.05e-04
- C14 O1 order 40->80, amrex serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, amrex serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C11 smooth O2 order 80->160, amrex serial: L1(h) 1.34e-03 / 3.97e-04 / 1.23e-04
- C14 O2 order 40->80, amrex serial: L1(h) 7.69e-04 / 7.64e-04 / 2.67e-04
- C2 L1 order 40->80, amrex serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, amrex serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, amrex serial: L1(h) 8.67e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, amrex serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, amrex serial: L1(h) 8.81e-03 / 2.41e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, amrex serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, amrex serial: L1(h) 8.32e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), amrex serial: |h30-h60| 2.79e-04, |h60-h120| 7.38e-05
- C10 space O2 time O1 order (Richardson), amrex serial: |h30-h60| 2.59e-04, |h60-h120| 6.46e-05
- C10 space O2 time O2 order (Richardson), amrex serial: |h30-h60| 2.59e-04, |h60-h120| 6.43e-05
- C18 Sod O1 order 50->100, amrex serial: L1(rho) 3.03e-02 / 2.09e-02
- C18 Sod O2 order 50->100, amrex serial: L1(rho) 9.59e-03 / 5.13e-03
- C19 wave o1 order 16->24, amrex serial: L1(rho) 6.99e-02 / 4.38e-02 / 3.10e-02
- C19 wave o1 max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C19 wave o2none order 16->24, amrex serial: L1(rho) 3.88e-02 / 7.25e-03 / 2.35e-03
- C19 wave o2none max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C19 wave o2minmod order 16->24, amrex serial: L1(rho) 4.02e-02 / 7.45e-03 / 2.41e-03
- C19 wave o2minmod max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C24 Sod O1 order 50->100, amrex serial: L1(rho) 2.98e-02 / 2.05e-02
- C24 Sod O2 order 50->100, amrex serial: L1(rho) 9.62e-03 / 5.14e-03
- C24 wave o1 order 32->64, amrex serial: L1(rho) 3.03e-02 / 1.60e-02 / 8.25e-03
- C24 wave o1 max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C24 wave o2none order 32->64, amrex serial: L1(rho) 3.41e-03 / 5.59e-04 / 1.12e-04
- C24 wave o2none max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C24 wave o2minmod order 32->64, amrex serial: L1(rho) 3.60e-03 / 8.58e-04 / 2.43e-04
- C24 wave o2minmod max|u - u0|, |p - 1|, amrex serial: contact: velocity and pressure constant
- C23 O1 n8 max rel vs JAX, amrex serial: same stencil
- C23 O1 n16 max rel vs JAX, amrex serial: same stencil
- C23 O1 n24 max rel vs JAX, amrex serial: same stencil
- C27 ODE time_order 1 order 32->64 steps, amrex serial: err 3.25e-02 / 1.60e-02 / 7.96e-03
- C27 ODE time_order 2 order 32->64 steps, amrex serial: err 2.48e-04 / 6.95e-05 / 1.83e-05
- C27 ODE time_order 3 order 32->64 steps, amrex serial: err 4.12e-05 / 5.15e-06 / 6.43e-07
- C27 wave time_order 1 order dt/2->dt/4, amrex serial: err 1.10e-02 / 4.82e-03 / 2.01e-03
- C27 wave time_order 2 order dt/2->dt/4, amrex serial: err 4.31e-04 / 1.05e-04 / 2.49e-05
- C27 wave time_order 3 order dt/2->dt/4, amrex serial: err 1.86e-05 / 2.32e-06 / 2.86e-07
- C28 1d expl dir0 max rel vs JAX, amrex serial: L_inf vs exact 3.86e-04
- C28 1d expl neu max rel vs JAX, amrex serial: L_inf vs exact 3.86e-04
- C28 1d expl dirL max rel vs JAX, amrex serial: L_inf vs exact 3.86e-04
- C28 1d expl per max rel vs JAX, amrex serial: L_inf vs exact 1.40e-03
- C28 1d expl steady max rel vs JAX, amrex serial: L_inf vs exact 0.00e+00
- C28 1d expl steady max|u - L|, amrex serial: on the backend itself
- C28 1d impl dir0 max rel vs JAX, amrex serial: L_inf vs exact 9.54e-04
- C28 1d impl neu max rel vs JAX, amrex serial: L_inf vs exact 9.54e-04
- C28 1d impl dirL max rel vs JAX, amrex serial: L_inf vs exact 9.54e-04
- C28 1d impl per max rel vs JAX, amrex serial: L_inf vs exact 3.35e-03
- C28 1d impl steady max rel vs JAX, amrex serial: L_inf vs exact 0.00e+00
- C28 1d impl steady max|u - L|, amrex serial: on the backend itself
- C28 quad expl dir0 max rel vs JAX, amrex serial: L_inf vs exact 4.70e-04
- C28 quad expl neu max rel vs JAX, amrex serial: L_inf vs exact 4.70e-04
- C28 quad expl dirL max rel vs JAX, amrex serial: L_inf vs exact 4.70e-04
- C28 quad expl per max rel vs JAX, amrex serial: L_inf vs exact 3.84e-04
- C28 quad expl steady max rel vs JAX, amrex serial: L_inf vs exact 0.00e+00
- C28 quad expl steady max|u - L|, amrex serial: on the backend itself
- C28 quad impl dir0 max rel vs JAX, amrex serial: L_inf vs exact 1.13e-03
- C28 quad impl neu max rel vs JAX, amrex serial: L_inf vs exact 1.13e-03
- C28 quad impl dirL max rel vs JAX, amrex serial: L_inf vs exact 1.13e-03
- C28 quad impl per max rel vs JAX, amrex serial: L_inf vs exact 8.16e-04
- C28 quad impl steady max rel vs JAX, amrex serial: L_inf vs exact 0.00e+00
- C28 quad impl steady max|u - L|, amrex serial: on the backend itself
- C17 boxnr mass drift, amrex serial: no reflux: not conservative
- C17 boxnr energy drift, amrex serial: no reflux: not conservative
- C17 boxnr vs CPU 1 rank (max rel), amrex mpi2: no reflux: not conservative
- C17 boxnr vs CPU 1 rank (max rel), amrex gpu1: no reflux: not conservative
- C6 C1 mass change (rel), amrex serial: max over 6 runs
- C6 C1 mass change (rel), amrex mpi2: max over 1 runs
- C6 C1 mass change (rel), amrex mpi4: max over 1 runs
- C6 C1 mass change (rel), amrex gpu1: max over 1 runs
- C6 C2 mass change (rel), amrex serial: max over 2 runs
- C6 C3 mass change (rel), amrex serial: max over 2 runs
- C6 C5 mass change (rel), amrex serial: max over 1 runs
- C6 C5 mass change (rel), amrex mpi2: max over 1 runs
- C6 C5 mass change (rel), amrex mpi4: max over 1 runs
- C6 C5 mass change (rel), amrex gpu1: max over 1 runs
- C6 C5 mass change (rel), amrex gpu2: max over 1 runs
- C6 C11 mass change (rel), amrex serial: max over 6 runs
- C6 C13 mass change (rel), amrex serial: max over 4 runs
- C6 C13 mass change (rel), amrex mpi2: max over 4 runs
- C6 C13 mass change (rel), amrex mpi4: max over 4 runs
- C6 C14 mass change (rel), amrex serial: max over 6 runs
- C6 C14 mass change (rel), amrex mpi2: max over 1 runs
- C6 C14 mass change (rel), amrex mpi4: max over 1 runs
- C6 C14 mass change (rel), amrex gpu1: max over 1 runs
- C6 C14 mass change (rel), amrex gpu2: max over 1 runs
- C6 C20 mass change (rel), amrex serial: max over 2 runs
- C6 C23 mass change (rel), amrex serial: max over 6 runs

Summed run wall time (s, without process start-up): amrex 4750, foam 5840, jax 7369
