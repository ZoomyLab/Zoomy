| case | check | backend | serial | mpi2 | mpi4 | gpu1 | gpu-mpi | threshold (source) |
|---|---|---|---|---|---|---|---|---|
| C1 | O1 max rel vs JAX | amrex | 8.16e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | amrex | 6.88e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 max rel vs JAX | amrex | 9.51e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O2 order 40->80 | amrex | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C1 | O1 max rel vs JAX | foam | 8.60e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | foam | 6.88e-01 PASS | - | - | n/a | n/a | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 foam-JAX L1 diff order 40->80 | foam | 2.68e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C1 | O2 order 40->80 | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C1 | O1 order 40->80 | jax | 6.88e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 order 40->80 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C2 | L1 order 40->80 | amrex | 9.23e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C2 | max rel vs JAX | amrex | 1.81e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C2 | L1 order 40->80 | foam | 9.22e-01 PASS | - | - | n/a | n/a | >= 0.6 (spec) |
| C2 | L1 vs JAX n80 (rel) | foam | 2.82e-04 PASS | - | - | n/a | n/a | <= 0.0015 (observed (foam 7.1e-4)) |
| C2 | L1 order 40->80 | jax | 9.23e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C3 | O1 max|u| after 100 steps | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | amrex | 5.23e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | max rel vs JAX | amrex | 5.70e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C3 | O1 max|u| after 100 steps | foam | 4.90e-16 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | foam | 1.25e-15 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O1 max|u| after 100 steps | jax | 2.56e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | jax | 7.28e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 max rel vs JAX | amrex | 1.39e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | amrex | 1.38e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | amrex | 3.35e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | amrex | 5.60e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max rel vs JAX | amrex | 1.20e-13 PASS | - | - | - | - | <= 2.61261e-12 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 2.6e-13)) |
| C4 | sub O2 max|q/q_in - 1| n100 | amrex | 2.95e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | amrex | 7.23e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | amrex | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | amrex | 9.64e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | amrex | 5.36e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | amrex | 1.49e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | amrex | 6.95e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max rel vs JAX | amrex | 1.98e-13 PASS | - | - | - | - | <= 1.50174e-10 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 1.5e-11)) |
| C4 | trans O2 max|q/q_in - 1| n100 | amrex | 1.98e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | amrex | 3.75e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | amrex | 2.88e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | sub O1 max rel vs JAX | foam | 1.19e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | foam | 1.38e-02 | - | - | n/a | n/a | information |
| C4 | sub O1 max|q_y| n100 | foam | 4.17e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | foam | 5.60e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 foam-JAX L1 diff order 50->100 | foam | 4.16e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C4 | sub O2 max|q/q_in - 1| n100 | foam | 2.95e-03 | - | - | n/a | n/a | information |
| C4 | sub O2 max|q_y| n100 | foam | 8.42e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | foam | 2.23e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | foam | 6.89e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | foam | 5.36e-02 | - | - | n/a | n/a | information |
| C4 | trans O1 max|q_y| n100 | foam | 1.75e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | foam | 6.95e-01 PASS | - | - | n/a | n/a | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 foam-JAX L1 diff n100 | foam | 8.64e-14 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C4 | trans O2 max|q/q_in - 1| n100 | foam | 1.98e-03 | - | - | n/a | n/a | information |
| C4 | trans O2 max|q_y| n100 | foam | 2.94e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | foam | 2.88e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | sub O1 max|q/q_in - 1| n100 | jax | 1.38e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | jax | 3.56e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | jax | 5.60e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max|q/q_in - 1| n100 | jax | 2.95e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | jax | 9.02e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | jax | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max|q/q_in - 1| n100 | jax | 5.36e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | jax | 1.26e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | jax | 6.95e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max|q/q_in - 1| n100 | jax | 1.98e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | jax | 3.34e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | jax | 2.88e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C5 | h_min | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C5 | max rel vs JAX | amrex | 6.47e-15 PASS | - | - | - | - | <= 1e-09 (observed 6e-10 (wet/dry eigenvalue cluster)) |
| C5 | L1 rel vs JAX | foam | 1.70e-02 PASS | - | - | n/a | n/a | <= 0.035 (observed (foam 1.7e-2)) |
| C5 | h_min | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | >= 0 (theory (positivity)) |
| C5 | h_min | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C6 | C1 mass change (rel) | amrex | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | amrex | 3.47e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | amrex | 1.73e-16 PASS | 1.73e-16 PASS | 1.73e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | amrex | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | amrex | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | amrex | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | amrex | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | foam | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | foam | 3.47e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | foam | 1.41e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | foam | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | foam | 2.20e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | foam | 2.22e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | foam | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | jax | 1.41e-16 PASS | 0.00e+00 PASS | 1.41e-16 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | jax | 3.47e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | jax | 1.73e-16 PASS | 1.73e-16 PASS | 1.73e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | jax | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | jax | 2.82e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | jax | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | jax | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | jax | 1.95e-16 PASS | 1.95e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 1.95e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | amrex | - | 1.55e-21 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | amrex | - | 1.58e-14 PASS | 2.38e-14 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | foam | - | 9.42e-16 PASS | 1.13e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | foam | - | 5.30e-16 PASS | 6.36e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | foam | - | 6.72e-15 PASS | 7.68e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | foam | - | 3.18e-16 PASS | 3.18e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | foam | - | 4.31e-16 PASS | 5.74e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | foam | - | 7.99e-16 PASS | 9.33e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | foam | - | 1.70e-14 PASS | 1.60e-14 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | foam | - | 2.09e-15 PASS | 2.27e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | foam | - | 2.34e-14 PASS | 2.13e-14 PASS | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | jax | - | 1.22e-13 PASS | 1.22e-13 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | jax | - | 1.32e-15 PASS | 1.32e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | jax | - | 5.00e-16 PASS | 5.00e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | jax | - | 3.18e-16 PASS | 3.18e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | jax | - | 5.84e-16 PASS | 5.84e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | jax | - | 1.78e-15 PASS | 1.15e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | jax | - | 1.83e-15 PASS | 1.94e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | jax | - | 3.62e-15 PASS | 3.39e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | jax | - | 8.73e-10 PASS | 8.67e-10 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | amrex | 8.35e-10 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam max rel vs JAX | amrex | 8.46e-10 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_newton max rel vs JAX | amrex | 1.13e-10 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_t2 max rel vs JAX | amrex | 1.13e-10 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | foam | 8.35e-10 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam L1 rel vs JAX | foam | 6.08e-10 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_newton L1 rel vs JAX | foam | 1.37e-10 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_t2 L1 rel vs JAX | foam | 8.68e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_tri L1 rel vs JAX | foam | 2.98e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | Newton vs linear pressure (max rel) | jax | 3.66e-10 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 8.18e-15 PASS | 8.18e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | amrex | - | - | - | 1.91e-14 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 1.67e-13 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 5.84e-16 PASS | 5.32e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | jax | - | - | - | 1.12e-15 PASS | 1.02e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | jax | - | - | - | 3.14e-15 PASS | 3.10e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | jax | - | - | - | 2.79e-10 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C10 | space O1 time O1 order (Richardson) | amrex | 1.91e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | amrex | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | amrex | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | foam | 1.91e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | foam | 2.01e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | jax | 1.91e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | jax | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | jax | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C11 | smooth O1 max rel vs JAX | amrex | 1.33e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | amrex | 9.57e-01 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 max rel vs JAX | amrex | 1.35e-13 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O2 order 80->160 | amrex | 1.81e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | amrex | 1.18e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | amrex | 1.38e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | amrex | 5.65e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max rel vs JAX | amrex | 1.75e-07 PASS | - | - | - | - | <= 4.88043e-06 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 4.9e-07)) |
| C11 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | amrex | 2.24e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C11 | smooth O1 max rel vs JAX | foam | 1.61e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | foam | 9.57e-01 PASS | - | - | n/a | n/a | >= 0.9 (theory 1) |
| C11 | smooth O2 foam-JAX L1 diff order 80->160 | foam | 2.81e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | smooth O2 order 80->160 | foam | 1.78e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | foam | 9.91e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | foam | 1.38e-02 | - | - | n/a | n/a | information |
| C11 | sub O1 order 50->100 | foam | 5.65e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 foam-JAX L1 diff order 50->100 | foam | 3.86e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | sub O2 max|q/q_in - 1| n100 | foam | 2.96e-03 | - | - | n/a | n/a | information |
| C11 | sub O2 order 50->100 | foam | 2.24e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C11 | smooth O1 order 80->160 | jax | 9.57e-01 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 order 80->160 | jax | 1.81e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max|q/q_in - 1| n100 | jax | 1.38e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | jax | 5.65e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | jax | 2.24e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C12 | runs | amrex | n/a | - | - | - | - | information |
| C12 | bump O1 foam-JAX L1 diff n40 | foam | 2.28e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C12 | bump O1 order 20->40 | foam | 6.33e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 foam-JAX L1 diff order 20->40 | foam | 1.47e+00 PASS | - | - | n/a | n/a | >= 0.25 (observed 0.33-0.39 (O1 triangles: both converge slowly)) |
| C12 | bump O2 order 20->40 | foam | 1.90e+00 PASS | - | - | n/a | n/a | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 foam-JAX L1 diff n100 | foam | 3.08e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C12 | sub O1 max|q/q_in - 1| n100 | foam | 9.56e-03 | - | - | n/a | n/a | information |
| C12 | sub O1 max|q_y| n100 | foam | 6.27e-03 | - | - | n/a | n/a | information |
| C12 | sub O1 order 50->100 | foam | 6.80e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 foam-JAX L1 diff order 50->100 | foam | 2.13e+00 PASS | - | - | n/a | n/a | >= 0.25 (observed 0.33-0.39 (O1 triangles: both converge slowly)) |
| C12 | sub O2 max|q/q_in - 1| n100 | foam | 1.67e-03 | - | - | n/a | n/a | information |
| C12 | sub O2 max|q_y| n100 | foam | 4.48e-03 | - | - | n/a | n/a | information |
| C12 | sub O2 order 50->100 | foam | 2.75e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C12 | bump O1 order 20->40 | jax | 6.33e-01 PASS | - | - | - | - | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 order 20->40 | jax | 1.70e+00 PASS | - | - | - | - | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 max|q/q_in - 1| n100 | jax | 9.56e-03 | - | - | - | - | information |
| C12 | sub O1 max|q_y| n100 | jax | 6.27e-03 | - | - | - | - | information |
| C12 | sub O1 order 50->100 | jax | 6.80e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 max|q/q_in - 1| n100 | jax | 1.51e-03 | - | - | - | - | information |
| C12 | sub O2 max|q_y| n100 | jax | 3.19e-03 | - | - | - | - | information |
| C12 | sub O2 order 50->100 | jax | 2.41e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C13 | O1 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | max rel vs JAX | amrex | 1.17e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 max rel vs JAX | foam | 7.46e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | foam | 2.13e-16 PASS | 1.07e-16 PASS | 1.07e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | foam | 4.24e-16 PASS | 4.77e-16 PASS | 4.77e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | jax | 3.73e-16 PASS | 2.13e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | jax | 1.17e-15 PASS | 2.12e-16 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C14 | O1 max rel vs JAX | amrex | 1.14e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | amrex | 5.94e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 max rel vs JAX | amrex | 1.27e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O2 order 40->80 | amrex | 1.49e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 max rel vs JAX | foam | 1.13e-12 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | foam | 5.94e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 foam-JAX L1 diff order 40->80 | foam | 1.12e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C14 | O2 order 40->80 | foam | 1.53e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 order 40->80 | jax | 5.94e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 order 40->80 | jax | 1.49e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C15 | runs | amrex | n/a | - | - | - | - | information |
| C15 | runs | foam | n/a | - | - | n/a | n/a | information |
| C15 | chorin jvp vs serial (rel) | jax | - | 5.36e-12 PASS | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C15 | chorin |FD - jvp| (rel) | jax | 6.39e-09 PASS | 6.40e-09 PASS | - | - | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | chorin |rev - jvp| (rel) | jax | 8.87e-15 PASS | 4.07e-11 PASS | - | - | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | hyperbolic jvp vs serial (rel) | jax | - | 5.55e-16 PASS | - | 1.85e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | hyperbolic |FD - jvp| (rel) | jax | 7.88e-09 PASS | 1.39e-08 PASS | - | 7.88e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | hyperbolic |rev - jvp| (rel) | jax | 7.77e-15 PASS | 1.65e-14 PASS | - | 1.85e-15 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | imex jvp vs serial (rel) | jax | - | 1.75e-15 PASS | - | 2.10e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | imex |FD - jvp| (rel) | jax | 3.66e-07 PASS | 1.93e-07 PASS | - | 1.36e-07 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | imex |rev - jvp| (rel) | jax | 5.79e-15 PASS | 1.49e-14 PASS | - | - | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C16 | amrex_amrex coupled vs single (max) | amrex | 3.39e-15 PASS | 3.39e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_amrex mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax coupled vs single (max) | amrex | 3.05e-15 PASS | 3.05e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam coupled vs single (max) | foam | 4.72e-15 PASS | 4.72e-15 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam mass drift | foam | 0.00e+00 PASS | 0.00e+00 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam coupled vs single (max) | jax | 4.27e-15 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam mass drift | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam runs | jax | - | unsupported | - | - | - | information |
| C16 | jax_jax coupled vs single (max) | jax | 3.05e-15 PASS | 3.05e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
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
| C17 | lake max|u| | amrex | 1.25e-15 PASS | - | - | - | - | <= 1e-12 (theory (well-balanced)) |
| C17 | lake vs CPU 1 rank (max rel) | amrex | - | - | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
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
| C26 | max rel vs JAX | amrex | 1.05e-13 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C26 | L1 rel vs JAX | foam | 1.72e-03 | - | - | n/a | n/a | information |
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
| C31 | swe_ssp2_k0 max rel vs JAX | amrex | 2.40e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C31 | vam_ssp2_k0 max rel vs JAX | amrex | 7.23e-12 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C31 | swe_ssp2_k0 max rel vs JAX | foam | 2.11e-14 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C31 | SWE lie1 order dt/4->dt/8 | jax | 1.22e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C31 | SWE lie2 order dt/4->dt/8 | jax | 1.22e+00 PASS | - | - | - | - | <= 1.4 (theory 1: Lie splitting caps SSP-RK2 (observed 1.22, the dt/32 reference lifts the last pair)) |
| C31 | SWE ssp2 order dt/4->dt/8 | jax | 2.08e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C31 | VAM predictor alone ssp2 order (Richardson) | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C31 | VAM ssp2 order (Richardson) | jax | 9.09e-01 PASS | - | - | - | - | >= 0.7 (observed 0.91 (the approximate projection caps it, README; the predictor alone: 2)) |
| C31 | VAM t1 order (Richardson) | jax | 9.80e-01 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C31 | VAM t2 order (Richardson) | jax | 6.69e-01 PASS | - | - | - | - | >= 0.5 (observed 0.67 (theory 1: Lie source; the approximate projection, README)) |
| C32 | free n20 L_inf vs exact | amrex | 3.32e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 max rel vs JAX | amrex | 3.33e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C32 | free n20 max|h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 L_inf vs exact | amrex | 3.33e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 max rel vs JAX | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C32 | noslip n20 max|h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 L_inf vs exact | foam | 3.33e-04 PASS | - | - | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 L_inf|foam - JAX| / L_inf(JAX - exact) | foam | 2.07e-01 PASS | - | - | n/a | n/a | <= 0.5 (observed 0.21 (foam's own stencils at second order)) |
| C32 | free n20 max|h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | foam | 6.40e-17 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | foam | 5.70e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 L_inf vs exact | foam | 3.34e-04 PASS | - | - | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 L_inf|foam - JAX| / L_inf(JAX - exact) | foam | 2.29e-02 PASS | - | - | n/a | n/a | <= 0.5 (observed 0.21 (foam's own stencils at second order)) |
| C32 | noslip n20 max|h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | foam | 1.15e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | foam | 5.70e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | control: no-slip profile on free-slip walls, L_inf vs the no-slip exact | jax | 4.17e-01 PASS | - | - | - | - | >= 0.1 (observed 0.42 (a wall that acts: the no-slip profile is not a free-slip solution)) |
| C32 | free L_inf order 40->80 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C32 | free n20 L_inf vs exact | jax | 3.32e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 max|h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | jax | 1.42e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip L_inf order 40->80 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C32 | noslip n20 L_inf vs exact | jax | 3.33e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 max|h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | jax | 1.42e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction) | jax | 1.69e-04 PASS | - | - | - | - | <= 0.0004 (observed 1.7e-4 (a regression bound; 2.5e-3 at g = 9.81)) |
| C33 | imex max rel vs JAX | amrex | 1.30e-12 PASS | - | - | - | - | <= 1e-10 (tolerance (Newton 1e-12; observed 1.3e-12)) |
| C33 | imex q(T) vs expm at dt 0.0125 (rel) | amrex | 5.11e-04 PASS | - | - | - | - | <= 0.001 (observed 5.1e-4 (implicit Euler, theory O(dt))) |
| C33 | imex q(T) vs expm, order dt/2 -> dt/4 | amrex | 9.95e-01 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C33 | imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel) | amrex | 2.98e-12 PASS | - | - | - | - | <= 1e-10 (tolerance (Newton 1e-12; observed 3e-12 C++, 4e-15 JAX)) |
| C33 | imex uniform state: max|Q - Q[:, 0]|, |h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 max rel vs JAX | amrex | 1.11e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 q(T) vs expm at dt 0.0125 (rel) | amrex | 3.34e-06 PASS | - | - | - | - | <= 7e-06 (observed 3.3e-6 (SSP-RK2, theory O(dt^2))) |
| C33 | rk2 q(T) vs expm, order dt/2 -> dt/4 | amrex | 2.03e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C33 | rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel) | amrex | 2.79e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 uniform state: max|Q - Q[:, 0]|, |h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | imex max rel vs JAX | foam | 1.30e-12 PASS | - | - | n/a | n/a | <= 1e-10 (tolerance (Newton 1e-12; observed 1.3e-12)) |
| C33 | imex q(T) vs expm at dt 0.0125 (rel) | foam | 5.11e-04 PASS | - | - | n/a | n/a | <= 0.001 (observed 5.1e-4 (implicit Euler, theory O(dt))) |
| C33 | imex q(T) vs expm, order dt/2 -> dt/4 | foam | 9.95e-01 PASS | - | - | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C33 | imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel) | foam | 2.98e-12 PASS | - | - | n/a | n/a | <= 1e-10 (tolerance (Newton 1e-12; observed 3e-12 C++, 4e-15 JAX)) |
| C33 | imex uniform state: max|Q - Q[:, 0]|, |h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 max rel vs JAX | foam | 1.11e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 q(T) vs expm at dt 0.0125 (rel) | foam | 3.34e-06 PASS | - | - | n/a | n/a | <= 7e-06 (observed 3.3e-6 (SSP-RK2, theory O(dt^2))) |
| C33 | rk2 q(T) vs expm, order dt/2 -> dt/4 | foam | 2.03e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C33 | rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel) | foam | 2.79e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 uniform state: max|Q - Q[:, 0]|, |h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | imex q(T) vs expm at dt 0.0125 (rel) | jax | 5.11e-04 PASS | - | - | - | - | <= 0.001 (observed 5.1e-4 (implicit Euler, theory O(dt))) |
| C33 | imex q(T) vs expm, order dt/2 -> dt/4 | jax | 9.95e-01 PASS | - | - | - | - | >= 0.8 (theory 1, slack 0.2) |
| C33 | imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel) | jax | 4.69e-15 PASS | - | - | - | - | <= 1e-10 (tolerance (Newton 1e-12; observed 3e-12 C++, 4e-15 JAX)) |
| C33 | imex uniform state: max|Q - Q[:, 0]|, |h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 q(T) vs expm at dt 0.0125 (rel) | jax | 3.34e-06 PASS | - | - | - | - | <= 7e-06 (observed 3.3e-6 (SSP-RK2, theory O(dt^2))) |
| C33 | rk2 q(T) vs expm, order dt/2 -> dt/4 | jax | 2.03e+00 PASS | - | - | - | - | >= 1.8 (theory 2, slack 0.2) |
| C33 | rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel) | jax | 2.79e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 uniform state: max|Q - Q[:, 0]|, |h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |

## Large tier

| level | run | result |
|---|---|---|
| L1 | amrex_gpu1 | cells 40000, setup 54.3, ms_step 12.94, maxrss_gb 0.9062, ranks 1 |
| L1 | amrex_gpu2 | cells 80089, setup 58.28, ms_step 14.68, maxrss_gb 0.9062, ranks 2 |
| L1 | amrex_gpu4 | cells 160000, setup 56.15, ms_step 13.48, maxrss_gb 0.9062, ranks 4 |
| L1 | amrex_mpi2 | cells 80089, setup 27.62, ms_step 897.9, maxrss_gb 0.4977, ranks 2 |
| L1 | amrex_mpi4 | cells 160000, setup 26.67, ms_step 808.7, maxrss_gb 0.6206, ranks 4 |
| L1 | amrex_serial | cells 40000, setup 20.76, ms_step 845, maxrss_gb 0.4975, ranks 1 |
| L1 | foam_mpi2 | cells 80089, setup 46.79, ms_step 1380, maxrss_gb 0.9041, ranks 2 |
| L1 | foam_mpi4 | cells 160000, setup 53, ms_step 1380, maxrss_gb 0.9043, ranks 4 |
| L1 | foam_serial | cells 40000, setup 51.05, ms_step 1387, maxrss_gb 0.9033, ranks 1 |
| L1 | jax_gpu1 | cells 40000, setup 16.67, compile 2.297, ms_step 2.915, maxrss_gb 1.441, ranks 1 |
| L1 | jax_gpu2 | cells 80089, setup 18.23, compile 8.665, ms_step 3.389, maxrss_gb 1.369, ranks 2 |
| L1 | jax_gpu4 | cells 160000, setup 20.25, compile 8.035, ms_step 3.56, maxrss_gb 1.415, ranks 4 |
| L1 | jax_mpi2 | cells 80089, setup 20.77, compile 5.208, ms_step 422.8, maxrss_gb 1.197, ranks 2 |
| L1 | jax_mpi4 | cells 160000, setup 20.29, compile 5.855, ms_step 388.9, maxrss_gb 1.237, ranks 4 |
| L1 | jax_serial | cells 40000, setup 15.31, compile 4.734, ms_step 380.4, maxrss_gb 1.011, ranks 1 |
| L2 | amrex_gpu1 | cells 2560000, setup 42.72, ms_step 432.7, maxrss_gb 5.892, ranks 1 |
| L2 | amrex_gpu2 | cells 2560000, setup 34.52, ms_step 221.3, maxrss_gb 5.895, ranks 2 |
| L2 | amrex_gpu4 | cells 2560000, setup 34.69, ms_step 112.4, maxrss_gb 5.896, ranks 4 |
| L2 | amrex_mpi2 | cells 2560000, setup 36, ms_step 1.971e+04, maxrss_gb 5.895, ranks 2 |
| L2 | amrex_mpi4 | cells 2560000, setup 36.21, ms_step 1.094e+04, maxrss_gb 5.894, ranks 4 |
| L2 | amrex_serial | cells 2560000, setup 35.77, ms_step 3.626e+04, maxrss_gb 5.912, ranks 1 |
| L2 | foam_mpi2 | cells 2560000, setup 162.2, ms_step 3.466e+04, maxrss_gb 7.164, ranks 2 |
| L2 | foam_mpi4 | cells 2560000, setup 149.7, ms_step 1.796e+04, maxrss_gb 7.145, ranks 4 |
| L2 | foam_serial | cells 2560000, setup 144, ms_step 6.904e+04, maxrss_gb 11.61, ranks 1 |
| L2 | jax_gpu1 | cells 2560000, setup 65.82, compile 7.42, ms_step 252.8, maxrss_gb 8.777, ranks 1 |
| L2 | jax_gpu2 | cells 2560000, setup 72.85, compile 8.16, ms_step 127.4, maxrss_gb 6.692, ranks 2 |
| L2 | jax_gpu4 | cells 2560000, setup 60.69, compile 8.726, ms_step 63.2, maxrss_gb 5.935, ranks 4 |
| L2 | jax_mpi2 | cells 2560000, setup 66.65, compile 2.476, ms_step 1.095e+04, maxrss_gb 10.38, ranks 2 |
| L2 | jax_mpi4 | cells 2560000, setup 55.52, compile 2.542, ms_step 5909, maxrss_gb 6.807, ranks 4 |
| L2 | jax_serial | cells 2560000, setup 58.98, compile 4.333, ms_step 1.868e+04, maxrss_gb 15.64, ranks 1 |
| L2P | amrex_gpu1 | does not fit 10 min (timeout 595 s) |
| L2P | amrex_gpu2 | does not fit 10 min (timeout 595 s) |
| L2P | amrex_gpu4 | does not fit 5 min (timeout 295 s) |
| L2P | amrex_mpi2 | does not fit 10 min (timeout 595 s) |
| L2P | amrex_mpi4 | does not fit 5 min (timeout 295 s) |
| L2P | amrex_serial | does not fit 10 min (timeout 595 s) |
| L2P | foam_mpi2 | does not fit 5 min (timeout 295 s) |
| L2P | foam_mpi4 | does not fit 5 min (timeout 295 s) |
| L2P | foam_serial | does not fit 5 min (timeout 295 s) |
| L2P | jax_gpu1 | cells 1024000, setup 87.42, compile 49.06, ms_step 1410, maxrss_gb 8.313, ranks 1, solver amgx |
| L2P | jax_gpu2 | cells 1024000, setup 80.35, compile 48.31, ms_step 1338, maxrss_gb 8.031, ranks 2, solver amgx |
| L2P | jax_gpu4 | cells 1024000, setup 75.36, compile 45.3, ms_step 1092, maxrss_gb 5.35, ranks 4, solver amgx |
| L2P | jax_mpi2 | does not fit 10 min (timeout 595 s) |
| L2P | jax_mpi4 | does not fit 10 min (timeout 595 s) |
| L2P | jax_serial | cells 1024000, setup 88.79, compile 35.12, ms_step 3.171e+04, maxrss_gb 27.49, ranks 1, solver pyamg |
| L3 | amrex_gpu1 | catchment 10 m: 291930 cells (148339 catchment), gpu1: setup 51.9 s, 5 s simulated in 0.03 s (23 steps, 1.4 ms/step); water change 618.079167 m3 vs rain 618.079167 m3; h_max 4.1576 m; vs JAX max|dh|/max h 6.516e-15 |
| L3 | amrex_gpu4 | catchment 10 m: 291930 cells (148339 catchment), gpu4: setup 10.8 s, 5 s simulated in 0.01 s (23 steps, 0.6 ms/step); water change 618.079167 m3 vs rain 618.079167 m3; h_max 4.1576 m; vs JAX max|dh|/max h 6.516e-15 |
| L3 | amrex_mpi4 | catchment 10 m: 291930 cells (148339 catchment), cpu4: setup 9.1 s, 5 s simulated in 1.42 s (23 steps, 61.7 ms/step); water change 618.079167 m3 vs rain 618.079167 m3; h_max 4.1576 m; vs JAX max|dh|/max h 6.516e-15 |
| L3 | amrex_serial | catchment 10 m: 291930 cells (148339 catchment), cpu1: setup 8.4 s, 5 s simulated in 5.09 s (23 steps, 221.5 ms/step); water change 618.079167 m3 vs rain 618.079167 m3; h_max 4.1576 m; vs JAX max|dh|/max h 6.516e-15 |
| L3 | jax_gpu4 | fine run: 14.77M cells on 4 GPUs (3691674 owned/rank 0): setup 70 s, compile 2 s, 5 s simulated in 6.7 s; h_max 2.98 m; water change 615.277 m3 vs rain 615.277 m3; CFL dt 0.0218 s (start) 0.0214 s (end) -> ~231 steps, ~29 ms/step; 1 h of catchment ~1.3 h wall |
| L4 | amrex_100 | cells 10000, setup 26.7, ms_step 252.7, maxrss_gb 0.497 |
| L4 | amrex_1600 | cells 2560000, setup 55.2, ms_step 3.601e+04, maxrss_gb 5.894 |
| L4 | amrex_200 | cells 40000, setup 12.71, ms_step 983.9, maxrss_gb 0.3361 |
| L4 | amrex_400 | cells 160000, setup 9.442, ms_step 3025, maxrss_gb 0.6125 |
| L4 | amrex_800 | cells 640000, setup 30.86, ms_step 9693, maxrss_gb 1.698 |
| L4 | foam_100 | cells 10000, setup 39.7, ms_step 449.4, maxrss_gb 0.9043 |
| L4 | foam_1600 | cells 2560000, setup 141.2, ms_step 6.948e+04, maxrss_gb 11.61 |
| L4 | foam_200 | cells 40000, setup 18.96, ms_step 1785, maxrss_gb 0.3798 |
| L4 | foam_400 | cells 160000, setup 20.28, ms_step 5030, maxrss_gb 0.7851 |
| L4 | foam_800 | cells 640000, setup 67.64, ms_step 1.722e+04, maxrss_gb 2.992 |
| L4 | jax_100 | cells 10000, setup 15.6, compile 4.549, ms_step 118.3, maxrss_gb 0.7641 |
| L4 | jax_1600 | cells 2560000, setup 60.98, compile 7.203, ms_step 1.813e+04, maxrss_gb 15.79 |
| L4 | jax_200 | cells 40000, setup 18.73, compile 2.185, ms_step 380.3, maxrss_gb 0.8954 |
| L4 | jax_400 | cells 160000, setup 18.38, compile 3.999, ms_step 1378, maxrss_gb 1.895 |
| L4 | jax_800 | cells 640000, setup 32.82, compile 4.76, ms_step 5316, maxrss_gb 4.632 |
| L5 | write_1ranks | fine mesh 1.00 m: 14.77M cells, ranks 1: build 2.5 s, parallel write 55.4 s, file 1.77 GB |
| L5 | write_2ranks | fine mesh 1.00 m: 14.77M cells, ranks 2: build 1.1 s, parallel write 35.2 s, file 1.77 GB |
| L5 | write_4ranks | fine mesh 1.00 m: 14.77M cells, ranks 4: build 0.7 s, parallel write 26.7 s, file 1.77 GB |

Failures:
- none

Notes:
- C16 jax_foam runs, jax mpi2: jax/mpi2/C16/jax_foam.unsupported: ...
- C1 O1 order 40->80, jax serial: L1(h) 1.93e-03 / 1.29e-03 / 8.01e-04
- C12 bump O1 order 20->40, jax serial: L1(h) 2.37e-03 / 1.63e-03 / 1.05e-03
- C11 smooth O1 order 80->160, jax serial: L1(h) 1.41e-03 / 7.39e-04 / 3.81e-04
- C14 O1 order 40->80, jax serial: L1(h) 1.49e-03 / 1.18e-03 / 7.79e-04
- C1 O2 order 40->80, jax serial: L1(h) 1.12e-03 / 4.57e-04 / 1.14e-04
- C12 bump O2 order 20->40, jax serial: L1(h) 1.34e-03 / 7.13e-04 / 2.20e-04
- C11 smooth O2 order 80->160, jax serial: L1(h) 1.18e-03 / 3.71e-04 / 1.06e-04
- C14 O2 order 40->80, jax serial: L1(h) 8.17e-04 / 7.62e-04 / 2.72e-04
- C2 L1 order 40->80, jax serial: L1(h) 2.24e-03 / 1.18e-03
- C4 sub O1 order 50->100, jax serial: L1(h) 1.59e-02 / 1.15e-02 / 7.79e-03
- C4 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, jax serial: L1(h) 8.72e-03 / 2.82e-03 / 6.03e-04
- C4 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, jax serial: L1(h) 5.84e-02 / 3.89e-02 / 2.40e-02
- C4 trans O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, jax serial: L1(h) 9.14e-03 / 2.55e-03 / 3.45e-04
- C4 trans O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, jax serial: L1(h) 1.60e-02 / 1.16e-02 / 7.87e-03
- C11 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, jax serial: L1(h) 8.50e-03 / 2.84e-03 / 6.01e-04
- C11 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 order 50->100, jax serial: L1(h) 1.18e-02 / 8.36e-03 / 5.22e-03
- C12 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 order 50->100, jax serial: L1(h) 3.46e-03 / 7.78e-04 / 1.46e-04
- C12 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), jax serial: |h30-h60| 2.80e-04, |h60-h120| 7.43e-05
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
- C27 wave time_order 1 order dt/2->dt/4, jax serial: err 1.07e-02 / 4.72e-03 / 1.97e-03
- C27 wave time_order 2 order dt/2->dt/4, jax serial: err 4.23e-04 / 1.03e-04 / 2.44e-05
- C27 wave time_order 3 order dt/2->dt/4, jax serial: err 1.80e-05 / 2.24e-06 / 2.75e-07
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
- C31 SWE lie1 order dt/4->dt/8, jax serial: L1(h) vs dt/32 3.77e-05 / 1.81e-05 / 8.43e-06 / 3.61e-06
- C31 SWE lie2 order dt/4->dt/8, jax serial: L1(h) vs dt/32 5.61e-06 / 2.70e-06 / 1.26e-06 / 5.38e-07
- C31 SWE ssp2 order dt/4->dt/8, jax serial: L1(h) vs dt/32 3.20e-07 / 7.93e-08 / 1.95e-08 / 4.64e-09
- C31 VAM t1 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 8.23e-06 / 4.09e-06 / 2.07e-06
- C31 VAM t2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 8.58e-07 / 4.90e-07 / 3.08e-07
- C31 VAM ssp2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 6.44e-07 / 2.90e-07 / 1.55e-07
- C31 VAM predictor alone ssp2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 5.66e-07 / 1.40e-07 / 3.47e-08
- C32 noslip L_inf order 40->80, jax serial: L_inf 3.33e-04 / 9.12e-05 / 2.38e-05; order 20->40 1.87
- C32 noslip n20 max|h - 1|, jax serial: theory: h stays 1
- C32 noslip n20 max|q_y|, jax serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), jax serial: theory: u = u(y)
- C32 free L_inf order 40->80, jax serial: L_inf 3.32e-04 / 9.11e-05 / 2.38e-05; order 20->40 1.87
- C32 free n20 max|h - 1|, jax serial: theory: h stays 1
- C32 free n20 max|q_y|, jax serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), jax serial: theory: u = u(y)
- C32 control: no-slip profile on free-slip walls, L_inf vs the no-slip exact, jax serial: the walls act
- C32 nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction), jax serial: Dirichlet(momentum) also sets the Riemann outer state: Rusanov damps the wall cells
- C33 imex q(T) vs expm, order dt/2 -> dt/4, jax serial: rel err 2.02e-03 / 1.02e-03 / 5.11e-04 (dt 0.05 / 2^k); implicit Euler (I - dt J)^-N
- C33 imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel), jax serial: the integrator in use, to round-off / the implicit tolerance
- C33 imex uniform state: max|Q - Q[:, 0]|, |h - 1|, jax serial: theory: no flux divergence, h stays
- C33 rk2 q(T) vs expm, order dt/2 -> dt/4, jax serial: rel err 5.71e-05 / 1.36e-05 / 3.34e-06 (dt 0.05 / 2^k); SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N
- C33 rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel), jax serial: the integrator in use, to round-off / the implicit tolerance
- C33 rk2 uniform state: max|Q - Q[:, 0]|, |h - 1|, jax serial: theory: no flux divergence, h stays
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
- C6 C32 mass change (rel), jax serial: max over 8 runs
- C6 C33 mass change (rel), jax serial: max over 6 runs
- C1 O1 order 40->80, foam serial: L1(h) 1.93e-03 / 1.29e-03 / 8.01e-04
- C12 bump O1 order 20->40, foam serial: L1(h) 2.37e-03 / 1.63e-03 / 1.05e-03
- C11 smooth O1 order 80->160, foam serial: L1(h) 1.41e-03 / 7.39e-04 / 3.81e-04
- C14 O1 order 40->80, foam serial: L1(h) 1.49e-03 / 1.18e-03 / 7.79e-04
- C1 O1 max rel vs JAX, foam serial: same stencil
- C11 smooth O1 max rel vs JAX, foam serial: same stencil
- C12 bump O1 foam-JAX L1 diff n40, foam serial: 6.5e-16 / 1.1e-15 / 2.3e-15
- C14 O1 max rel vs JAX, foam serial: same stencil
- C1 O2 order 40->80, foam serial: L1(h) 1.11e-03 / 4.32e-04 / 1.08e-04
- C12 bump O2 order 20->40, foam serial: L1(h) 1.10e-03 / 4.71e-04 / 1.26e-04
- C11 smooth O2 order 80->160, foam serial: L1(h) 1.13e-03 / 3.60e-04 / 1.05e-04
- C14 O2 order 40->80, foam serial: L1(h) 7.43e-04 / 7.81e-04 / 2.70e-04
- C1 O2 foam-JAX L1 diff order 40->80, foam serial: 1.9e-03 / 5.1e-04 / 7.9e-05
- C11 smooth O2 foam-JAX L1 diff order 80->160, foam serial: 4.1e-04 / 9.4e-05 / 1.3e-05
- C12 bump O2 foam-JAX L1 diff order 20->40, foam serial: 6.5e-03 / 3.1e-03 / 1.1e-03
- C14 O2 foam-JAX L1 diff order 40->80, foam serial: 7.7e-04 / 2.8e-04 / 1.3e-04
- C2 L1 order 40->80, foam serial: L1(h) 2.19e-03 / 1.16e-03
- C4 sub O1 order 50->100, foam serial: L1(h) 1.59e-02 / 1.15e-02 / 7.79e-03
- C4 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O1 max rel vs JAX, foam serial: same stencil
- C4 sub O2 order 50->100, foam serial: L1(h) 8.72e-03 / 2.82e-03 / 6.03e-04
- C4 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 foam-JAX L1 diff order 50->100, foam serial: 9.1e-06 / 2.2e-06 / 1.3e-07
- C4 trans O1 order 50->100, foam serial: L1(h) 5.84e-02 / 3.89e-02 / 2.40e-02
- C4 trans O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 max rel vs JAX, foam serial: same stencil
- C4 trans O2 order 50->100, foam serial: L1(h) 9.14e-03 / 2.55e-03 / 3.45e-04
- C4 trans O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 foam-JAX L1 diff n100, foam serial: 2.7e-06 / 4.9e-16 / 8.6e-14
- C11 sub O1 order 50->100, foam serial: L1(h) 1.60e-02 / 1.16e-02 / 7.87e-03
- C11 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 max rel vs JAX, foam serial: same stencil
- C11 sub O2 order 50->100, foam serial: L1(h) 8.50e-03 / 2.84e-03 / 6.01e-04
- C11 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 foam-JAX L1 diff order 50->100, foam serial: 1.3e-05 / 2.0e-06 / 1.4e-07
- C12 sub O1 order 50->100, foam serial: L1(h) 1.18e-02 / 8.36e-03 / 5.22e-03
- C12 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 foam-JAX L1 diff n100, foam serial: 9.2e-16 / 1.7e-15 / 3.1e-15
- C12 sub O2 order 50->100, foam serial: L1(h) 3.30e-03 / 8.09e-04 / 1.20e-04
- C12 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 foam-JAX L1 diff order 50->100, foam serial: 1.2e-03 / 5.4e-04 / 1.2e-04
- C10 space O1 time O1 order (Richardson), foam serial: |h30-h60| 2.80e-04, |h60-h120| 7.43e-05
- C10 space O2 time O1 order (Richardson), foam serial: |h30-h60| 2.60e-04, |h60-h120| 6.51e-05
- C10 space O2 time O2 order (Richardson), foam serial: |h30-h60| 2.61e-04, |h60-h120| 6.49e-05
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
- C32 noslip n20 max|h - 1|, foam serial: theory: h stays 1
- C32 noslip n20 max|q_y|, foam serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), foam serial: theory: u = u(y)
- C32 noslip n20 L_inf|foam - JAX| / L_inf(JAX - exact), foam serial: foam's own stencils at second order
- C32 free n20 max|h - 1|, foam serial: theory: h stays 1
- C32 free n20 max|q_y|, foam serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), foam serial: theory: u = u(y)
- C32 free n20 L_inf|foam - JAX| / L_inf(JAX - exact), foam serial: foam's own stencils at second order
- C33 imex q(T) vs expm, order dt/2 -> dt/4, foam serial: rel err 2.02e-03 / 1.02e-03 / 5.11e-04 (dt 0.05 / 2^k); implicit Euler (I - dt J)^-N
- C33 imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel), foam serial: the integrator in use, to round-off / the implicit tolerance
- C33 imex uniform state: max|Q - Q[:, 0]|, |h - 1|, foam serial: theory: no flux divergence, h stays
- C33 rk2 q(T) vs expm, order dt/2 -> dt/4, foam serial: rel err 5.71e-05 / 1.36e-05 / 3.34e-06 (dt 0.05 / 2^k); SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N
- C33 rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel), foam serial: the integrator in use, to round-off / the implicit tolerance
- C33 rk2 uniform state: max|Q - Q[:, 0]|, |h - 1|, foam serial: theory: no flux divergence, h stays
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
- C6 C32 mass change (rel), foam serial: max over 2 runs
- C6 C33 mass change (rel), foam serial: max over 6 runs
- C1 O1 order 40->80, amrex serial: L1(h) 1.93e-03 / 1.29e-03 / 8.01e-04
- C11 smooth O1 order 80->160, amrex serial: L1(h) 1.41e-03 / 7.39e-04 / 3.81e-04
- C14 O1 order 40->80, amrex serial: L1(h) 1.49e-03 / 1.18e-03 / 7.79e-04
- C1 O2 order 40->80, amrex serial: L1(h) 1.12e-03 / 4.57e-04 / 1.14e-04
- C11 smooth O2 order 80->160, amrex serial: L1(h) 1.18e-03 / 3.71e-04 / 1.06e-04
- C14 O2 order 40->80, amrex serial: L1(h) 8.17e-04 / 7.62e-04 / 2.72e-04
- C2 L1 order 40->80, amrex serial: L1(h) 2.24e-03 / 1.18e-03
- C4 sub O1 order 50->100, amrex serial: L1(h) 1.59e-02 / 1.15e-02 / 7.79e-03
- C4 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, amrex serial: L1(h) 8.72e-03 / 2.82e-03 / 6.03e-04
- C4 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, amrex serial: L1(h) 5.84e-02 / 3.89e-02 / 2.40e-02
- C4 trans O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, amrex serial: L1(h) 9.14e-03 / 2.55e-03 / 3.45e-04
- C4 trans O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, amrex serial: L1(h) 1.60e-02 / 1.16e-02 / 7.87e-03
- C11 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, amrex serial: L1(h) 8.50e-03 / 2.84e-03 / 6.01e-04
- C11 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), amrex serial: |h30-h60| 2.80e-04, |h60-h120| 7.43e-05
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
- C32 noslip n20 max|h - 1|, amrex serial: theory: h stays 1
- C32 noslip n20 max|q_y|, amrex serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), amrex serial: theory: u = u(y)
- C32 free n20 max|h - 1|, amrex serial: theory: h stays 1
- C32 free n20 max|q_y|, amrex serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), amrex serial: theory: u = u(y)
- C33 imex q(T) vs expm, order dt/2 -> dt/4, amrex serial: rel err 2.02e-03 / 1.02e-03 / 5.11e-04 (dt 0.05 / 2^k); implicit Euler (I - dt J)^-N
- C33 imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel), amrex serial: the integrator in use, to round-off / the implicit tolerance
- C33 imex uniform state: max|Q - Q[:, 0]|, |h - 1|, amrex serial: theory: no flux divergence, h stays
- C33 rk2 q(T) vs expm, order dt/2 -> dt/4, amrex serial: rel err 5.71e-05 / 1.36e-05 / 3.34e-06 (dt 0.05 / 2^k); SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N
- C33 rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel), amrex serial: the integrator in use, to round-off / the implicit tolerance
- C33 rk2 uniform state: max|Q - Q[:, 0]|, |h - 1|, amrex serial: theory: no flux divergence, h stays
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
- C6 C32 mass change (rel), amrex serial: max over 2 runs
- C6 C33 mass change (rel), amrex serial: max over 6 runs

Summed run wall time (s, without process start-up): amrex 4039, foam 3867, jax 6912
