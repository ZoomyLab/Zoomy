| case | check | backend | serial | mpi2 | mpi4 | gpu1 | gpu-mpi | threshold (source) |
|---|---|---|---|---|---|---|---|---|
| C1 | O1 max rel vs JAX | amrex | 8.30e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | amrex | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 max rel vs JAX | amrex | 1.14e-13 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C1 | O2 order 40->80 | amrex | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C1 | O1 max rel vs JAX | foam | 8.60e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C1 | O1 order 40->80 | foam | 7.15e-01 PASS | - | - | n/a | n/a | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 foam-JAX L1 diff order 40->80 | foam | 2.70e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C1 | O2 order 40->80 | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C1 | O1 order 40->80 | jax | 7.15e-01 PASS | - | - | - | - | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 order 40->80 | jax | 2.00e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C1 | O1 order 40->80 | numpy | 7.15e-01 PASS | n/a | n/a | n/a | n/a | >= 0.6 (observed 0.71 (pre-asymptotic, theory 1)) |
| C1 | O2 order 40->80 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C1 | numpy = JAX (max rel, worst run) | numpy | 5.86e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C2 | L1 order 40->80 | amrex | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C2 | max rel vs JAX | amrex | 1.29e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C2 | L1 order 40->80 | foam | 8.94e-01 PASS | - | - | n/a | n/a | >= 0.6 (spec) |
| C2 | L1 vs JAX n80 (rel) | foam | 2.58e-04 PASS | - | - | n/a | n/a | <= 0.0015 (observed (foam 7.1e-4)) |
| C2 | L1 order 40->80 | jax | 8.96e-01 PASS | - | - | - | - | >= 0.6 (spec) |
| C2 | L1 order 40->80 | numpy | 8.96e-01 PASS | n/a | n/a | n/a | n/a | >= 0.6 (spec) |
| C2 | numpy = JAX (max rel, worst run) | numpy | 1.67e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C3 | O1 max|u| after 100 steps | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | amrex | 5.36e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | max rel vs JAX | amrex | 5.80e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C3 | O1 max|u| after 100 steps | foam | 5.11e-16 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | foam | 1.27e-15 PASS | - | - | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O1 max|u| after 100 steps | jax | 2.60e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | jax | 7.39e-16 PASS | - | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O1 max|u| after 100 steps | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | numpy | 7.39e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | numpy = JAX (max rel, worst run) | numpy | 3.38e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max rel vs JAX | amrex | 7.93e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | amrex | 5.64e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | amrex | 5.82e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max rel vs JAX | amrex | 3.61e-14 PASS | - | - | - | - | <= 3.27173e-12 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 3.3e-13)) |
| C4 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | amrex | 7.41e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | amrex | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | amrex | 8.28e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | amrex | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | amrex | 7.04e-17 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | amrex | 7.05e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max rel vs JAX | amrex | 3.57e-13 PASS | - | - | - | - | <= 3.42079e-11 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 3.4e-12)) |
| C4 | trans O2 max|q/q_in - 1| n100 | amrex | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | amrex | 4.20e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | amrex | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | sub O1 max rel vs JAX | foam | 7.93e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | sub O1 max|q/q_in - 1| n100 | foam | 1.37e-02 | - | - | n/a | n/a | information |
| C4 | sub O1 max|q_y| n100 | foam | 5.98e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | foam | 5.82e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 foam-JAX L1 diff order 50->100 | foam | 4.37e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C4 | sub O2 max|q/q_in - 1| n100 | foam | 2.96e-03 | - | - | n/a | n/a | information |
| C4 | sub O2 max|q_y| n100 | foam | 8.37e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | foam | 2.22e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | trans O1 max rel vs JAX | foam | 9.66e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C4 | trans O1 max|q/q_in - 1| n100 | foam | 5.13e-02 | - | - | n/a | n/a | information |
| C4 | trans O1 max|q_y| n100 | foam | 1.65e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | foam | 7.05e-01 PASS | - | - | n/a | n/a | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 foam-JAX L1 diff n100 | foam | 1.96e-13 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C4 | trans O2 max|q/q_in - 1| n100 | foam | 1.91e-03 | - | - | n/a | n/a | information |
| C4 | trans O2 max|q_y| n100 | foam | 2.11e-16 PASS | - | - | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | foam | 2.85e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C4 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C4 | sub O1 max|q_y| n100 | jax | 3.59e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | jax | 5.82e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C4 | sub O2 max|q_y| n100 | jax | 8.31e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | jax | 2.22e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | trans O1 max|q/q_in - 1| n100 | jax | 5.13e-02 | - | - | - | - | information |
| C4 | trans O1 max|q_y| n100 | jax | 9.76e-17 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | jax | 7.05e-01 PASS | - | - | - | - | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max|q/q_in - 1| n100 | jax | 1.91e-03 | - | - | - | - | information |
| C4 | trans O2 max|q_y| n100 | jax | 3.55e-16 PASS | - | - | - | - | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | jax | 2.85e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C4 | numpy = JAX (max rel, worst run) | numpy | 2.54e-12 PASS | n/a | n/a | n/a | n/a | <= 3.42079e-11 (10 x JAX round-off sensitivity (3.4e-12)) |
| C4 | sub O1 max|q/q_in - 1| n100 | numpy | 1.37e-02 | n/a | n/a | n/a | n/a | information |
| C4 | sub O1 max|q_y| n100 | numpy | 5.98e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O1 order 50->100 | numpy | 5.82e-01 PASS | n/a | n/a | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C4 | sub O2 max|q/q_in - 1| n100 | numpy | 2.96e-03 | n/a | n/a | n/a | n/a | information |
| C4 | sub O2 max|q_y| n100 | numpy | 1.21e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | sub O2 order 50->100 | numpy | 2.22e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C4 | trans O1 max|q/q_in - 1| n100 | numpy | 5.13e-02 | n/a | n/a | n/a | n/a | information |
| C4 | trans O1 max|q_y| n100 | numpy | 1.32e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O1 order 50->100 | numpy | 7.05e-01 PASS | n/a | n/a | n/a | n/a | >= 0.55 (observed 0.71, pre-asymptotic (theory 1)) |
| C4 | trans O2 max|q/q_in - 1| n100 | numpy | 1.91e-03 | n/a | n/a | n/a | n/a | information |
| C4 | trans O2 max|q_y| n100 | numpy | 2.58e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (theory (1-D flow, symmetric stencils)) |
| C4 | trans O2 order 50->100 | numpy | 2.85e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C5 | h_min | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C5 | max rel vs JAX | amrex | 7.10e-15 PASS | - | - | - | - | <= 1e-09 (observed 6e-10 (wet/dry eigenvalue cluster)) |
| C5 | L1 rel vs JAX | foam | 1.68e-02 PASS | - | - | n/a | n/a | <= 0.035 (observed (foam 1.7e-2)) |
| C5 | h_min | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | >= 0 (theory (positivity)) |
| C5 | h_min | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C5 | h_min | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | >= 0 (theory (positivity)) |
| C5 | numpy = JAX (max rel, worst run) | numpy | 6.53e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | amrex | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | amrex | 1.73e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | amrex | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | amrex | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | amrex | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | foam | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | foam | 1.73e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | foam | 2.82e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | foam | 1.73e-16 PASS | 1.73e-16 PASS | 1.73e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | foam | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | foam | 2.20e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | foam | 2.22e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | foam | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | jax | 1.41e-16 PASS | 2.82e-16 PASS | 2.82e-16 PASS | 2.82e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | jax | 0.00e+00 PASS | 1.73e-16 PASS | 1.73e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | jax | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | 1.41e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | jax | 2.20e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | jax | 2.22e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C36 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | jax | 1.95e-16 PASS | 1.95e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 1.95e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C1 mass change (rel) | numpy | 1.41e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C11 mass change (rel) | numpy | 3.47e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C12 mass change (rel) | numpy | 1.41e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C13 mass change (rel) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C14 mass change (rel) | numpy | 1.41e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | numpy | 1.41e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C20 mass change (rel) | numpy | 2.20e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C23 mass change (rel) | numpy | 2.22e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C32 mass change (rel) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C33 mass change (rel) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C36 mass change (rel) | numpy | 1.11e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | numpy | 1.95e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | amrex | - | 2.04e-16 PASS | 2.04e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | amrex | - | 7.69e-12 PASS | 9.56e-12 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | foam | - | 1.04e-15 PASS | 7.46e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | foam | - | 7.42e-16 PASS | 9.80e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | foam | - | 1.11e-14 PASS | 9.99e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | foam | - | 3.18e-16 PASS | 3.18e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | foam | - | 4.31e-16 PASS | 5.74e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | foam | - | 7.99e-16 PASS | 9.33e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | foam | - | 6.61e-16 PASS | 8.26e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | foam | - | 9.73e-15 PASS | 4.18e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | foam | - | 7.98e-12 PASS | 8.56e-12 PASS | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C7 | C1 o2_n40 parallel vs serial CPU (max rel) | jax | - | 1.38e-13 PASS | 1.38e-13 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C13 o2_x2.5 parallel vs serial CPU (max rel) | jax | - | 1.88e-15 PASS | 1.88e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C14 o2_n40 parallel vs serial CPU (max rel) | jax | - | 4.15e-16 PASS | 4.15e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o1 parallel vs serial CPU (max rel) | jax | - | 3.18e-16 PASS | 3.18e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 pulse_o2 parallel vs serial CPU (max rel) | jax | - | 5.84e-16 PASS | 5.84e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C21 sod_y parallel vs serial CPU (max rel) | jax | - | 1.78e-15 PASS | 1.15e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C26 visc parallel vs serial CPU (max rel) | jax | - | 6.87e-16 PASS | 6.62e-16 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 n48 parallel vs serial CPU (max rel) | jax | - | 4.66e-15 PASS | 4.89e-15 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C8 vam parallel vs serial CPU (max rel) | jax | - | 4.25e-11 PASS | 5.27e-11 PASS | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | amrex | 7.53e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam max rel vs JAX | amrex | 1.77e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_newton max rel vs JAX | amrex | 1.29e-12 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | vam_t2 max rel vs JAX | amrex | 4.07e-11 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | foam | 7.72e-12 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | vam L1 rel vs JAX | foam | 1.12e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_newton L1 rel vs JAX | foam | 1.14e-12 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_t2 L1 rel vs JAX | foam | 3.98e-11 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | vam_tri L1 rel vs JAX | foam | 4.43e-12 PASS | - | - | n/a | n/a | <= 0.03 (observed (foam 1.6e-2)) |
| C8 | Newton vs linear pressure (max rel) | jax | 1.28e-11 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | Newton vs linear pressure (max rel) | numpy | 5.36e-11 PASS | n/a | n/a | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)) |
| C8 | numpy = JAX (max rel, worst run) | numpy | 4.28e-10 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | amrex | - | - | - | 8.05e-15 PASS | 8.05e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | amrex | - | - | - | 7.18e-12 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C9 | C1 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 4.03e-14 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C14 o2_n40 GPU vs serial CPU (max rel) | jax | - | - | - | 4.15e-16 PASS | 5.32e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C26 visc GPU vs serial CPU (max rel) | jax | - | - | - | 8.66e-16 PASS | 9.17e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 n48 GPU vs serial CPU (max rel) | jax | - | - | - | 3.18e-15 PASS | 5.18e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C9 | C8 vam GPU vs serial CPU (max rel) | jax | - | - | - | 6.73e-12 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C10 | space O1 time O1 order (Richardson) | amrex | 1.92e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | amrex | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | amrex | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | foam | 1.92e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | foam | 2.00e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | foam | 2.01e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed (theory 2, minmod)) |
| C10 | space O1 time O1 order (Richardson) | jax | 1.92e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | jax | 2.00e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | jax | 2.01e+00 PASS | - | - | - | - | >= 1.5 (observed (theory 2, minmod)) |
| C10 | numpy = JAX (max rel, worst run) | numpy | 2.48e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C10 | space O1 time O1 order (Richardson) | numpy | 1.92e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1) |
| C10 | space O2 time O1 order (Richardson) | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1 (time)) |
| C10 | space O2 time O2 order (Richardson) | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.5 (observed (theory 2, minmod)) |
| C11 | smooth O1 max rel vs JAX | amrex | 1.67e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | amrex | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 max rel vs JAX | amrex | 2.42e-13 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O2 order 80->160 | amrex | 1.65e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | amrex | 1.59e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | amrex | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | amrex | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max rel vs JAX | amrex | 3.83e-06 PASS | - | - | - | - | <= 0.00021946 (10 x JAX round-off sensitivity (Q0 perturbed 1e-14: 2.2e-05)) |
| C11 | sub O2 max|q/q_in - 1| n100 | amrex | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | amrex | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C11 | smooth O1 max rel vs JAX | foam | 2.19e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | foam | 1.02e+00 PASS | - | - | n/a | n/a | >= 0.9 (theory 1) |
| C11 | smooth O2 foam-JAX L1 diff order 80->160 | foam | 2.08e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | smooth O2 order 80->160 | foam | 1.63e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max rel vs JAX | foam | 1.78e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | sub O1 max|q/q_in - 1| n100 | foam | 1.37e-02 | - | - | n/a | n/a | information |
| C11 | sub O1 order 50->100 | foam | 5.89e-01 PASS | - | - | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 foam-JAX L1 diff order 50->100 | foam | 2.65e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C11 | sub O2 max|q/q_in - 1| n100 | foam | 2.96e-03 | - | - | n/a | n/a | information |
| C11 | sub O2 order 50->100 | foam | 2.28e+00 PASS | - | - | n/a | n/a | >= 1.8 (theory 2) |
| C11 | smooth O1 order 80->160 | jax | 1.02e+00 PASS | - | - | - | - | >= 0.9 (theory 1) |
| C11 | smooth O2 order 80->160 | jax | 1.65e+00 PASS | - | - | - | - | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max|q/q_in - 1| n100 | jax | 1.37e-02 | - | - | - | - | information |
| C11 | sub O1 order 50->100 | jax | 5.89e-01 PASS | - | - | - | - | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max|q/q_in - 1| n100 | jax | 2.96e-03 | - | - | - | - | information |
| C11 | sub O2 order 50->100 | jax | 2.28e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C11 | numpy = JAX (max rel, worst run) | numpy | 2.41e-13 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C11 | smooth O1 order 80->160 | numpy | 1.02e+00 PASS | n/a | n/a | n/a | n/a | >= 0.9 (theory 1) |
| C11 | smooth O2 order 80->160 | numpy | 1.65e+00 PASS | n/a | n/a | n/a | n/a | >= 1.5 (observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)) |
| C11 | sub O1 max|q/q_in - 1| n100 | numpy | 1.37e-02 | n/a | n/a | n/a | n/a | information |
| C11 | sub O1 order 50->100 | numpy | 5.89e-01 PASS | n/a | n/a | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C11 | sub O2 max|q/q_in - 1| n100 | numpy | 2.96e-03 | n/a | n/a | n/a | n/a | information |
| C11 | sub O2 order 50->100 | numpy | 2.28e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C12 | runs | amrex | n/a | - | - | - | - | information |
| C12 | bump O1 foam-JAX L1 diff n40 | foam | 2.30e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
| C12 | bump O1 order 20->40 | foam | 6.52e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 foam-JAX L1 diff order 20->40 | foam | 1.44e+00 PASS | - | - | n/a | n/a | >= 0.25 (observed 0.33-0.39 (O1 triangles: both converge slowly)) |
| C12 | bump O2 order 20->40 | foam | 1.92e+00 PASS | - | - | n/a | n/a | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | sub O1 foam-JAX L1 diff n100 | foam | 2.96e-15 PASS | - | - | n/a | n/a | <= 1e-11 (tolerance (round-off, L1 rel): foam = JAX) |
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
| C12 | bump O1 order 20->40 | numpy | 6.52e-01 PASS | n/a | n/a | n/a | n/a | >= 0.5 (observed 0.65 (coarse triangles, theory 1)) |
| C12 | bump O2 order 20->40 | numpy | 1.69e+00 PASS | n/a | n/a | n/a | n/a | >= 1.4 (observed 1.69 (coarse triangles, theory 2)) |
| C12 | numpy = JAX (max rel, worst run) | numpy | 2.38e-13 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C12 | sub O1 max|q/q_in - 1| n100 | numpy | 9.43e-03 | n/a | n/a | n/a | n/a | information |
| C12 | sub O1 max|q_y| n100 | numpy | 6.70e-03 | n/a | n/a | n/a | n/a | information |
| C12 | sub O1 order 50->100 | numpy | 6.94e-01 PASS | n/a | n/a | n/a | n/a | >= 0.45 (observed 0.58, pre-asymptotic (theory 1)) |
| C12 | sub O2 max|q/q_in - 1| n100 | numpy | 1.47e-03 | n/a | n/a | n/a | n/a | information |
| C12 | sub O2 max|q_y| n100 | numpy | 3.24e-03 | n/a | n/a | n/a | n/a | information |
| C12 | sub O2 order 50->100 | numpy | 2.42e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C13 | O1 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | max rel vs JAX | amrex | 1.38e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 max rel vs JAX | foam | 4.53e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | foam | 2.13e-16 PASS | 2.13e-16 PASS | 2.13e-16 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | foam | 1.96e-15 PASS | 1.48e-15 PASS | 1.69e-15 PASS | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | jax | 3.20e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | jax | 9.27e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | - | <= 1e-12 (tolerance (round-off)) |
| C13 | O1 shift by L/2 (max rel) | numpy | 2.13e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | O2 shift by L/2 (max rel) | numpy | 6.89e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C13 | numpy = JAX (max rel, worst run) | numpy | 2.12e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C14 | O1 max rel vs JAX | amrex | 1.50e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | amrex | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 max rel vs JAX | amrex | 1.70e-12 PASS | - | - | - | - | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O2 order 40->80 | amrex | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 max rel vs JAX | foam | 1.50e-12 PASS | - | - | n/a | n/a | <= 1e-08 (tolerance (implicit Newton 1e-8; foam: FD Jacobian)) |
| C14 | O1 order 40->80 | foam | 6.31e-01 PASS | - | - | n/a | n/a | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 foam-JAX L1 diff order 40->80 | foam | 1.20e+00 PASS | - | - | n/a | n/a | >= 0.5 (observed: foam-JAX difference shrinks under refinement) |
| C14 | O2 order 40->80 | foam | 1.55e+00 PASS | - | - | n/a | n/a | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 order 40->80 | jax | 6.31e-01 PASS | - | - | - | - | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 order 40->80 | jax | 1.52e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | O1 order 40->80 | numpy | 6.31e-01 PASS | n/a | n/a | n/a | n/a | >= 0.5 (observed 0.63 (pre-asymptotic, theory 1)) |
| C14 | O2 order 40->80 | numpy | 1.52e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1 (Lie splitting)) |
| C14 | numpy = JAX (max rel, worst run) | numpy | 7.11e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-08 (tolerance (implicit Newton 1e-8)) |
| C15 | runs | amrex | n/a | - | - | - | - | information |
| C15 | runs | foam | n/a | - | - | n/a | n/a | information |
| C15 | chorin jvp vs serial (rel) | jax | - | 5.41e-12 PASS | - | 2.24e-15 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C15 | chorin |FD - jvp| (rel) | jax | 1.65e-08 PASS | 7.92e-09 PASS | - | 4.29e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | chorin |rev - jvp| (rel) | jax | 1.16e-12 PASS | 3.66e-11 PASS | - | 8.52e-14 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | hyperbolic jvp vs serial (rel) | jax | - | 1.29e-15 PASS | - | 2.40e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | hyperbolic |FD - jvp| (rel) | jax | 7.88e-09 PASS | 1.39e-08 PASS | - | 7.88e-09 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | hyperbolic |rev - jvp| (rel) | jax | 4.25e-15 PASS | 1.15e-14 PASS | - | 1.11e-15 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | imex jvp vs serial (rel) | jax | - | 3.86e-15 PASS | - | 2.81e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C15 | imex |FD - jvp| (rel) | jax | 3.66e-07 PASS | 1.93e-07 PASS | - | 1.36e-07 PASS | - | <= 1e-05 (tolerance (central FD, eps 1e-3..1e-5)) |
| C15 | imex |rev - jvp| (rel) | jax | 1.32e-14 PASS | 9.12e-15 PASS | - | 3.16e-15 PASS | - | <= 1e-10 (tolerance (checkpointed reverse vs forward)) |
| C15 | runs | numpy | n/a | n/a | n/a | n/a | n/a | information |
| C16 | amrex_amrex coupled vs single (max) | amrex | 3.55e-15 PASS | 3.55e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_amrex mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax coupled vs single (max) | amrex | 3.33e-15 PASS | 3.33e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | amrex_jax mass drift | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam coupled vs single (max) | foam | 5.16e-15 PASS | 5.16e-15 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | foam_foam mass drift | foam | 0.00e+00 PASS | 0.00e+00 PASS | - | n/a | n/a | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam coupled vs single (max) | jax | 4.94e-15 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam mass drift | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_foam runs | jax | - | unsupported | - | - | - | information |
| C16 | jax_jax coupled vs single (max) | jax | 3.61e-15 PASS | 3.61e-15 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | jax_jax mass drift | jax | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 1e-14 (tolerance (gate 26: coupled = single domain at O1)) |
| C16 | runs | numpy | n/a | n/a | n/a | n/a | n/a | information |
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
| C17 | dry runs | amrex | nan **FAIL** | - | - | - | - | <= 0 (the run completes) |
| C17 | lake max|u| | amrex | 1.29e-15 PASS | - | - | - | - | <= 1e-12 (theory (well-balanced)) |
| C17 | lake vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | wave order 16->32 (composite) | amrex | 2.53e+00 PASS | - | - | - | - | >= 1.5 (observed (O2, static patch; theory 2)) |
| C17 | wave16 L1(rho - exact) | amrex | 5.27e-03 | - | - | - | - | information |
| C17 | wave16 vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | wave32 L1(rho - exact) | amrex | 9.15e-04 | - | - | - | - | information |
| C17 | wave32 vs CPU 1 rank (max rel) | amrex | - | 0.00e+00 PASS | - | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C17 | runs | foam | n/a | - | - | n/a | n/a | information |
| C17 | runs | jax | n/a | - | - | - | - | information |
| C17 | runs | numpy | n/a | n/a | n/a | n/a | n/a | information |
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
| C18 | Sod O1 L1(rho) n100 | numpy | 2.09e-02 PASS | n/a | n/a | n/a | n/a | <= 0.042 (observed 2.09e-2 (3-D worker)) |
| C18 | Sod O1 max|u_perp| | numpy | 3.25e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O1 order 50->100 | numpy | 5.36e-01 PASS | n/a | n/a | n/a | n/a | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C18 | Sod O1 x / y / z profiles (max) | numpy | 1.33e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 L1(rho) n100 | numpy | 5.13e-03 PASS | n/a | n/a | n/a | n/a | <= 0.01 (observed 5.13e-3) |
| C18 | Sod O2 max|u_perp| | numpy | 1.21e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | Sod O2 order 50->100 | numpy | 9.02e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C18 | Sod O2 x / y / z profiles (max) | numpy | 4.44e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C18 | numpy = JAX (max rel, worst run) | numpy | 2.09e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
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
| C19 | numpy = JAX (max rel, worst run) | numpy | 6.77e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 max|u - u0|, |p - 1| | numpy | 8.88e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o1 order 16->24 | numpy | 8.53e-01 PASS | n/a | n/a | n/a | n/a | >= 0.75 (observed 0.85, pre-asymptotic) |
| C19 | wave o2minmod max|u - u0|, |p - 1| | numpy | 1.11e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2minmod order 16->24 | numpy | 2.79e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2 (observed 2.53)) |
| C19 | wave o2none max|u - u0|, |p - 1| | numpy | 1.11e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C19 | wave o2none order 16->24 | numpy | 2.77e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
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
| C20 | O1 energy change (rel) | numpy | 1.72e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O1 mirror / permutation symmetry (max rel) | numpy | 3.23e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 energy change (rel) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | O2 mirror / permutation symmetry (max rel) | numpy | 5.55e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C20 | numpy = JAX (max rel, worst run) | numpy | 1.11e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C21 | numpy = JAX (max rel, worst run) | numpy | 7.11e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C22 | runs | amrex | n/a | - | - | - | - | information |
| C22 | runs | foam | n/a | - | - | n/a | n/a | information |
| C22 | o1_adaptive |FD - jvp| (rel) | jax | 8.81e-09 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_adaptive |rev - jvp| (rel) | jax | 7.47e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o1_dt |FD - jvp| (rel) | jax | 2.05e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o1_dt |rev - jvp| (rel) | jax | 2.12e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | o2_dt |FD - jvp| (rel) | jax | 4.44e-08 PASS | - | - | - | - | <= 1e-06 (observed 4.4e-8 (best eps of 1e-3..1e-6)) |
| C22 | o2_dt |rev - jvp| (rel) | jax | 1.58e-15 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C22 | runs | numpy | n/a | n/a | n/a | n/a | n/a | information |
| C23 | O1 n16 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n24 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n8 max rel vs JAX | amrex | 3.54e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O2 max rel vs JAX | amrex | 3.08e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n16 max rel vs JAX | foam | 3.54e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n24 max rel vs JAX | foam | 5.31e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O1 n8 max rel vs JAX | foam | 3.54e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C23 | O2 L1|backend - JAX| order 16->24 | foam | 1.98e+00 PASS | - | - | n/a | n/a | >= 1.5 (observed 2.11 (different stencils, converging)) |
| C23 | numpy = JAX (max rel, worst run) | numpy | 5.16e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
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
| C24 | Sod O1 L1(rho) n100 | numpy | 2.05e-02 | n/a | n/a | n/a | n/a | information |
| C24 | Sod O1 max|u_perp| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O1 order 50->100 | numpy | 5.39e-01 PASS | n/a | n/a | n/a | n/a | >= 0.45 (observed 0.54 (discontinuous: theory <= 1)) |
| C24 | Sod O1 x / y / z profiles (max) | numpy | 6.66e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 L1(rho) n100 | numpy | 5.14e-03 | n/a | n/a | n/a | n/a | information |
| C24 | Sod O2 max|u_perp| | numpy | 6.29e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | Sod O2 order 50->100 | numpy | 9.02e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (observed 0.90 (discontinuous: theory <= 1)) |
| C24 | Sod O2 x / y / z profiles (max) | numpy | 1.78e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | numpy = JAX (max rel, worst run) | numpy | 1.60e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 max|u - u0|, |p - 1| | numpy | 1.11e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o1 order 32->64 | numpy | 9.59e-01 PASS | n/a | n/a | n/a | n/a | >= 0.85 (observed 0.96 (theory 1)) |
| C24 | wave o2minmod max|u - u0|, |p - 1| | numpy | 1.33e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2minmod order 32->64 | numpy | 1.82e+00 PASS | n/a | n/a | n/a | n/a | >= 1.7 (observed 1.83) |
| C24 | wave o2none max|u - u0|, |p - 1| | numpy | 1.55e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C24 | wave o2none order 32->64 | numpy | 2.32e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C26 | max rel vs JAX | amrex | 8.15e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C26 | L1 rel vs JAX | foam | 1.63e-03 | - | - | n/a | n/a | information |
| C26 | numpy = JAX (max rel, worst run) | numpy | 2.26e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
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
| C27 | ODE time_order 1 order 32->64 steps | numpy | 1.01e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C27 | ODE time_order 2 order 32->64 steps | numpy | 1.92e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C27 | ODE time_order 3 order 32->64 steps | numpy | 3.00e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory 3, slack 0.2) |
| C27 | numpy = JAX (max rel, worst run) | numpy | 6.22e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C27 | wave time_order 1 order dt/2->dt/4 | numpy | 1.26e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C27 | wave time_order 2 order dt/2->dt/4 | numpy | 2.08e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C27 | wave time_order 3 order dt/2->dt/4 | numpy | 3.02e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory 3, slack 0.2) |
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
| C28 | 1d impl neu max rel vs JAX | foam | 5.44e-11 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl per max rel vs JAX | foam | 2.74e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
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
| C28 | quad impl per max rel vs JAX | foam | 1.48e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri expl dir0 max rel vs JAX | foam | 5.97e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl dirL max rel vs JAX | foam | 3.01e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl neu max rel vs JAX | foam | 5.97e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl per max rel vs JAX | foam | 3.52e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max rel vs JAX | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max|u - L| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri impl dir0 max rel vs JAX | foam | 7.46e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl dirL max rel vs JAX | foam | 1.51e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl neu max rel vs JAX | foam | 7.46e-16 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl per max rel vs JAX | foam | 1.07e-10 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
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
| C28 | 1d expl dir0 L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl dir0 max rel vs JAX | numpy | 1.28e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl dirL L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl dirL max rel vs JAX | numpy | 1.04e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl neu L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl neu max rel vs JAX | numpy | 1.46e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl per L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d expl per max rel vs JAX | numpy | 1.21e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | 1d expl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | 1d impl dir0 L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl dir0 max rel vs JAX | numpy | 3.90e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl dirL L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl dirL max rel vs JAX | numpy | 9.81e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl neu L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl neu max rel vs JAX | numpy | 3.90e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl per L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | 1d impl per max rel vs JAX | numpy | 1.60e-11 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | 1d impl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | quad expl dir0 L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl dir0 max rel vs JAX | numpy | 1.50e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl dirL L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl dirL max rel vs JAX | numpy | 3.04e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl neu L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl neu max rel vs JAX | numpy | 4.51e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl per L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad expl per max rel vs JAX | numpy | 5.49e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | quad expl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | quad impl dir0 L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl dir0 max rel vs JAX | numpy | 7.06e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl dirL L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl dirL max rel vs JAX | numpy | 6.35e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl neu L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl neu max rel vs JAX | numpy | 7.06e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl per L_inf order 32->64 | numpy | 2.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | quad impl per max rel vs JAX | numpy | 2.83e-11 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | quad impl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri expl dir0 L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl dir0 max rel vs JAX | numpy | 2.99e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl dirL L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl dirL max rel vs JAX | numpy | 1.51e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl neu L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl neu max rel vs JAX | numpy | 2.99e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl per L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri expl per max rel vs JAX | numpy | 1.07e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance, gate 43: the same two-point fluxes and ghosts) |
| C28 | tri expl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C28 | tri impl dir0 L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl dir0 max rel vs JAX | numpy | 7.06e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl dirL L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl dirL max rel vs JAX | numpy | 5.33e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl neu L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl neu max rel vs JAX | numpy | 7.07e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl per L_inf order 32->64 | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, gate 41) |
| C28 | tri impl per max rel vs JAX | numpy | 2.85e-11 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl steady max rel vs JAX | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance, gate 43: Newton 1e-10, GMRES 1e-13) |
| C28 | tri impl steady max|u - L| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
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
| C29 | p2 dir d1 boundary order 24->48 | numpy | 1.97e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 dir d1 interior order 24->48 | numpy | 1.98e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 dir d2 boundary order 24->48 | numpy | 9.29e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 dir d2 interior order 24->48 | numpy | 1.99e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 neu d1 boundary order 24->48 | numpy | 1.94e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 neu d1 interior order 24->48 | numpy | 1.98e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p2 neu d2 boundary order 24->48 | numpy | 9.20e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p2 neu d2 interior order 24->48 | numpy | 1.99e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 dir d1 boundary order 24->48 | numpy | 2.94e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 dir d1 interior order 24->48 | numpy | 3.32e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 dir d2 boundary order 24->48 | numpy | 1.96e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 dir d2 interior order 24->48 | numpy | 1.93e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 dir d3 boundary order 24->48 | numpy | 9.34e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 dir d3 interior order 24->48 | numpy | 1.34e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 neu d1 boundary order 24->48 | numpy | 2.94e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 neu d1 interior order 24->48 | numpy | 3.32e+00 PASS | n/a | n/a | n/a | n/a | >= 2.8 (theory p+1-k = 3, slack 0.2, gate 44) |
| C29 | p3 neu d2 boundary order 24->48 | numpy | 1.96e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 neu d2 interior order 24->48 | numpy | 1.93e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory p+1-k = 2, slack 0.2, gate 44) |
| C29 | p3 neu d3 boundary order 24->48 | numpy | 8.27e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C29 | p3 neu d3 interior order 24->48 | numpy | 1.34e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory p+1-k = 1, slack 0.2, gate 44) |
| C30 | runs (n/a: gate 39 calls the JAX runtime (lsq_gradient_per_field)) | amrex | n/a | - | - | - | - | information |
| C30 | runs (n/a: gate 39 calls the JAX runtime (lsq_gradient_per_field)) | foam | n/a | - | - | n/a | n/a | information |
| C30 | gate 39 checks failing | jax | 0.00e+00 PASS | - | - | - | - | <= 0 (spec, gate 39: exact to 1e-09) |
| C30 | gate 39 checks failing | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 0 (spec, gate 39: exact to 1e-09) |
| C31 | swe_ssp2_k0 max rel vs JAX | amrex | 2.93e-14 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C31 | vam_ssp2_k0 max rel vs JAX | amrex | 2.97e-13 PASS | - | - | - | - | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C31 | swe_ssp2_k0 max rel vs JAX | foam | 1.91e-14 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C31 | vam_ssp2_k0 max rel vs JAX | foam | 2.97e-13 PASS | - | - | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C31 | SWE lie1 order dt/4->dt/8 | jax | 1.22e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C31 | SWE lie2 order dt/4->dt/8 | jax | 1.22e+00 PASS | - | - | - | - | <= 1.4 (theory 1: Lie splitting caps SSP-RK2 (observed 1.22, the dt/32 reference lifts the last pair)) |
| C31 | SWE ssp2 order dt/4->dt/8 | jax | 2.07e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C31 | VAM predictor alone ssp2 order (Richardson) | jax | 2.01e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C31 | VAM ssp2 order (Richardson) | jax | 1.96e+00 PASS | - | - | - | - | >= 1.8 (theory 2 (increment projection, ANALYSIS 45e)) |
| C31 | VAM t1 order (Richardson) | jax | 1.03e+00 PASS | - | - | - | - | >= 0.8 (theory 1) |
| C31 | VAM t2 order (Richardson) | jax | 1.13e+00 PASS | - | - | - | - | >= 0.8 (theory 1 (the Lie source per stage; observed 1.12, increment projection, ANALYSIS 45e)) |
| C31 | SWE lie1 order dt/4->dt/8 | numpy | 1.22e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1) |
| C31 | SWE lie2 order dt/4->dt/8 | numpy | 1.22e+00 PASS | n/a | n/a | n/a | n/a | <= 1.4 (theory 1: Lie splitting caps SSP-RK2 (observed 1.22, the dt/32 reference lifts the last pair)) |
| C31 | SWE ssp2 order dt/4->dt/8 | numpy | 2.07e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C31 | VAM predictor alone ssp2 order (Richardson) | numpy | 2.01e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C31 | VAM ssp2 order (Richardson) | numpy | 1.96e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2 (increment projection, ANALYSIS 45e)) |
| C31 | VAM t1 order (Richardson) | numpy | 1.03e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1) |
| C31 | VAM t2 order (Richardson) | numpy | 1.13e+00 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1 (the Lie source per stage; observed 1.12, increment projection, ANALYSIS 45e)) |
| C31 | numpy = JAX (max rel, worst run) | numpy | 1.08e-13 PASS | n/a | n/a | n/a | n/a | <= 1e-09 (tolerance (10 x pressure rtol 1e-10, over the steps)) |
| C32 | free n20 L_inf vs exact | amrex | 3.36e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 max rel vs JAX | amrex | 3.33e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C32 | free n20 max|h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 L_inf vs exact | amrex | 3.36e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 max rel vs JAX | amrex | 2.22e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C32 | noslip n20 max|h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 L_inf vs exact | amrex | 3.37e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip_split n20 max rel vs JAX | amrex | 2.57e-16 PASS | - | - | - | - | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
| C32 | noslip_split n20 max|h - 1| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 max|q_y| | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 x-variation of q_x (max rel) | amrex | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 L_inf vs exact | foam | 3.36e-04 PASS | - | - | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 L_inf|foam - JAX| / L_inf(JAX - exact) | foam | 1.93e-01 PASS | - | - | n/a | n/a | <= 0.5 (observed 0.21 (foam's own stencils at second order)) |
| C32 | free n20 max|h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | foam | 1.14e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | foam | 1.12e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 L_inf vs exact | foam | 3.37e-04 PASS | - | - | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 L_inf|foam - JAX| / L_inf(JAX - exact) | foam | 2.14e-02 PASS | - | - | n/a | n/a | <= 0.5 (observed 0.21 (foam's own stencils at second order)) |
| C32 | noslip n20 max|h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | foam | 7.43e-17 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | foam | 1.03e-15 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 L_inf vs exact | foam | 3.38e-04 PASS | - | - | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip_split n20 L_inf|foam - JAX| / L_inf(JAX - exact) | foam | 1.39e-02 PASS | - | - | n/a | n/a | <= 0.5 (observed 0.21 (foam's own stencils at second order)) |
| C32 | noslip_split n20 max|h - 1| | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 max|q_y| | foam | 6.83e-17 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 x-variation of q_x (max rel) | foam | 9.97e-16 PASS | - | - | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | control: no-slip profile on free-slip walls, L_inf vs the no-slip exact | jax | 4.17e-01 PASS | - | - | - | - | >= 0.1 (observed 0.42 (a wall that acts: the no-slip profile is not a free-slip solution)) |
| C32 | free L_inf order 40->80 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C32 | free n20 L_inf vs exact | jax | 3.36e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 max|h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | jax | 1.48e-17 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | jax | 8.90e-18 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip L_inf order 40->80 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C32 | noslip n20 L_inf vs exact | jax | 3.36e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 max|h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | jax | 9.95e-18 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | jax | 1.42e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split L_inf order 40->80 | jax | 1.94e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C32 | noslip_split n20 L_inf vs exact | jax | 3.37e-04 PASS | - | - | - | - | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip_split n20 max|h - 1| | jax | 0.00e+00 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 max|q_y| | jax | 1.32e-17 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 x-variation of q_x (max rel) | jax | 2.85e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction) | jax | 1.62e-04 PASS | - | - | - | - | <= 0.0004 (observed 1.7e-4 (a regression bound; 2.5e-3 at g = 9.81)) |
| C32 | nu 0 on no-slip walls: wall friction, rel. change of sum(q_x vol) | jax | 4.31e-05 | - | - | - | - | information |
| C32 | nu 0 on split walls: max|u - u0| | jax | 3.87e-04 PASS | - | - | - | - | <= 0.0008 (observed 4.1e-4 (a regression bound: O2 Rusanov dissipation, no wall exit)) |
| C32 | nu 0 on split walls: wall friction, rel. change of sum(q_x vol) | jax | 4.36e-16 PASS | - | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C32 | control: no-slip profile on free-slip walls, L_inf vs the no-slip exact | numpy | 4.17e-01 PASS | n/a | n/a | n/a | n/a | >= 0.1 (observed 0.42 (a wall that acts: the no-slip profile is not a free-slip solution)) |
| C32 | free L_inf order 40->80 | numpy | 1.94e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C32 | free n20 L_inf vs exact | numpy | 3.36e-04 PASS | n/a | n/a | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | free n20 max|h - 1| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 max|q_y| | numpy | 3.45e-17 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | free n20 x-variation of q_x (max rel) | numpy | 2.85e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip L_inf order 40->80 | numpy | 1.94e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C32 | noslip n20 L_inf vs exact | numpy | 3.36e-04 PASS | n/a | n/a | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip n20 max|h - 1| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 max|q_y| | numpy | 3.65e-17 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip n20 x-variation of q_x (max rel) | numpy | 2.85e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split L_inf order 40->80 | numpy | 1.94e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C32 | noslip_split n20 L_inf vs exact | numpy | 3.37e-04 PASS | n/a | n/a | n/a | n/a | <= 0.0007 (observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)) |
| C32 | noslip_split n20 max|h - 1| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 max|q_y| | numpy | 2.75e-17 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | noslip_split n20 x-variation of q_x (max rel) | numpy | 4.27e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction) | numpy | 1.62e-04 PASS | n/a | n/a | n/a | n/a | <= 0.0004 (observed 1.7e-4 (a regression bound; 2.5e-3 at g = 9.81)) |
| C32 | nu 0 on no-slip walls: wall friction, rel. change of sum(q_x vol) | numpy | 4.31e-05 | n/a | n/a | n/a | n/a | information |
| C32 | nu 0 on split walls: max|u - u0| | numpy | 3.87e-04 PASS | n/a | n/a | n/a | n/a | <= 0.0008 (observed 4.1e-4 (a regression bound: O2 Rusanov dissipation, no wall exit)) |
| C32 | nu 0 on split walls: wall friction, rel. change of sum(q_x vol) | numpy | 2.18e-16 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C32 | numpy = JAX (max rel, worst run) | numpy | 1.57e-12 PASS | n/a | n/a | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |
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
| C33 | imex q(T) vs expm at dt 0.0125 (rel) | numpy | 5.11e-04 PASS | n/a | n/a | n/a | n/a | <= 0.001 (observed 5.1e-4 (implicit Euler, theory O(dt))) |
| C33 | imex q(T) vs expm, order dt/2 -> dt/4 | numpy | 9.95e-01 PASS | n/a | n/a | n/a | n/a | >= 0.8 (theory 1, slack 0.2) |
| C33 | imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel) | numpy | 1.27e-13 PASS | n/a | n/a | n/a | n/a | <= 1e-10 (tolerance (Newton 1e-12; observed 3e-12 C++, 4e-15 JAX)) |
| C33 | imex uniform state: max|Q - Q[:, 0]|, |h - 1| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | numpy = JAX (max rel, worst run) | numpy | 5.36e-14 PASS | n/a | n/a | n/a | n/a | <= 1e-10 (tolerance (Newton 1e-12; observed 1.3e-12)) |
| C33 | rk2 q(T) vs expm at dt 0.0125 (rel) | numpy | 3.34e-06 PASS | n/a | n/a | n/a | n/a | <= 7e-06 (observed 3.3e-6 (SSP-RK2, theory O(dt^2))) |
| C33 | rk2 q(T) vs expm, order dt/2 -> dt/4 | numpy | 2.03e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2, slack 0.2) |
| C33 | rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel) | numpy | 2.79e-15 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C33 | rk2 uniform state: max|Q - Q[:, 0]|, |h - 1| | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 1e-12 (tolerance (round-off)) |
| C34 | nx 16 max rel |L - L_JAX| | amrex | 5.62e-10 PASS | - | - | - | - | <= 2e-07 (observed 7e-8 foam (its own O2 stencils in the explicit stage: O(dt) in L), 6e-10 AMReX (round-off / dt)) |
| C34 | nx 16 max rel |L - L_JAX| | foam | 6.22e-08 PASS | - | - | n/a | n/a | <= 2e-07 (observed 7e-8 foam (its own O2 stencils in the explicit stage: O(dt) in L), 6e-10 AMReX (round-off / dt)) |
| C34 | q_x_0: vs the exact div(h tau') (mixed derivatives), order 32->64 | jax | 1.99e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C34 | q_x_0: vs the exact div(h tau') at nx 64 (max rel) | jax | 2.26e-03 PASS | - | - | - | - | <= 0.006 (observed 3.9e-3 (q_y), 2.3e-3 (q_x) on JAX vs the exact div(h tau')) |
| C34 | q_y_0: vs the exact div(h tau') (mixed derivatives), order 32->64 | jax | 1.98e+00 PASS | - | - | - | - | >= 1.8 (theory 2) |
| C34 | q_y_0: vs the exact div(h tau') at nx 64 (max rel) | jax | 3.85e-03 PASS | - | - | - | - | <= 0.006 (observed 3.9e-3 (q_y), 2.3e-3 (q_x) on JAX vs the exact div(h tau')) |
| C34 | numpy = JAX (max rel, worst run) | numpy | 1.12e-11 PASS | n/a | n/a | n/a | n/a | <= 2e-07 (observed 7e-8 foam (its own O2 stencils in the explicit stage: O(dt) in L), 6e-10 AMReX (round-off / dt)) |
| C34 | nx 16 max rel |L - L_JAX| | numpy | 1.12e-11 PASS | n/a | n/a | n/a | n/a | <= 2e-07 (observed 7e-8 foam (its own O2 stencils in the explicit stage: O(dt) in L), 6e-10 AMReX (round-off / dt)) |
| C34 | q_x_0: vs the exact div(h tau') (mixed derivatives), order 32->64 | numpy | 1.99e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C34 | q_x_0: vs the exact div(h tau') at nx 64 (max rel) | numpy | 2.26e-03 PASS | n/a | n/a | n/a | n/a | <= 0.006 (observed 3.9e-3 (q_y), 2.3e-3 (q_x) on JAX vs the exact div(h tau')) |
| C34 | q_y_0: vs the exact div(h tau') (mixed derivatives), order 32->64 | numpy | 1.98e+00 PASS | n/a | n/a | n/a | n/a | >= 1.8 (theory 2) |
| C34 | q_y_0: vs the exact div(h tau') at nx 64 (max rel) | numpy | 3.85e-03 PASS | n/a | n/a | n/a | n/a | <= 0.006 (observed 3.9e-3 (q_y), 2.3e-3 (q_x) on JAX vs the exact div(h tau')) |
| C35 | c1: misspelt restart_from fails at once | amrex | 1.00e+00 PASS | 1.00e+00 PASS | - | - | - | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c1: restarted at T/2 vs 0 -> T (max abs) | amrex | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 0 (spec: bitwise) |
| C35 | c8: misspelt restart_from fails at once | amrex | 1.00e+00 PASS | - | - | - | - | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c8: restarted at T/2 vs 0 -> T (max abs) | amrex | 0.00e+00 PASS | - | - | - | - | <= 0 (spec: bitwise) |
| C35 | c1: misspelt restart_from fails at once | foam | 1.00e+00 PASS | 1.00e+00 PASS | - | n/a | n/a | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c1: restarted at T/2 vs 0 -> T (max abs) | foam | 0.00e+00 PASS | 0.00e+00 PASS | - | n/a | n/a | <= 0 (spec: bitwise) |
| C35 | c8: misspelt restart_from fails at once | foam | 1.00e+00 PASS | - | - | n/a | n/a | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c8: restarted at T/2 vs 0 -> T (max abs) | foam | 0.00e+00 PASS | - | - | n/a | n/a | <= 0 (spec: bitwise) |
| C35 | c1: misspelt restart_from fails at once | jax | 1.00e+00 PASS | 1.00e+00 PASS | - | - | - | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c1: restarted at T/2 vs 0 -> T (max abs) | jax | 0.00e+00 PASS | 0.00e+00 PASS | - | - | - | <= 0 (spec: bitwise) |
| C35 | c8: misspelt restart_from fails at once | jax | 1.00e+00 PASS | - | - | - | - | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c8: restarted at T/2 vs 0 -> T (max abs) | jax | 0.00e+00 PASS | - | - | - | - | <= 0 (spec: bitwise) |
| C35 | c1: misspelt restart_from fails at once | numpy | 1.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c1: restarted at T/2 vs 0 -> T (max abs) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 0 (spec: bitwise) |
| C35 | c8: misspelt restart_from fails at once | numpy | 1.00e+00 PASS | n/a | n/a | n/a | n/a | >= 1 (spec: FileNotFoundError naming restart_from) |
| C35 | c8: restarted at T/2 vs 0 -> T (max abs) | numpy | 0.00e+00 PASS | n/a | n/a | n/a | n/a | <= 0 (spec: bitwise) |
| C36 | runs | amrex | n/a | - | - | - | - | information |
| C36 | runs | foam | n/a | - | - | n/a | n/a | information |
| C36 | numpy = JAX (max rel, worst run) | numpy | 1.39e-13 PASS | n/a | n/a | n/a | n/a | <= 1e-08 (tolerance (Newton 1e-12, GMRES 1e-13)) |

## Large tier

| level | run | result |
|---|---|---|

Failures:
- C17 dry runs, amrex serial: nan (need <= 0) amrex/serial/C17/dry.err: ...

Notes:
- C17 dry runs, amrex serial: amrex/serial/C17/dry.err: ...
- C16 jax_foam runs, jax mpi2: jax/mpi2/C16/jax_foam.unsupported: ...
- C1 O1 order 40->80, jax serial: L1(h) 1.78e-03 / 1.18e-03 / 7.20e-04
- C12 bump O1 order 20->40, jax serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, jax serial: L1(h) 8.12e-04 / 4.16e-04 / 2.05e-04
- C14 O1 order 40->80, jax serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, jax serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C12 bump O2 order 20->40, jax serial: L1(h) 1.33e-03 / 7.13e-04 / 2.21e-04
- C11 smooth O2 order 80->160, jax serial: L1(h) 1.34e-03 / 3.91e-04 / 1.25e-04
- C14 O2 order 40->80, jax serial: L1(h) 7.69e-04 / 7.63e-04 / 2.67e-04
- C2 L1 order 40->80, jax serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, jax serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, jax serial: L1(h) 8.67e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, jax serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, jax serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, jax serial: L1(h) 8.82e-03 / 2.42e-03 / 3.36e-04
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
- C27 wave time_order 3 order dt/2->dt/4, jax serial: err 1.85e-05 / 2.32e-06 / 2.85e-07
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
- C34 q_x_0: vs the exact div(h tau') (mixed derivatives), order 32->64, jax serial: max rel 3.55e-02 / 8.99e-03 / 2.26e-03; vs the SystemModel operator 3.55e-02 / 8.99e-03 / 2.26e-03; vs its diagonal-index part only 2.92e-01 / 3.20e-01 / 3.28e-01
- C34 q_y_0: vs the exact div(h tau') (mixed derivatives), order 32->64, jax serial: max rel 5.83e-02 / 1.52e-02 / 3.85e-03; vs the SystemModel operator 5.83e-02 / 1.52e-02 / 3.85e-03; vs its diagonal-index part only 9.38e-01 / 1.04e+00 / 1.07e+00
- C30 gate 39 checks failing, jax serial: PASS=72 FAIL=0 (tol 1e-09); the ghost as a plain sample is inexact in 15 cases (sensitivity)
- C31 SWE lie1 order dt/4->dt/8, jax serial: L1(h) vs dt/32 3.87e-05 / 1.86e-05 / 8.66e-06 / 3.71e-06
- C31 SWE lie2 order dt/4->dt/8, jax serial: L1(h) vs dt/32 5.63e-06 / 2.71e-06 / 1.26e-06 / 5.39e-07
- C31 SWE ssp2 order dt/4->dt/8, jax serial: L1(h) vs dt/32 3.31e-07 / 8.21e-08 / 2.02e-08 / 4.81e-09
- C31 VAM t1 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 8.22e-06 / 3.98e-06 / 1.95e-06
- C31 VAM t2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 4.41e-07 / 1.71e-07 / 7.85e-08
- C31 VAM ssp2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 3.76e-07 / 9.59e-08 / 2.47e-08
- C31 VAM predictor alone ssp2 order (Richardson), jax serial: L1(h) of dt/2^k - dt/2^(k+1) 5.77e-07 / 1.42e-07 / 3.53e-08
- C35 c1: misspelt restart_from fails at once, jax serial: after 0.5 s: restart_from '/Users/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_jax_serial/jax/serial/C35/c1_half/stat.h5': no such file; did you mean 'state.h5'?
- C35 c1: misspelt restart_from fails at once, jax mpi2: after 0.8 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-wspeed/docs/dev/symbolic_solver/matrix/work/jax/mpi2/C35/c1_half/stat.h5': no such file; did you mean 'state.h5'?
- C35 c8: misspelt restart_from fails at once, jax serial: after 4.3 s: restart_from '/Users/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_jax_serial/jax/serial/C35/c8_half/stat.h5': no such file; did you mean 'state.h5'?
- C32 noslip L_inf order 40->80, jax serial: L_inf 3.36e-04 / 9.16e-05 / 2.38e-05; order 20->40 1.88
- C32 noslip n20 max|h - 1|, jax serial: theory: h stays 1
- C32 noslip n20 max|q_y|, jax serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), jax serial: theory: u = u(y)
- C32 free L_inf order 40->80, jax serial: L_inf 3.36e-04 / 9.16e-05 / 2.38e-05; order 20->40 1.87
- C32 free n20 max|h - 1|, jax serial: theory: h stays 1
- C32 free n20 max|q_y|, jax serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), jax serial: theory: u = u(y)
- C32 noslip_split L_inf order 40->80, jax serial: L_inf 3.37e-04 / 9.17e-05 / 2.38e-05; order 20->40 1.88
- C32 noslip_split n20 max|h - 1|, jax serial: theory: h stays 1
- C32 noslip_split n20 max|q_y|, jax serial: theory: v stays 0
- C32 noslip_split n20 x-variation of q_x (max rel), jax serial: theory: u = u(y)
- C32 control: no-slip profile on free-slip walls, L_inf vs the no-slip exact, jax serial: the walls act
- C32 nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction), jax serial: Dirichlet(momentum) also sets the Riemann outer state: Rusanov damps the wall cells
- C32 nu 0 on no-slip walls: wall friction, rel. change of sum(q_x vol), jax serial: the interior fluxes cancel: the walls' share
- C32 nu 0 on split walls: wall friction, rel. change of sum(q_x vol), jax serial: Wall face: the reflected Riemann state carries no x-momentum flux
- C32 nu 0 on split walls: max|u - u0|, jax serial: the interior Rusanov dissipation of the O2 face jumps, piled up in the wall cells
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
- C6 C32 mass change (rel), jax serial: max over 12 runs
- C6 C33 mass change (rel), jax serial: max over 6 runs
- C6 C36 mass change (rel), jax serial: max over 1 runs
- C1 O1 order 40->80, foam serial: L1(h) 1.78e-03 / 1.18e-03 / 7.20e-04
- C12 bump O1 order 20->40, foam serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, foam serial: L1(h) 8.12e-04 / 4.16e-04 / 2.05e-04
- C14 O1 order 40->80, foam serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O1 max rel vs JAX, foam serial: same stencil
- C11 smooth O1 max rel vs JAX, foam serial: same stencil
- C12 bump O1 foam-JAX L1 diff n40, foam serial: 7.3e-16 / 1.1e-15 / 2.3e-15
- C14 O1 max rel vs JAX, foam serial: same stencil
- C1 O2 order 40->80, foam serial: L1(h) 1.15e-03 / 4.45e-04 / 1.11e-04
- C12 bump O2 order 20->40, foam serial: L1(h) 1.11e-03 / 4.70e-04 / 1.24e-04
- C11 smooth O2 order 80->160, foam serial: L1(h) 1.28e-03 / 3.81e-04 / 1.23e-04
- C14 O2 order 40->80, foam serial: L1(h) 7.02e-04 / 7.79e-04 / 2.66e-04
- C1 O2 foam-JAX L1 diff order 40->80, foam serial: 1.8e-03 / 4.9e-04 / 7.6e-05
- C11 smooth O2 foam-JAX L1 diff order 80->160, foam serial: 5.0e-04 / 8.6e-05 / 2.0e-05
- C12 bump O2 foam-JAX L1 diff order 20->40, foam serial: 6.4e-03 / 3.0e-03 / 1.1e-03
- C14 O2 foam-JAX L1 diff order 40->80, foam serial: 7.2e-04 / 2.5e-04 / 1.1e-04
- C2 L1 order 40->80, foam serial: L1(h) 2.11e-03 / 1.14e-03
- C4 sub O1 order 50->100, foam serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O1 max rel vs JAX, foam serial: same stencil
- C4 sub O2 order 50->100, foam serial: L1(h) 8.67e-03 / 2.73e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 foam-JAX L1 diff order 50->100, foam serial: 8.7e-06 / 2.4e-06 / 1.2e-07
- C4 trans O1 order 50->100, foam serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 max rel vs JAX, foam serial: same stencil
- C4 trans O2 order 50->100, foam serial: L1(h) 8.82e-03 / 2.42e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 foam-JAX L1 diff n100, foam serial: 2.5e-06 / 1.2e-15 / 2.0e-13
- C11 sub O1 order 50->100, foam serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 max rel vs JAX, foam serial: same stencil
- C11 sub O2 order 50->100, foam serial: L1(h) 8.33e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 foam-JAX L1 diff order 50->100, foam serial: 1.1e-05 / 1.7e-06 / 2.7e-07
- C12 sub O1 order 50->100, foam serial: L1(h) 1.16e-02 / 8.13e-03 / 5.02e-03
- C12 sub O1 max|q/q_in - 1| n100, foam serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 foam-JAX L1 diff n100, foam serial: 9.0e-16 / 1.5e-15 / 3.0e-15
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
- C34 nx 16 max rel |L - L_JAX|, foam serial: vs the exact div(h tau') 3.55e-02 / 5.83e-02
- C35 c1: misspelt restart_from fails at once, foam serial: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_foam_serial/foam/serial/C35/c1_hlf': no such file; did you mean 'c1_half'?
- C35 c1: misspelt restart_from fails at once, foam mpi2: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-wspeed/docs/dev/symbolic_solver/matrix/work/foam/mpi2/C35/c1_hlf': no such file; did you mean 'c1_half'?
- C35 c8: misspelt restart_from fails at once, foam serial: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_foam_serial/foam/serial/C35/c8_hlf': no such file; did you mean 'c8_half'?
- C32 noslip n20 max|h - 1|, foam serial: theory: h stays 1
- C32 noslip n20 max|q_y|, foam serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), foam serial: theory: u = u(y)
- C32 noslip n20 L_inf|foam - JAX| / L_inf(JAX - exact), foam serial: foam's own stencils at second order
- C32 free n20 max|h - 1|, foam serial: theory: h stays 1
- C32 free n20 max|q_y|, foam serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), foam serial: theory: u = u(y)
- C32 free n20 L_inf|foam - JAX| / L_inf(JAX - exact), foam serial: foam's own stencils at second order
- C32 noslip_split n20 max|h - 1|, foam serial: theory: h stays 1
- C32 noslip_split n20 max|q_y|, foam serial: theory: v stays 0
- C32 noslip_split n20 x-variation of q_x (max rel), foam serial: theory: u = u(y)
- C32 noslip_split n20 L_inf|foam - JAX| / L_inf(JAX - exact), foam serial: foam's own stencils at second order
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
- C6 C32 mass change (rel), foam serial: max over 3 runs
- C6 C33 mass change (rel), foam serial: max over 6 runs
- C1 O1 order 40->80, amrex serial: L1(h) 1.78e-03 / 1.18e-03 / 7.20e-04
- C11 smooth O1 order 80->160, amrex serial: L1(h) 8.12e-04 / 4.16e-04 / 2.05e-04
- C14 O1 order 40->80, amrex serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, amrex serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C11 smooth O2 order 80->160, amrex serial: L1(h) 1.34e-03 / 3.91e-04 / 1.25e-04
- C14 O2 order 40->80, amrex serial: L1(h) 7.69e-04 / 7.63e-04 / 2.67e-04
- C2 L1 order 40->80, amrex serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, amrex serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, amrex serial: L1(h) 8.67e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, amrex serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, amrex serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, amrex serial: L1(h) 8.82e-03 / 2.42e-03 / 3.36e-04
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
- C34 nx 16 max rel |L - L_JAX|, amrex serial: vs the exact div(h tau') 3.55e-02 / 5.83e-02
- C35 c1: misspelt restart_from fails at once, amrex serial: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_amrex_serial/amrex/serial/C35/c1_hlf': no such file; did you mean 'c1_half'?
- C35 c1: misspelt restart_from fails at once, amrex mpi2: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-wspeed/docs/dev/symbolic_solver/matrix/work/amrex/mpi2/C35/c1_hlf': no such file; did you mean 'c1_half'?
- C35 c8: misspelt restart_from fails at once, amrex serial: after 0.0 s: restart_from '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_amrex_serial/amrex/serial/C35/c8_hlf': no such file; did you mean 'c8_half'?
- C32 noslip n20 max|h - 1|, amrex serial: theory: h stays 1
- C32 noslip n20 max|q_y|, amrex serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), amrex serial: theory: u = u(y)
- C32 free n20 max|h - 1|, amrex serial: theory: h stays 1
- C32 free n20 max|q_y|, amrex serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), amrex serial: theory: u = u(y)
- C32 noslip_split n20 max|h - 1|, amrex serial: theory: h stays 1
- C32 noslip_split n20 max|q_y|, amrex serial: theory: v stays 0
- C32 noslip_split n20 x-variation of q_x (max rel), amrex serial: theory: u = u(y)
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
- C6 C32 mass change (rel), amrex serial: max over 3 runs
- C6 C33 mass change (rel), amrex serial: max over 6 runs
- C1 O1 order 40->80, numpy serial: L1(h) 1.78e-03 / 1.18e-03 / 7.20e-04
- C12 bump O1 order 20->40, numpy serial: L1(h) 2.27e-03 / 1.54e-03 / 9.80e-04
- C11 smooth O1 order 80->160, numpy serial: L1(h) 8.12e-04 / 4.16e-04 / 2.05e-04
- C14 O1 order 40->80, numpy serial: L1(h) 1.29e-03 / 1.07e-03 / 6.92e-04
- C1 O2 order 40->80, numpy serial: L1(h) 1.14e-03 / 4.69e-04 / 1.17e-04
- C12 bump O2 order 20->40, numpy serial: L1(h) 1.33e-03 / 7.13e-04 / 2.21e-04
- C11 smooth O2 order 80->160, numpy serial: L1(h) 1.34e-03 / 3.91e-04 / 1.25e-04
- C14 O2 order 40->80, numpy serial: L1(h) 7.69e-04 / 7.63e-04 / 2.67e-04
- C2 L1 order 40->80, numpy serial: L1(h) 2.16e-03 / 1.16e-03
- C4 sub O1 order 50->100, numpy serial: L1(h) 1.57e-02 / 1.13e-02 / 7.54e-03
- C4 sub O1 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C4 sub O2 order 50->100, numpy serial: L1(h) 8.67e-03 / 2.72e-03 / 5.85e-04
- C4 sub O2 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O1 order 50->100, numpy serial: L1(h) 5.65e-02 / 3.73e-02 / 2.29e-02
- C4 trans O1 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C4 trans O2 order 50->100, numpy serial: L1(h) 8.82e-03 / 2.42e-03 / 3.36e-04
- C4 trans O2 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O1 order 50->100, numpy serial: L1(h) 1.59e-02 / 1.15e-02 / 7.62e-03
- C11 sub O1 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C11 sub O2 order 50->100, numpy serial: L1(h) 8.32e-03 / 2.82e-03 / 5.82e-04
- C11 sub O2 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O1 order 50->100, numpy serial: L1(h) 1.16e-02 / 8.13e-03 / 5.02e-03
- C12 sub O1 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C12 sub O2 order 50->100, numpy serial: L1(h) 3.29e-03 / 7.65e-04 / 1.43e-04
- C12 sub O2 max|q/q_in - 1| n100, numpy serial: steady discharge = inflow (open-domain mass balance)
- C10 space O1 time O1 order (Richardson), numpy serial: |h30-h60| 2.79e-04, |h60-h120| 7.38e-05
- C10 space O2 time O1 order (Richardson), numpy serial: |h30-h60| 2.59e-04, |h60-h120| 6.46e-05
- C10 space O2 time O2 order (Richardson), numpy serial: |h30-h60| 2.59e-04, |h60-h120| 6.43e-05
- C1 numpy = JAX (max rel, worst run), numpy serial: worst run o2_n40
- C2 numpy = JAX (max rel, worst run), numpy serial: worst run n80
- C3 numpy = JAX (max rel, worst run), numpy serial: worst run o2
- C4 numpy = JAX (max rel, worst run), numpy serial: worst run trans_o2_n100
- C5 numpy = JAX (max rel, worst run), numpy serial: worst run n48
- C8 numpy = JAX (max rel, worst run), numpy serial: worst run vam_tri
- C10 numpy = JAX (max rel, worst run), numpy serial: worst run s1_t1_n30
- C11 numpy = JAX (max rel, worst run), numpy serial: worst run smooth_o2_n80
- C12 numpy = JAX (max rel, worst run), numpy serial: worst run bump_o2_n20
- C13 numpy = JAX (max rel, worst run), numpy serial: worst run o2_x2.5
- C14 numpy = JAX (max rel, worst run), numpy serial: worst run o2_n20
- C18 numpy = JAX (max rel, worst run), numpy serial: worst run o2_n100_ax0
- C19 numpy = JAX (max rel, worst run), numpy serial: worst run o2none_n16
- C20 numpy = JAX (max rel, worst run), numpy serial: worst run o2
- C21 numpy = JAX (max rel, worst run), numpy serial: worst run sod_y
- C23 numpy = JAX (max rel, worst run), numpy serial: worst run o2_n24
- C24 numpy = JAX (max rel, worst run), numpy serial: worst run sod_o2_n100_ax1
- C26 numpy = JAX (max rel, worst run), numpy serial: worst run visc
- C27 numpy = JAX (max rel, worst run), numpy serial: worst run wave_t3
- C31 numpy = JAX (max rel, worst run), numpy serial: worst run vam_t1_k0
- C32 numpy = JAX (max rel, worst run), numpy serial: worst run noslip_n20
- C33 numpy = JAX (max rel, worst run), numpy serial: worst run imex_k1
- C34 numpy = JAX (max rel, worst run), numpy serial: worst run n16
- C36 numpy = JAX (max rel, worst run), numpy serial: worst run sme1
- C18 Sod O1 order 50->100, numpy serial: L1(rho) 3.03e-02 / 2.09e-02
- C18 Sod O2 order 50->100, numpy serial: L1(rho) 9.59e-03 / 5.13e-03
- C19 wave o1 order 16->24, numpy serial: L1(rho) 6.99e-02 / 4.38e-02 / 3.10e-02
- C19 wave o1 max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C19 wave o2none order 16->24, numpy serial: L1(rho) 3.88e-02 / 7.25e-03 / 2.35e-03
- C19 wave o2none max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C19 wave o2minmod order 16->24, numpy serial: L1(rho) 4.02e-02 / 7.45e-03 / 2.41e-03
- C19 wave o2minmod max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C24 Sod O1 order 50->100, numpy serial: L1(rho) 2.98e-02 / 2.05e-02
- C24 Sod O2 order 50->100, numpy serial: L1(rho) 9.62e-03 / 5.14e-03
- C24 wave o1 order 32->64, numpy serial: L1(rho) 3.03e-02 / 1.60e-02 / 8.25e-03
- C24 wave o1 max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C24 wave o2none order 32->64, numpy serial: L1(rho) 3.41e-03 / 5.59e-04 / 1.12e-04
- C24 wave o2none max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C24 wave o2minmod order 32->64, numpy serial: L1(rho) 3.60e-03 / 8.58e-04 / 2.43e-04
- C24 wave o2minmod max|u - u0|, |p - 1|, numpy serial: contact: velocity and pressure constant
- C27 ODE time_order 1 order 32->64 steps, numpy serial: err 3.25e-02 / 1.60e-02 / 7.96e-03
- C27 ODE time_order 2 order 32->64 steps, numpy serial: err 2.48e-04 / 6.95e-05 / 1.83e-05
- C27 ODE time_order 3 order 32->64 steps, numpy serial: err 4.12e-05 / 5.15e-06 / 6.43e-07
- C27 wave time_order 1 order dt/2->dt/4, numpy serial: err 1.10e-02 / 4.82e-03 / 2.01e-03
- C27 wave time_order 2 order dt/2->dt/4, numpy serial: err 4.31e-04 / 1.05e-04 / 2.49e-05
- C27 wave time_order 3 order dt/2->dt/4, numpy serial: err 1.85e-05 / 2.32e-06 / 2.85e-07
- C28 1d expl dir0 L_inf order 32->64, numpy serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl neu L_inf order 32->64, numpy serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl dirL L_inf order 32->64, numpy serial: L_inf 3.86e-04 / 9.67e-05 / 2.42e-05
- C28 1d expl per L_inf order 32->64, numpy serial: L_inf 1.40e-03 / 3.52e-04 / 8.81e-05
- C28 1d expl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 1d expl dir0 max rel vs JAX, numpy serial: L_inf vs exact 3.86e-04
- C28 1d expl neu max rel vs JAX, numpy serial: L_inf vs exact 3.86e-04
- C28 1d expl dirL max rel vs JAX, numpy serial: L_inf vs exact 3.86e-04
- C28 1d expl per max rel vs JAX, numpy serial: L_inf vs exact 1.40e-03
- C28 1d expl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 1d expl steady max|u - L|, numpy serial: on the backend itself
- C28 1d impl dir0 L_inf order 32->64, numpy serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl neu L_inf order 32->64, numpy serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl dirL L_inf order 32->64, numpy serial: L_inf 9.54e-04 / 2.39e-04 / 5.99e-05
- C28 1d impl per L_inf order 32->64, numpy serial: L_inf 3.35e-03 / 8.45e-04 / 2.12e-04
- C28 1d impl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 1d impl dir0 max rel vs JAX, numpy serial: L_inf vs exact 9.54e-04
- C28 1d impl neu max rel vs JAX, numpy serial: L_inf vs exact 9.54e-04
- C28 1d impl dirL max rel vs JAX, numpy serial: L_inf vs exact 9.54e-04
- C28 1d impl per max rel vs JAX, numpy serial: L_inf vs exact 3.35e-03
- C28 1d impl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 1d impl steady max|u - L|, numpy serial: on the backend itself
- C28 quad expl dir0 L_inf order 32->64, numpy serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl neu L_inf order 32->64, numpy serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl dirL L_inf order 32->64, numpy serial: L_inf 4.70e-04 / 1.18e-04 / 2.95e-05
- C28 quad expl per L_inf order 32->64, numpy serial: L_inf 3.84e-04 / 9.75e-05 / 2.44e-05
- C28 quad expl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 quad expl dir0 max rel vs JAX, numpy serial: L_inf vs exact 4.70e-04
- C28 quad expl neu max rel vs JAX, numpy serial: L_inf vs exact 4.70e-04
- C28 quad expl dirL max rel vs JAX, numpy serial: L_inf vs exact 4.70e-04
- C28 quad expl per max rel vs JAX, numpy serial: L_inf vs exact 3.84e-04
- C28 quad expl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 quad expl steady max|u - L|, numpy serial: on the backend itself
- C28 quad impl dir0 L_inf order 32->64, numpy serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl neu L_inf order 32->64, numpy serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl dirL L_inf order 32->64, numpy serial: L_inf 1.13e-03 / 2.83e-04 / 7.10e-05
- C28 quad impl per L_inf order 32->64, numpy serial: L_inf 8.16e-04 / 2.06e-04 / 5.17e-05
- C28 quad impl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 quad impl dir0 max rel vs JAX, numpy serial: L_inf vs exact 1.13e-03
- C28 quad impl neu max rel vs JAX, numpy serial: L_inf vs exact 1.13e-03
- C28 quad impl dirL max rel vs JAX, numpy serial: L_inf vs exact 1.13e-03
- C28 quad impl per max rel vs JAX, numpy serial: L_inf vs exact 8.16e-04
- C28 quad impl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 quad impl steady max|u - L|, numpy serial: on the backend itself
- C28 tri expl dir0 L_inf order 32->64, numpy serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl neu L_inf order 32->64, numpy serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl dirL L_inf order 32->64, numpy serial: L_inf 8.15e-04 / 2.01e-04 / 5.00e-05
- C28 tri expl per L_inf order 32->64, numpy serial: L_inf 4.72e-04 / 1.15e-04 / 2.86e-05
- C28 tri expl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 tri expl dir0 max rel vs JAX, numpy serial: L_inf vs exact 8.15e-04
- C28 tri expl neu max rel vs JAX, numpy serial: L_inf vs exact 8.15e-04
- C28 tri expl dirL max rel vs JAX, numpy serial: L_inf vs exact 8.15e-04
- C28 tri expl per max rel vs JAX, numpy serial: L_inf vs exact 4.72e-04
- C28 tri expl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 tri expl steady max|u - L|, numpy serial: on the backend itself
- C28 tri impl dir0 L_inf order 32->64, numpy serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl neu L_inf order 32->64, numpy serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl dirL L_inf order 32->64, numpy serial: L_inf 8.25e-04 / 2.02e-04 / 5.00e-05
- C28 tri impl per L_inf order 32->64, numpy serial: L_inf 5.20e-04 / 1.27e-04 / 3.14e-05
- C28 tri impl steady max|u - L|, numpy serial: linear Dirichlet field kept to round-off
- C28 tri impl dir0 max rel vs JAX, numpy serial: L_inf vs exact 8.25e-04
- C28 tri impl neu max rel vs JAX, numpy serial: L_inf vs exact 8.25e-04
- C28 tri impl dirL max rel vs JAX, numpy serial: L_inf vs exact 8.25e-04
- C28 tri impl per max rel vs JAX, numpy serial: L_inf vs exact 5.20e-04
- C28 tri impl steady max rel vs JAX, numpy serial: L_inf vs exact 0.00e+00
- C28 tri impl steady max|u - L|, numpy serial: on the backend itself
- C29 p2 dir d1 boundary order 24->48, numpy serial: max err 2.01e-02 / 5.32e-03 / 1.35e-03; ghost as sample: 3.10e-02 / 1.21e-02 / 4.85e-03
- C29 p2 dir d2 boundary order 24->48, numpy serial: max err 4.95e-02 / 2.42e-02 / 1.27e-02; ghost as sample: 6.45e-02 / 5.33e-02 / 4.96e-02
- C29 p2 dir d1 interior order 24->48, numpy serial: max err 4.18e-02 / 1.10e-02 / 2.81e-03; ghost as sample: 4.18e-02 / 1.10e-02 / 2.81e-03
- C29 p2 dir d2 interior order 24->48, numpy serial: max err 1.40e-02 / 3.65e-03 / 9.20e-04; ghost as sample: 1.40e-02 / 3.65e-03 / 9.20e-04
- C29 p2 neu d1 boundary order 24->48, numpy serial: max err 2.19e-02 / 6.19e-03 / 1.61e-03; ghost as sample: 2.19e-02 / 6.19e-03 / 1.61e-03
- C29 p2 neu d2 boundary order 24->48, numpy serial: max err 4.95e-02 / 2.04e-02 / 1.08e-02; ghost as sample: 4.95e-02 / 2.04e-02 / 1.08e-02
- C29 p2 neu d1 interior order 24->48, numpy serial: max err 4.18e-02 / 1.10e-02 / 2.81e-03; ghost as sample: 4.18e-02 / 1.10e-02 / 2.81e-03
- C29 p2 neu d2 interior order 24->48, numpy serial: max err 1.40e-02 / 3.65e-03 / 9.20e-04; ghost as sample: 1.40e-02 / 3.65e-03 / 9.20e-04
- C29 p3 dir d1 boundary order 24->48, numpy serial: max err 1.04e-02 / 1.52e-03 / 1.98e-04; ghost as sample: 2.01e-02 / 7.59e-03 / 3.42e-03
- C29 p3 dir d2 boundary order 24->48, numpy serial: max err 3.12e-02 / 8.71e-03 / 2.25e-03; ghost as sample: 4.33e-02 / 4.61e-02 / 4.71e-02
- C29 p3 dir d3 boundary order 24->48, numpy serial: max err 6.32e-02 / 2.91e-02 / 1.53e-02; ghost as sample: 3.54e-02 / 9.11e-02 / 2.16e-01
- C29 p3 dir d1 interior order 24->48, numpy serial: max err 2.12e-03 / 2.43e-04 / 2.43e-05; ghost as sample: 2.12e-03 / 2.43e-04 / 2.43e-05
- C29 p3 dir d2 interior order 24->48, numpy serial: max err 2.07e-02 / 5.39e-03 / 1.42e-03; ghost as sample: 2.07e-02 / 5.39e-03 / 1.42e-03
- C29 p3 dir d3 interior order 24->48, numpy serial: max err 1.91e-02 / 8.77e-03 / 3.47e-03; ghost as sample: 1.91e-02 / 8.77e-03 / 3.47e-03
- C29 p3 neu d1 boundary order 24->48, numpy serial: max err 1.04e-02 / 1.52e-03 / 1.98e-04; ghost as sample: 1.04e-02 / 1.52e-03 / 1.98e-04
- C29 p3 neu d2 boundary order 24->48, numpy serial: max err 3.12e-02 / 8.71e-03 / 2.25e-03; ghost as sample: 3.12e-02 / 8.71e-03 / 2.25e-03
- C29 p3 neu d3 boundary order 24->48, numpy serial: max err 5.35e-02 / 2.54e-02 / 1.43e-02; ghost as sample: 5.71e-02 / 2.58e-02 / 1.46e-02
- C29 p3 neu d1 interior order 24->48, numpy serial: max err 2.12e-03 / 2.43e-04 / 2.43e-05; ghost as sample: 2.12e-03 / 2.43e-04 / 2.43e-05
- C29 p3 neu d2 interior order 24->48, numpy serial: max err 2.07e-02 / 5.39e-03 / 1.42e-03; ghost as sample: 2.07e-02 / 5.39e-03 / 1.42e-03
- C29 p3 neu d3 interior order 24->48, numpy serial: max err 1.91e-02 / 8.77e-03 / 3.47e-03; ghost as sample: 1.91e-02 / 8.77e-03 / 3.47e-03
- C34 q_x_0: vs the exact div(h tau') (mixed derivatives), order 32->64, numpy serial: max rel 3.55e-02 / 8.99e-03 / 2.26e-03; vs the SystemModel operator 3.55e-02 / 8.99e-03 / 2.26e-03; vs its diagonal-index part only 2.92e-01 / 3.20e-01 / 3.28e-01
- C34 q_y_0: vs the exact div(h tau') (mixed derivatives), order 32->64, numpy serial: max rel 5.83e-02 / 1.52e-02 / 3.85e-03; vs the SystemModel operator 5.83e-02 / 1.52e-02 / 3.85e-03; vs its diagonal-index part only 9.38e-01 / 1.04e+00 / 1.07e+00
- C34 nx 16 max rel |L - L_JAX|, numpy serial: vs the exact div(h tau') 3.55e-02 / 5.83e-02
- C30 gate 39 checks failing, numpy serial: PASS=72 FAIL=0 (tol 1e-09); the ghost as a plain sample is inexact in 15 cases (sensitivity)
- C31 SWE lie1 order dt/4->dt/8, numpy serial: L1(h) vs dt/32 3.87e-05 / 1.86e-05 / 8.66e-06 / 3.71e-06
- C31 SWE lie2 order dt/4->dt/8, numpy serial: L1(h) vs dt/32 5.63e-06 / 2.71e-06 / 1.26e-06 / 5.39e-07
- C31 SWE ssp2 order dt/4->dt/8, numpy serial: L1(h) vs dt/32 3.31e-07 / 8.21e-08 / 2.02e-08 / 4.81e-09
- C31 VAM t1 order (Richardson), numpy serial: L1(h) of dt/2^k - dt/2^(k+1) 8.22e-06 / 3.98e-06 / 1.95e-06
- C31 VAM t2 order (Richardson), numpy serial: L1(h) of dt/2^k - dt/2^(k+1) 4.41e-07 / 1.71e-07 / 7.85e-08
- C31 VAM ssp2 order (Richardson), numpy serial: L1(h) of dt/2^k - dt/2^(k+1) 3.76e-07 / 9.59e-08 / 2.47e-08
- C31 VAM predictor alone ssp2 order (Richardson), numpy serial: L1(h) of dt/2^k - dt/2^(k+1) 5.77e-07 / 1.42e-07 / 3.53e-08
- C35 c1: misspelt restart_from fails at once, numpy serial: after 0.6 s: restart_from '/Users/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_numpy_serial/numpy/serial/C35/c1_half/stat.h5': no such file; did you mean 'state.h5'?
- C35 c8: misspelt restart_from fails at once, numpy serial: after 4.9 s: restart_from '/Users/adam-obbpb5az1dhsjzf/git/wt-numfix-matrix/work_numpy_serial/numpy/serial/C35/c8_half/stat.h5': no such file; did you mean 'state.h5'?
- C32 noslip L_inf order 40->80, numpy serial: L_inf 3.36e-04 / 9.16e-05 / 2.38e-05; order 20->40 1.88
- C32 noslip n20 max|h - 1|, numpy serial: theory: h stays 1
- C32 noslip n20 max|q_y|, numpy serial: theory: v stays 0
- C32 noslip n20 x-variation of q_x (max rel), numpy serial: theory: u = u(y)
- C32 free L_inf order 40->80, numpy serial: L_inf 3.36e-04 / 9.16e-05 / 2.38e-05; order 20->40 1.87
- C32 free n20 max|h - 1|, numpy serial: theory: h stays 1
- C32 free n20 max|q_y|, numpy serial: theory: v stays 0
- C32 free n20 x-variation of q_x (max rel), numpy serial: theory: u = u(y)
- C32 noslip_split L_inf order 40->80, numpy serial: L_inf 3.37e-04 / 9.17e-05 / 2.38e-05; order 20->40 1.88
- C32 noslip_split n20 max|h - 1|, numpy serial: theory: h stays 1
- C32 noslip_split n20 max|q_y|, numpy serial: theory: v stays 0
- C32 noslip_split n20 x-variation of q_x (max rel), numpy serial: theory: u = u(y)
- C32 control: no-slip profile on free-slip walls, L_inf vs the no-slip exact, numpy serial: the walls act
- C32 nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction), numpy serial: Dirichlet(momentum) also sets the Riemann outer state: Rusanov damps the wall cells
- C32 nu 0 on no-slip walls: wall friction, rel. change of sum(q_x vol), numpy serial: the interior fluxes cancel: the walls' share
- C32 nu 0 on split walls: wall friction, rel. change of sum(q_x vol), numpy serial: Wall face: the reflected Riemann state carries no x-momentum flux
- C32 nu 0 on split walls: max|u - u0|, numpy serial: the interior Rusanov dissipation of the O2 face jumps, piled up in the wall cells
- C33 imex q(T) vs expm, order dt/2 -> dt/4, numpy serial: rel err 2.02e-03 / 1.02e-03 / 5.11e-04 (dt 0.05 / 2^k); implicit Euler (I - dt J)^-N
- C33 imex q(T) vs its own scheme, implicit Euler (I - dt J)^-N (max rel), numpy serial: the integrator in use, to round-off / the implicit tolerance
- C33 imex uniform state: max|Q - Q[:, 0]|, |h - 1|, numpy serial: theory: no flux divergence, h stays
- C33 rk2 q(T) vs expm, order dt/2 -> dt/4, numpy serial: rel err 5.71e-05 / 1.36e-05 / 3.34e-06 (dt 0.05 / 2^k); SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N
- C33 rk2 q(T) vs its own scheme, SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N (max rel), numpy serial: the integrator in use, to round-off / the implicit tolerance
- C33 rk2 uniform state: max|Q - Q[:, 0]|, |h - 1|, numpy serial: theory: no flux divergence, h stays
- C6 C1 mass change (rel), numpy serial: max over 6 runs
- C6 C2 mass change (rel), numpy serial: max over 2 runs
- C6 C3 mass change (rel), numpy serial: max over 2 runs
- C6 C5 mass change (rel), numpy serial: max over 1 runs
- C6 C11 mass change (rel), numpy serial: max over 6 runs
- C6 C12 mass change (rel), numpy serial: max over 6 runs
- C6 C13 mass change (rel), numpy serial: max over 4 runs
- C6 C14 mass change (rel), numpy serial: max over 6 runs
- C6 C20 mass change (rel), numpy serial: max over 2 runs
- C6 C23 mass change (rel), numpy serial: max over 6 runs
- C6 C32 mass change (rel), numpy serial: max over 12 runs
- C6 C33 mass change (rel), numpy serial: max over 6 runs
- C6 C36 mass change (rel), numpy serial: max over 1 runs

Summed run wall time (s, without process start-up): amrex 5391, foam 4796, jax 8716, numpy 2913
