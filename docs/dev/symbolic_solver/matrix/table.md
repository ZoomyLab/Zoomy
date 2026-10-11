| case | check | backend | serial | mpi2 | mpi4 | gpu1 | threshold (source) |
|---|---|---|---|---|---|---|---|
| C8 | vam runs | amrex | nan **FAIL** | nan **FAIL** | nan **FAIL** | nan **FAIL** | <= 0 (the run completes) |
| C1 | O1 order 80->160 | jax | 8.23e-01 PASS | - | - | - | >= 0.75 (observed: JAX 0.82, pre-asymptotic (spec 0.9)) |
| C1 | O2 order 80->160 | jax | 2.37e+00 PASS | - | - | - | >= 1.8 (spec (theory 2)) |
| C2 | L1 order 40->80 | jax | 9.23e-01 PASS | - | - | - | >= 0.6 (spec) |
| C3 | O1 max|u| after 100 steps | jax | 3.41e-16 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | jax | 9.84e-16 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 order 100->200 | jax | 8.56e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | sub O1 L1(h) n200 | jax | 3.70e-03 PASS | - | - | - | <= 0.0075 (observed 3.7e-3) |
| C4 | sub O1 max|q/q_in - 1| n200 | jax | 9.47e-03 PASS | - | - | - | <= 0.019 (observed 9.5e-3) |
| C4 | sub O1 max|q_y| n200 (info) | jax | 8.07e-16 | - | - | - | - |
| C4 | sub O2 order 100->200 | jax | 2.77e+00 PASS | - | - | - | >= 1.8 (theory 2 (observed 2.8)) |
| C4 | sub O2 L1(h) n200 | jax | 5.37e-05 PASS | - | - | - | <= 0.00011 (observed 5.4e-5) |
| C4 | sub O2 max|q/q_in - 1| n200 | jax | 8.20e-04 PASS | - | - | - | <= 0.0017 (observed 8.3e-4) |
| C4 | sub O2 max|q_y| n200 (info) | jax | 5.65e-04 | - | - | - | - |
| C4 | trans O1 order 100->200 | jax | 8.60e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | trans O1 L1(h) n200 | jax | 3.05e-02 PASS | - | - | - | <= 0.06 (observed 3.1e-2) |
| C4 | trans O1 max|q/q_in - 1| n200 | jax | 3.48e-02 PASS | - | - | - | <= 0.07 (observed 3.5e-2) |
| C4 | trans O1 max|q_y| n200 (info) | jax | 1.01e-16 | - | - | - | - |
| C4 | trans O2 order 100->200 | jax | 1.12e+00 PASS | - | - | - | >= 0.9 (observed 1.12 (sonic point)) |
| C4 | trans O2 L1(h) n200 | jax | 1.55e-04 PASS | - | - | - | <= 0.00031 (observed 1.6e-4) |
| C4 | trans O2 max|q/q_in - 1| n200 | jax | 4.72e-04 PASS | - | - | - | <= 0.00095 (observed 4.7e-4) |
| C4 | trans O2 max|q_y| n200 (info) | jax | 6.08e-04 | - | - | - | - |
| C5 | h_min | jax | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C6 | C1 mass change (rel) | jax | 2.82e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | jax | 1.41e-16 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | jax | 0.00e+00 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | jax | 1.95e-16 PASS | 1.95e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 parallel vs serial (max rel) | jax | - | 1.29e-15 PASS | 1.31e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 parallel vs serial (max rel) | jax | - | 5.25e-15 PASS | 5.61e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 GPU vs CPU (max rel) | jax | - | - | - | 4.94e-15 PASS | <= 1e-12 (tolerance (round-off)) |
| C8 | C8 parallel vs serial (max rel) | jax | - | 1.14e-10 PASS | 1.19e-10 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol, over the steps)) |
| C1 | O1 order 80->160 | foam | 8.23e-01 PASS | - | - | - | >= 0.75 (observed: JAX 0.82, pre-asymptotic (spec 0.9)) |
| C1 | O1 n80 max rel vs JAX (info) | foam | 8.60e-16 | - | - | - | - |
| C1 | O2 order 80->160 | foam | 2.35e+00 PASS | - | - | - | >= 1.8 (spec (theory 2)) |
| C1 | O2 n80 max rel vs JAX (info) | foam | 3.34e-04 | - | - | - | - |
| C2 | L1 order 40->80 | foam | 9.05e-01 PASS | - | - | - | >= 0.6 (spec) |
| C2 | L1 vs JAX n80 (rel) | foam | 7.11e-04 PASS | - | - | - | <= 0.0015 (observed (foam 7.1e-4)) |
| C3 | O1 max|u| after 100 steps | foam | 4.44e-16 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | foam | 1.35e-15 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 order 100->200 | foam | 8.56e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | sub O1 L1(h) n200 | foam | 3.70e-03 PASS | - | - | - | <= 0.0075 (observed 3.7e-3) |
| C4 | sub O1 max|q/q_in - 1| n200 | foam | 9.47e-03 PASS | - | - | - | <= 0.019 (observed 9.5e-3) |
| C4 | sub O1 max|q_y| n200 (info) | foam | 6.35e-16 | - | - | - | - |
| C4 | sub O2 order 100->200 | foam | 2.79e+00 PASS | - | - | - | >= 1.8 (theory 2 (observed 2.8)) |
| C4 | sub O2 L1(h) n200 | foam | 5.42e-05 PASS | - | - | - | <= 0.00011 (observed 5.4e-5) |
| C4 | sub O2 max|q/q_in - 1| n200 | foam | 8.30e-04 PASS | - | - | - | <= 0.0017 (observed 8.3e-4) |
| C4 | sub O2 max|q_y| n200 (info) | foam | 9.40e-16 | - | - | - | - |
| C4 | trans O1 order 100->200 | foam | 8.60e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | trans O1 L1(h) n200 | foam | 3.05e-02 PASS | - | - | - | <= 0.06 (observed 3.1e-2) |
| C4 | trans O1 max|q/q_in - 1| n200 | foam | 3.48e-02 PASS | - | - | - | <= 0.07 (observed 3.5e-2) |
| C4 | trans O1 max|q_y| n200 (info) | foam | 1.40e-16 | - | - | - | - |
| C4 | trans O2 order 100->200 | foam | 1.12e+00 PASS | - | - | - | >= 0.9 (observed 1.12 (sonic point)) |
| C4 | trans O2 L1(h) n200 | foam | 1.55e-04 PASS | - | - | - | <= 0.00031 (observed 1.6e-4) |
| C4 | trans O2 max|q/q_in - 1| n200 | foam | 4.72e-04 PASS | - | - | - | <= 0.00095 (observed 4.7e-4) |
| C4 | trans O2 max|q_y| n200 (info) | foam | 3.67e-16 | - | - | - | - |
| C5 | h_min | foam | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | >= 0 (theory (positivity)) |
| C5 | L1 vs JAX (rel) | foam | 1.69e-02 | - | - | - | - |
| C8 | L1 vs JAX (rel) | foam | 1.57e-02 | - | - | - | - |
| C6 | C1 mass change (rel) | foam | 1.41e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | foam | 1.41e-16 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | foam | 0.00e+00 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | foam | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 parallel vs serial (max rel) | foam | - | 1.37e-15 PASS | 1.39e-15 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 parallel vs serial (max rel) | foam | - | 1.27e-14 PASS | 9.19e-14 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C8 | C8 parallel vs serial (max rel) | foam | - | 3.55e-11 PASS | 6.85e-11 PASS | - | <= 1e-09 (tolerance (10 x pressure rtol, over the steps)) |
| C1 | O1 order 80->160 | amrex | 8.23e-01 PASS | - | - | - | >= 0.75 (observed: JAX 0.82, pre-asymptotic (spec 0.9)) |
| C1 | O1 n80 max rel vs JAX (info) | amrex | 8.60e-16 | - | - | - | - |
| C1 | O2 order 80->160 | amrex | 2.37e+00 PASS | - | - | - | >= 1.8 (spec (theory 2)) |
| C1 | O2 n80 max rel vs JAX (info) | amrex | 1.39e-15 | - | - | - | - |
| C2 | L1 order 40->80 | amrex | 9.23e-01 PASS | - | - | - | >= 0.6 (spec) |
| C2 | L1 vs JAX n80 (rel) | amrex | 3.09e-16 PASS | - | - | - | <= 0.0015 (observed (foam 7.1e-4)) |
| C3 | O1 max|u| after 100 steps | amrex | 0.00e+00 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C3 | O2 max|u| after 100 steps | amrex | 9.19e-16 PASS | - | - | - | <= 1e-12 (spec (well-balanced: round-off)) |
| C4 | sub O1 order 100->200 | amrex | 8.56e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | sub O1 L1(h) n200 | amrex | 3.70e-03 PASS | - | - | - | <= 0.0075 (observed 3.7e-3) |
| C4 | sub O1 max|q/q_in - 1| n200 | amrex | 9.47e-03 PASS | - | - | - | <= 0.019 (observed 9.5e-3) |
| C4 | sub O1 max|q_y| n200 (info) | amrex | 4.41e-16 | - | - | - | - |
| C4 | sub O2 order 100->200 | amrex | 2.78e+00 PASS | - | - | - | >= 1.8 (theory 2 (observed 2.8)) |
| C4 | sub O2 L1(h) n200 | amrex | 5.41e-05 PASS | - | - | - | <= 0.00011 (observed 5.4e-5) |
| C4 | sub O2 max|q/q_in - 1| n200 | amrex | 8.29e-04 PASS | - | - | - | <= 0.0017 (observed 8.3e-4) |
| C4 | sub O2 max|q_y| n200 (info) | amrex | 5.53e-04 | - | - | - | - |
| C4 | trans O1 order 100->200 | amrex | 8.60e-01 PASS | - | - | - | >= 0.75 (observed 0.86 (theory 1)) |
| C4 | trans O1 L1(h) n200 | amrex | 3.05e-02 PASS | - | - | - | <= 0.06 (observed 3.1e-2) |
| C4 | trans O1 max|q/q_in - 1| n200 | amrex | 3.48e-02 PASS | - | - | - | <= 0.07 (observed 3.5e-2) |
| C4 | trans O1 max|q_y| n200 (info) | amrex | 1.04e-16 | - | - | - | - |
| C4 | trans O2 order 100->200 | amrex | 1.12e+00 PASS | - | - | - | >= 0.9 (observed 1.12 (sonic point)) |
| C4 | trans O2 L1(h) n200 | amrex | 1.55e-04 PASS | - | - | - | <= 0.00031 (observed 1.6e-4) |
| C4 | trans O2 max|q/q_in - 1| n200 | amrex | 4.72e-04 PASS | - | - | - | <= 0.00095 (observed 4.7e-4) |
| C4 | trans O2 max|q_y| n200 (info) | amrex | 6.08e-04 | - | - | - | - |
| C5 | h_min | amrex | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | >= 0 (theory (positivity)) |
| C5 | L1 vs JAX (rel) | amrex | 6.14e-10 | - | - | - | - |
| C6 | C1 mass change (rel) | amrex | 2.82e-16 PASS | 0.00e+00 PASS | 0.00e+00 PASS | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C6 | C2 mass change (rel) | amrex | 1.41e-16 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C3 mass change (rel) | amrex | 0.00e+00 PASS | - | - | - | <= 1e-12 (tolerance (round-off)) |
| C6 | C5 mass change (rel) | amrex | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | 1.95e-16 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C1 parallel vs serial (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C1 GPU vs CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |
| C7 | C5 parallel vs serial (max rel) | amrex | - | 0.00e+00 PASS | 0.00e+00 PASS | - | <= 1e-12 (tolerance (round-off)) |
| C9 | C5 GPU vs CPU (max rel) | amrex | - | - | - | 0.00e+00 PASS | <= 1e-12 (tolerance (round-off)) |

Failures:
- C8 vam runs, amrex serial: nan (need <= 0) /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/serial/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C8 vam runs, amrex mpi2: nan (need <= 0) /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/mpi2/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C8 vam runs, amrex mpi4: nan (need <= 0) /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/mpi4/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C8 vam runs, amrex gpu1: nan (need <= 0) /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/gpu1/C8/vam.err: ../Source/Solver.H(7): error: identifier "nsm" is undefined

Notes:
- C8 vam runs, amrex gpu1: /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/gpu1/C8/vam.err: ../Source/Solver.H(7): error: identifier "nsm" is undefined
- C8 vam runs, amrex mpi2: /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/mpi2/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C8 vam runs, amrex mpi4: /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/mpi4/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C8 vam runs, amrex serial: /mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/wt-matrix/docs/dev/symbolic_solver/matrix/store/amrex/serial/C8/vam.err: /usr/include/c++/13/tuple:2302:27: error: no matching function for call to '__invoke(zs::<lambda(auto:14, auto:15, auto:16, auto:17, auto:18, auto:19, auto:20)>::<lambda(auto:26, auto:27, auto:28, aut
- C1 O1 order 80->160, jax serial: L1(h) 1.18e-03 / 7.19e-04 / 4.07e-04
- C1 O2 order 80->160, jax serial: L1(h) 4.70e-04 / 1.17e-04 / 2.26e-05; corrected for the reference's error 2.06
- C2 L1 order 40->80, jax serial: L1(h) 2.29e-03 / 1.21e-03; corrected for the reference's error 0.58
- C4 sub O1 order 100->200, jax serial: L1(h) 1.11e-02 / 6.70e-03 / 3.70e-03
- C4 sub O1 max|q/q_in - 1| n200, jax serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 sub O2 order 100->200, jax serial: L1(h) 2.18e-03 / 3.68e-04 / 5.37e-05
- C4 sub O2 max|q/q_in - 1| n200, jax serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O1 order 100->200, jax serial: L1(h) 9.35e-02 / 5.53e-02 / 3.05e-02
- C4 trans O1 max|q/q_in - 1| n200, jax serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O2 order 100->200, jax serial: L1(h) 2.57e-03 / 3.38e-04 / 1.55e-04
- C4 trans O2 max|q/q_in - 1| n200, jax serial: steady state: discharge = inflow (the open-domain mass balance)
- C6 C1 mass change (rel), jax serial: max over 6 runs
- C6 C1 mass change (rel), jax mpi2: max over 1 runs
- C6 C1 mass change (rel), jax mpi4: max over 1 runs
- C6 C2 mass change (rel), jax serial: max over 2 runs
- C6 C3 mass change (rel), jax serial: max over 2 runs
- C6 C5 mass change (rel), jax serial: max over 1 runs
- C6 C5 mass change (rel), jax mpi2: max over 1 runs
- C6 C5 mass change (rel), jax mpi4: max over 1 runs
- C6 C5 mass change (rel), jax gpu1: max over 1 runs
- C1 O1 order 80->160, foam serial: L1(h) 1.18e-03 / 7.19e-04 / 4.07e-04
- C1 O2 order 80->160, foam serial: L1(h) 4.45e-04 / 1.11e-04 / 2.19e-05; corrected for the reference's error 2.03
- C2 L1 order 40->80, foam serial: L1(h) 2.11e-03 / 1.12e-03; corrected for the reference's error 0.55
- C4 sub O1 order 100->200, foam serial: L1(h) 1.11e-02 / 6.70e-03 / 3.70e-03
- C4 sub O1 max|q/q_in - 1| n200, foam serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 sub O2 order 100->200, foam serial: L1(h) 2.17e-03 / 3.74e-04 / 5.42e-05
- C4 sub O2 max|q/q_in - 1| n200, foam serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O1 order 100->200, foam serial: L1(h) 9.35e-02 / 5.53e-02 / 3.05e-02
- C4 trans O1 max|q/q_in - 1| n200, foam serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O2 order 100->200, foam serial: L1(h) 2.57e-03 / 3.38e-04 / 1.55e-04
- C4 trans O2 max|q/q_in - 1| n200, foam serial: steady state: discharge = inflow (the open-domain mass balance)
- C6 C1 mass change (rel), foam serial: max over 6 runs
- C6 C1 mass change (rel), foam mpi2: max over 1 runs
- C6 C1 mass change (rel), foam mpi4: max over 1 runs
- C6 C2 mass change (rel), foam serial: max over 2 runs
- C6 C3 mass change (rel), foam serial: max over 2 runs
- C6 C5 mass change (rel), foam serial: max over 1 runs
- C6 C5 mass change (rel), foam mpi2: max over 1 runs
- C6 C5 mass change (rel), foam mpi4: max over 1 runs
- C1 O1 order 80->160, amrex serial: L1(h) 1.18e-03 / 7.19e-04 / 4.07e-04
- C1 O2 order 80->160, amrex serial: L1(h) 4.70e-04 / 1.17e-04 / 2.26e-05; corrected for the reference's error 2.06
- C2 L1 order 40->80, amrex serial: L1(h) 2.29e-03 / 1.21e-03; corrected for the reference's error 0.58
- C4 sub O1 order 100->200, amrex serial: L1(h) 1.11e-02 / 6.70e-03 / 3.70e-03
- C4 sub O1 max|q/q_in - 1| n200, amrex serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 sub O2 order 100->200, amrex serial: L1(h) 2.18e-03 / 3.71e-04 / 5.41e-05
- C4 sub O2 max|q/q_in - 1| n200, amrex serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O1 order 100->200, amrex serial: L1(h) 9.35e-02 / 5.53e-02 / 3.05e-02
- C4 trans O1 max|q/q_in - 1| n200, amrex serial: steady state: discharge = inflow (the open-domain mass balance)
- C4 trans O2 order 100->200, amrex serial: L1(h) 2.57e-03 / 3.38e-04 / 1.55e-04
- C4 trans O2 max|q/q_in - 1| n200, amrex serial: steady state: discharge = inflow (the open-domain mass balance)
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
