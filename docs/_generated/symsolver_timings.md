Per case, backend and config: the march time per cell and step (µs, of the fastest of the repetitions), the min / median march time (s) of the repetitions, ms per step, and setup / compile time (s) of the first repetition.

| case | backend | config | device | cells | steps | µs/cell/step | min / median (s) | ms/step | setup / compile (s) | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `swe_o1` | jax | gpu1 | gpu | 40000 | 50 | 0.0515 | 0.103 / 0.103 | 2.06 | 23.3 / 0.944 | ok |
| `swe_o1` | jax | serial | cpu | 40000 | 50 | 1.01 | 2.02 / 2.03 | 40.5 | 8.9 / 0.457 | ok |
| `swe_o1` | foam | serial | - | 40000 | 50 | 2.15 | 4.3 / 4.3 | 85.9 | 7.15 / 1.64 | ok |
| `swe_o1` | amrex | gpu1 | - | 40000 | 50 | 0.0189 | 0.0378 / 0.0378 | 0.756 | 6.01 / 23.5 | ok |
| `swe_o1` | amrex | serial | - | 40000 | 50 | 1.3 | 2.59 / 2.6 | 51.9 | 5.13 / 1.7 | ok |
| `swe_o2` | jax | gpu1 | gpu | 40000 | 50 | 0.0698 | 0.14 / 0.14 | 2.79 | 9.56 / 1.27 | ok |
| `swe_o2` | jax | serial | cpu | 40000 | 50 | 2.33 | 4.65 / 4.65 | 93 | 6.82 / 1.29 | ok |
| `swe_o2` | foam | serial | - | 40000 | 50 | 5.46 | 10.9 / 10.9 | 218 | 7.2 / 9.92 | ok |
| `swe_o2` | amrex | gpu1 | - | 40000 | 50 | 0.0427 | 0.0853 / 0.0853 | 1.71 | 6 / 24.8 | ok |
| `swe_o2` | amrex | serial | - | 40000 | 50 | 3.24 | 6.48 / 6.49 | 130 | 5.11 / 10.4 | ok |
| `vam1_chorin` | jax | gpu1 | gpu pressure=amgx | 4000 | 8 | 5.68 | 0.182 / 0.182 | 22.7 | 41.5 / 19.6 | ok |
| `vam1_chorin` | jax | serial | cpu pressure=pyamg | 4000 | 8 | 21.9 | 0.7 / 0.734 | 87.5 | 57.1 / 12.6 | ok |
| `vam1_chorin` | foam | serial | - | 4000 | 8 | 122 | 3.9 / 3.9 | 487 | 10.7 / 50.3 | ok |
| `vam1_chorin` | amrex | gpu1 | - | 4000 | 8 | 141 | 4.51 / 4.51 | 563 | 10.3 / 67.3 | ok |
| `vam1_chorin` | amrex | serial | - | 4000 | 8 | 214 | 6.83 / 6.83 | 854 | 9.67 / 51.9 | ok |
| `imex_swe` | jax | gpu1 | gpu | 10000 | 50 | 0.174 | 0.0869 / 0.0869 | 1.74 | 8.76 / 5.69 | ok |
| `imex_swe` | jax | serial | cpu | 10000 | 50 | 2.87 | 1.43 / 1.44 | 28.7 | 8.38 / 6.09 | ok |
| `imex_swe` | foam | serial | - | 10000 | 50 | 10.5 | 5.26 / 5.26 | 105 | 5.87 / 10.1 | ok |
| `imex_swe` | amrex | gpu1 | - | 10000 | 50 | 0.143 | 0.0715 / 0.0715 | 1.43 | 5.55 / 25.5 | ok |
| `imex_swe` | amrex | serial | - | 10000 | 50 | 3.99 | 1.99 / 2 | 39.9 | 4.98 / 10.8 | ok |
| `euler3d_o1` | jax | gpu1 | gpu | 64000 | 40 | 0.00396 | 0.0101 / 0.0102 | 0.253 | 6.17 / 0.588 | ok |
| `euler3d_o1` | jax | serial | cpu | 64000 | 40 | 0.355 | 0.91 / 0.91 | 22.7 | 5.71 / 0.875 | ok |
| `euler3d_o1` | foam | serial | - | 64000 | 40 | 1.64 | 4.2 / 4.2 | 105 | 7.14 / 8.49 | ok |
| `euler3d_o1` | amrex | gpu1 | - | 64000 | 40 | 0.00361 | 0.00924 / 0.00925 | 0.231 | 2.81 / 23.8 | ok |
| `euler3d_o1` | amrex | serial | - | 64000 | 40 | 0.383 | 0.98 / 0.983 | 24.5 | 2.21 / 8.94 | ok |
| `euler3d_o2` | jax | gpu1 | gpu | 64000 | 40 | 0.0397 | 0.102 / 0.102 | 2.54 | 5.39 / 6.83 | ok |
| `euler3d_o2` | jax | serial | cpu | 64000 | 40 | 2.4 | 6.15 / 6.16 | 154 | 4.18 / 1.92 | ok |
| `euler3d_o2` | foam | serial | - | 64000 | 40 | 4.01 | 10.3 / 10.3 | 257 | 7.11 / 8.06 | ok |
| `euler3d_o2` | amrex | gpu1 | - | 64000 | 40 | 0.0172 | 0.0441 / 0.0441 | 1.1 | 2.82 / 24.2 | ok |
| `euler3d_o2` | amrex | serial | - | 64000 | 40 | 3.13 | 8 / 8.01 | 200 | 2.22 / 8.92 | ok |
| `diffusion_impl` | jax | gpu1 | gpu | 2304 | 10 | 1.78 | 0.0409 / 0.0412 | 4.09 | 22.1 / 18 | ok |
| `diffusion_impl` | jax | serial | cpu | 2304 | 10 | 9.5 | 0.219 / 0.219 | 21.9 | 22.7 / 13 | ok |
| `diffusion_impl` | foam | serial | - | 2304 | 10 | 659 | 15.2 / 15.4 | 1.52e+03 | 15 / 16.3 | ok |
| `diffusion_impl` | amrex | gpu1 | - | 2304 | 10 | 3.31 | 0.0762 / 0.0762 | 7.62 | 15 / 34.3 | ok |
| `diffusion_impl` | amrex | serial | - | 2304 | 10 | 14.5 | 0.333 / 0.335 | 33.3 | 14.1 / 17.5 | ok |

Machines (records grouped by host, hardware, pinning and library commits):

| host | CPU | GPU | cores pinned (cpus) | wave speed | libraries | dates |
|---|---|---|---|---|---|---|
| 68G-gaia | INTEL(R) XEON(R) PLATINUM 8568Y+ | NVIDIA L40S | 1 (44) | power | superrepo c02eed4f, zoomy_amrex 4a7067d, zoomy_core 2f2ac62, zoomy_foam a71bff0, zoomy_jax 276cedb | 2026-10-12 .. 2026-10-12 |
| 68G-gaia | INTEL(R) XEON(R) PLATINUM 8568Y+ | NVIDIA L40S | 1 (44) | power | superrepo 9a5f303f, zoomy_amrex 0270878, zoomy_core 6cbf885, zoomy_foam 250b152, zoomy_jax 4b2ce3f | 2026-10-12 .. 2026-10-12 |
