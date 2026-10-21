# Extending

## A backend operation (block)

1. Declare it in `zoomy_core/symsolver/vocabulary.py` (or `core.py` for
   control flow): a `Block` subclass whose docstring is the contract
   (arguments, shapes, boundary behaviour).
2. Use it in a template or a discretization component.
3. Implement it under the same name in every runtime: an entry of the table
   returned by `zoomy_jax/symsolver/runtime.py: runtime()`, an `inline`
   function in `zoomy_foam/symsolver/rt.H` and in
   `zoomy_amrex/Source/symsolver/rt_amrex.H`.  A backend that cannot provide
   it raises `NotImplementedError` (the matrix records "unsupported").
4. Add or extend a matrix case that exercises it on every backend.

## A test-matrix case

The matrix is `docs/dev/symbolic_solver/matrix/` (README there).

1. `cases.py`: a builder returning `(mesh, nsm)` (Chorin:
   `(mesh, (pred, press, corr))`) and an entry `"Cnn": (title, [Run(...)])`
   in `CASES`; `Run` fixes `t_end`, `cfl`, `dt`, `mood`, `imex`,
   `time_order`, the configs and the backends.
2. `matrix.py`: the checks in `measure()`, each
   `add(case, check, backend, config, value, threshold, note)` with a
   threshold `(op, value, source)`; the source says where the number comes
   from (theory, tolerance, spec, observed).
3. Run and tabulate, then refresh the published results:

```bash
cd docs/dev/symbolic_solver/matrix && source env.sh
python matrix.py run jax Cnn          # -> store/jax/serial/Cnn/*.npz; foam, amrex alike
python matrix.py table                # -> table.md
cd - && MATRIX_STORE=docs/dev/symbolic_solver/matrix/store \
    python docs/scripts/generate_symsolver_results.py --from-store
```

`--from-store` rewrites `tests/results/symsolver/matrix_latest.json`;
commit it, and the [results page](tests.md) follows on the
next docs build.

## A timing case

`tests/symsolver/timing_cases.py`: a `TimingCase` (a matrix `Run`, a step
count and a dyadic `dt`, the configs per backend, the agreement checks
against JAX).  `pytest tests/symsolver -m benchmark` appends a record to
`tests/results/symsolver/timings.jsonl` and rewrites `timings_latest.json`
(see `tests/symsolver/README.md`).

## A backend

A runtime (the blocks), a kernel printer and a driver as in
[Templates and printers](templates.md), then one adapter class in
`docs/dev/symbolic_solver/matrix/backends.py` (`configs`, `launch`, `run`).
The backend is accepted when every matrix row passes for it; the checks know
nothing about the backend.
