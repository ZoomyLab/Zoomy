# Chat Summary: Derivative Workflow, Analysis, and IMEX Prototype

## Scope Covered

This chat focused on:

- redesigning model-side variable/derivative access (`Q`, `D`, `Qaux` mapping),
- building runnable SWE/GN tutorial scripts with printed symbolic model functions,
- adding symbolic linear analysis tooling (ansatz, linear system, phase velocity),
- prototyping IMEX source treatment in NumPy/SciPy,
- comparing matrix-free FD-Jv vs analytic chain-rule Jv.

---

## Main Additions

## 1) Derivative-aware model workflow (compatibility-first)

Added:

- `library/zoomy_core/zoomy_core/model/derivative_workflow.py`

Key elements:

- `StructuredDerivativeModel`:
  - named field access (`self.Q.h`, `self.Q.hu`, etc.),
  - derivative declarations via `DerivativeSpec`,
  - auto-derivative inference from `self.D.*(...)` calls (`auto_requested_derivatives=True`),
  - derivative key canonicalization strategy (default `time_first`).
- `DerivativeNamespace` (`D`):
  - `dt`, `dx`, `dxx`, `dtxx`, `diff`.
- `DerivativeAwareSolverMixin` + `DerivativeAwareSolver`:
  - computes declared derivatives and fills `Qaux` consistently.

Notes:

- Derivatives are exposed as model-side symbols, then mapped to `Qaux` slots for runtime compatibility.
- Auto-inference now supports direct `self.D.*(self.Q.field)` and alias style (`u = self.Q.u; self.D.*(u)`).

---

## 2) Generic model function printing moved to base model

Added method:

- `Model.print_model_functions(...)` in `library/zoomy_core/zoomy_core/model/basemodel.py`

Outcome:

- Function printing is now generic (not tied to derivative workflow),
- wrappers remain compatible.

---

## 3) Numpy transformation robustness/performance changes

Updated:

- `library/zoomy_core/zoomy_core/transformation/to_numpy.py`

Changes:

- robust handling for empty arrays in vectorization/lambdify path,
- enabled symbolic CSE at lambdify (`cse=True`, with fallback).

---

## 4) Tutorial scripts created/updated

Added/updated:

- `tutorials/swe/simple_swe_v2.py`
- `tutorials/swe/simple_gn_v2.py`
- `tutorials/swe/simple_gn_classical_sim_v2.py`
- `tutorials/swe/compare_three_models_v2.py`
- `tutorials/swe/gn_linear_analysis_v2.py`
- `tutorials/swe/gn_classical_linear_analysis_v2.py`

Capabilities added:

- print symbolic `flux`/`source`,
- run simulation on 1D mesh,
- save side-by-side matplotlib plots (`h` and `u`, initial vs final),
- three-model comparison plot.

---

## 5) Symbolic linear analysis tool

Added:

- `library/zoomy_core/zoomy_core/model/analysis_linear.py`

Features:

- build equations with ansatz substitution,
- resolve derivative symbols (including mixed derivatives),
- linearize (`O(eps)`),
- build matrix system and solve dispersion relation,
- compute phase velocity (`c = omega/k`),
- print intermediate expressions and solved systems.

Also added generic two-step pipeline:

- `linearize_from_quasilinear(...)`
- `solve_phase_velocity_from_linearized(...)`

with optional user-provided `assumptions`.

---

## 6) IMEX source prototype (NumPy/SciPy)

Added:

- `library/zoomy_core/zoomy_core/fvm/jvp_numpy.py`
- `library/zoomy_core/zoomy_core/fvm/solver_imex_numpy.py`
- `tutorials/swe/compare_jvp_fd_analytic_v2.py`
- `tutorials/swe/benchmark_imex_source_modes_v2.py`

Implemented:

- source mode selection:
  - `local`: cell-local implicit source solve,
  - `global`: Newton-Krylov source solve (matrix-free operator),
  - `auto`: chooses based on derivative-coupled source usage.
- two Jv backends for global source solve:
  - FD-Jv,
  - analytic chain-rule Jv.

Benchmark output now includes timing split:

- `total_time_s`,
- `init_time_s`,
- `runtime_only_s`,
- `implicit_time_s`,
- plus step/call counters and finite ratio.

---

## Key Technical Findings

- Chain rule is essential when `Qaux` depends on `Q`:
  - total source Jacobian uses `dS/dQ = ∂S/∂Q + ∂S/∂Qaux * dQaux/dQ`.
- Verification script (`compare_jvp_fd_analytic_v2.py`) showed:
  - lagged/partial-only analytic Jv vs FD-Jv: large mismatch,
  - full-chain analytic Jv vs FD-Jv: close agreement (~1e-8 relative error).
- In the current Python implementation, FD-Jv can be faster than analytic Jv due to overhead in symbolic Jacobian and derivative-action evaluation.

---

## What Was Tried And Did Not Work (or Was Revised)

1. **Printing from derivative workflow helper**
- Initial print helper checked membership incorrectly and skipped flux/source output.
- Fixed by moving printing to generic base model and correcting key handling.

2. **`noop_aux` workaround**
- Temporary workaround for empty-aux vectorization path.
- Replaced by robust empty-array handling in `to_numpy`; `Qaux` can now be truly empty where appropriate.

3. **Auto-derivative inference initial AST parsing**
- Failed due to indentation parsing of method source.
- Fixed via `textwrap.dedent`.
- Also expanded to support alias-based field references.

4. **Classical GN explicit simulation stability**
- Several settings produced dt collapse / NaN due to stiffness.
- Stabilized for demo by reducing horizon/amplitude/CFL in classical examples.

5. **IMEX solver constructor options**
- Passing custom attributes through attrs-generated constructor caused errors.
- Fixed by setting runtime options via `object.__setattr__` in benchmark setup.

---

## Current State

- You now have:
  - model-side derivative abstractions with runtime compatibility,
  - reusable linear analysis scripts for GN variants,
  - IMEX prototype with local/global source handling,
  - FD-Jv vs analytic Jv comparison and timing benchmark.

- This is a working baseline for your next abstraction round in a new chat.

