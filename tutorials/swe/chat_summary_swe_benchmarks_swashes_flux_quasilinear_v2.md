# SWE benchmarks: validation ladder, flux vs quasilinear, SWASHES (chat summary)

This document summarizes a working thread on shallow water equation (SWE) validation, symbolic numerics v2, flux versus quasilinear discretizations, hydrostatic reconstruction (HR), and external SWASHES reference cases. Use it to resume work in a new conversation.

## What we set out to do

1. **Clarify runup / benchmark output** from `benchmark_suite_symbolic_numerics_v2.py` and whether it matches literature expectations.
2. **Build confidence in the SWE stack** in a deliberate order:
   - flat bed, fully wet;
   - bathymetry, fully wet;
   - bathymetry with wet/dry fronts.
3. **Understand bathymetry treatment**—in particular whether it is handled consistently via hydrostatic reconstruction (HR) in the flux-based scheme.
4. **Compare flux-based vs quasilinear** solvers on smooth dynamic cases (`b = 0` and `b \neq 0`, wet), and probe HR in the quasilinear path.
5. **Add literature-grade wet/dry checks** using **SWASHES** analytic/reference solutions, including grid refinement and error metrics.
6. **Improve robustness** when the time step collapses (unstable or stiff configurations).

**Explicitly deferred / not done in this thread:** friction/source terms; returning to GN (Green–Naghdi) once SWE is trusted.

---

## What we did

### Validation ladder (internal SWE checks)

- Added **`benchmark_swe_validation_ladder_v2.py`**: finite values, positivity, mass conservation, peak/shoreline diagnostics, **lake-at-rest (well-balancing)** test, and **grid refinement** sweeps.
- Corrected the **bathy–fullwet** case so the surface elevation keeps the domain truly full wet (raised `eta` baseline so cells do not dry inadvertently).

### Flux vs quasilinear (diagnostics)

- Added **`benchmark_flux_vs_quasilinear_swe_v2.py`**: pairs **flux + HR** (`PositiveNonconservativeRusanov`) with **quasilinear** variants (`QuasilinearRusanov`, `PositiveQuasilinearRusanov`, and diagnostic subclasses).
- For **wet/dry dynamics**, the quasilinear runs proved **very stiff**; the benchmark was **shortened in time** and coarsened where needed so comparisons terminate instead of stalling.
- A separate investigation script **`investigate_bathy_fullwet_hr_vs_quasilinear_v2.py`** was used to focus on HR vs quasilinear behavior on smooth bathy-fullwet setups (positive-quasilinear variants were excluded where they diverged).

### Interpretation: why flux and “positive quasilinear” disagreed

- For **smooth full-wet** cases, **flux** and **quasilinear without HR** tracked each other closely.
- **`PositiveQuasilinearRusanov`** (flux piece + HR + quasilinear path integral) showed **large discrepancies** on bathymetry cases.
- Diagnostic variant **`PositiveQuasilinearNoJump`** (HR but **pressure-jump** part of fluctuations removed from the positive-quasilinear path) supported the hypothesis that **hydrostatic effects were effectively double-counted** in the positive-quasilinear construction—not a generic failure of smooth IC matching.

### SWASHES integration

- Installed **`swashes`** into the project **`.venv`** and added **`export_swashes_reference_v2.py`** to run the CLI, parse output, and write CSV + metadata.
- Added **`benchmark_swashes_wetdry_compare_v2.py`**: L1/L2/Linf-style norms for `h`, `\eta`, `u` vs reference; multiple grid levels; variants **`flux_hr`**, **`flux_nohr`**, **`quasi_nohr`**; **per-case variant lists** so unstable combinations are not run blindly.
- Added **`plot_swashes_convergence_v2.py`**: refinement grid **`[20, 40, 80, 120, 240]`**, **`h_L1`** log–log plots per case, **`convergence_summary.csv`**. **BCs** were fixed so the model receives `cfg["bcs"]` consistently.

### Solver robustness (`dt` floor)

- In **`solver_numpy.py`** and **`solver_imex_numpy.py`**, added **`min_dt`** (default `1e-6`) and a **`RuntimeError`** if `dt` is non-finite, non-positive, or below that floor—so “silent” `dt \to 0` stalls surface as explicit failures.

### Environment / execution

- Benchmarks were run with **`/home/ingo/git/Zoomy/.venv/bin/python`** (no extra `PYTHONPATH`; **`zoomy_core`** installed in the venv).

---

## What we tried that failed or was limited

| Item | Outcome |
|------|--------|
| **`ModuleNotFoundError: zoomy_core`** | Fixed by using the `.venv` interpreter / install, not code. |
| **Parameter extraction `TypeError`** (`SymPy` vs float) | Fixed: use **`model.parameter_values`**, not raw `parameters.values()`. |
| **`PositiveQuasilinearRusanov` + bathymetry** | **Large drift vs flux**; diagnosed as inconsistent HR + path / **pressure-jump** interaction. |
| **`quasi_nohr` on SWASHES Ritter (dry-front dam break)** | **`dt` collapse** / instability; **excluded** from that case in the benchmark config; flux variants used instead. |
| **`WetDryPathPositiveQuasilinearRusanov` `name` override** | **`param`** type error; removed invalid string override of `name`. |
| **`NameError` in `_compare_solutions`** | Fixed variable naming (`Q_ref` / `Q_test`). |
| **SWASHES export parsing** | Initial parse failed; fixed by detecting table start via **`(i-0.5)*dx`** in commented header lines. |
| **`swashes` binary missing** | Fixed with **`pip install swashes`** and robust lookup (`sys.executable` dir, `shutil.which`). |
| **`plot_swashes_convergence_v2` wrong BCs** | Fixed by passing **`cfg["bcs"]`** into the model factory. |
| **Beach runup vs literature** | **Not resolved** in this thread: focus shifted to systematic SWE + SWASHES validation; runup remains a separate credibility task. |

---

## Where we succeeded

- **Lake-at-rest / well-balancing**: zero drift reported for the ladder test configuration.
- **Internal ladder**: stable behavior and refinement trends for the three SWE scenarios (after fixing full-wet bathy `eta`).
- **Flux vs quasilinear (smooth wet)**: **flux** and **quasilinear no-HR** agree closely; **positive quasilinear + HR** identified as the problematic combination for bathymetry.
- **SWASHES**: reference CSVs exported; **flux_hr / flux_nohr** show **convergent `h_L1`** vs SWASHES on both cases (per run logs); **step** case allowed **`quasi_nohr`** with convergence; **Ritter** case **`quasi_nohr`** intentionally skipped.
- **Operational**: **`min_dt`** guard makes unstable runs **fail fast** instead of hanging.

---

## Files produced or heavily edited

### New or central tutorial / benchmark scripts (`web/tutorials/swe/`)

| File | Role |
|------|------|
| `benchmark_swe_validation_ladder_v2.py` | SWE ladder + well-balancing + refinement CSVs/plot |
| `benchmark_flux_vs_quasilinear_swe_v2.py` | flux vs quasilinear comparisons + diagnostic numerics subclasses |
| `investigate_bathy_fullwet_hr_vs_quasilinear_v2.py` | focused HR / quasilinear investigation |
| `export_swashes_reference_v2.py` | run `swashes`, parse output → reference CSV |
| `benchmark_swashes_wetdry_compare_v2.py` | norms vs SWASHES, per-case variants, `summary.csv` |
| `plot_swashes_convergence_v2.py` | refinement sweep, `convergence_summary.csv`, log–log `h_L1` plots |

Pre-existing entry points still relevant: **`benchmark_suite_symbolic_numerics_v2.py`**, **`beach_runup_swe_vs_gn_classical_v2.py`**, **`chat_summary_runup_symbolic_numerics_v2.md`**.

### Core library (this thread)

| File | Change |
|------|--------|
| `library/zoomy_core/zoomy_core/fvm/solver_numpy.py` | `min_dt` floor + error on bad `dt` |
| `library/zoomy_core/zoomy_core/fvm/solver_imex_numpy.py` | same |

Discussion-heavy / existing: `library/zoomy_core/zoomy_core/fvm/symbolic_numerics_v2.py` (HR, flux, quasilinear implementations).

### Typical output locations (after running scripts)

| Directory / artifact | Content |
|----------------------|--------|
| `outputs/swashes_reference_v2/` | e.g. `swashes_dam_break_dry_ritter.csv`, `swashes_dam_break_step.csv` + metadata |
| `outputs/benchmark_swashes_wetdry_compare_v2/` | `summary.csv`, `convergence_summary.csv`, `*_log_error_hL1.png`, temp solver dirs |
| `outputs/swe_validation_ladder_v2/` | `metrics.csv`, `well_balancing_metrics.csv`, `refinement_metrics.csv`, ladder plot |
| `outputs/flux_vs_quasilinear_swe_v2/` | `metrics.csv` |
| `outputs/investigate_bathy_fullwet_hr_vs_quasilinear_v2/` | investigation-derived plots/CSVs (as configured in script) |

Paths are relative to the **repository root** (e.g. `/home/ingo/git/Zoomy/`).

---

## Suggested next steps (for a new chat)

1. **Friction / bottom stress** on SWE, possibly using SWASHES cases if available.
2. **Beach runup** revisited with a **documented reference** (analytic, lab, or high-res benchmark) once SWASHES track is stable.
3. **GN equations** after SWE + friction narrative is closed.

---

## Transcript

Full Cursor conversation JSONL (parent session):  
`/home/ingo/.cursor/projects/home-ingo-git-Zoomy/agent-transcripts/120bbcfa-b119-44bd-bf80-c83aa3adb6a4/120bbcfa-b119-44bd-bf80-c83aa3adb6a4.jsonl`
