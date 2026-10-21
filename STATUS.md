# dev/jax-compile — status at wrap-up (2026-10-21)

Full write-up: docs/dev/symbolic_solver/ANALYSIS.md section 65.  Bench: docs/dev/symbolic_solver/cases/jax_compile_time.py.

## Measured (jax 0.9.2, GPU 1 L40S, taskset 24-31)
- Escalante ML-VAM(2) march (GPU): trace 12.5 s, lower 3.6 s, XLA compile 300-303 s; 113.7k jaxpr eqns,
  75.8k StableHLO, 163k optimised HLO.  CPU: compile 33.6 s (191k HLO).  C8 vam GPU 51.8 s (27.9k HLO), CPU 6.8 s; C1 CPU 2.7 s.
- XLA GPU phases (Escalante): HLO passes 241 s, of them priority-fusion 220 s (serial;
  --xla_gpu_force_compilation_parallelism=8: 215 s, no gain), IR emission 24 s, PTX 8 s, ptxas 9 s.
- Expected for the size: ~1.8 ms / HLO instr on GPU; legacy run_jit_steps same case: 228 s, 88k HLO, priority-fusion 148 s.
- Size: 55 % lambdified sympy kernels inlined per call site (edge source 26.9k x6 calls, face kernel 22.4k x6,
  pressure slot ops 17.9k).  Constants not the issue (1991 hoisted args, 155 kB; folding off).
- Callbacks: every Python callback kind (debug.callback ordered/unordered, io_callback, pure_callback) blocks jax's
  cache write.  In the march: check.cfl, check.march (always), io.write, io.save/load_state, accept gates, pyamg/hypre
  pure_callback, preCICE.  Key identical across processes; setup's small programs do hit.
- Setup 117-140 s = sympy subs in Numerics._model_eval (99 s; 155 calls, 56 distinct).

## Changes (committed)
1. zoomy_core riemann_solvers._substitute: xreplace where provably the same tree as subs (guards); 609 calls checked == subs.
   Escalante setup 120 -> 20.7 s; C8 31 -> 18 s.
2. zoomy_jax symsolver/compile_cache.py + hoisted_jit: executables with host callbacks cached in
   jax_compilation_cache_dir (zoomy-cb-<jax key>), loaded via deserialize_executable(host_callbacks=).
   Escalante 2nd run: compile 303 -> 12.5 s (load), run 323 -> 24 s; bitwise vs compiling run; on_write frames same.
   Test: library/zoomy_jax/tests/symsolver/test_compile_cache.py (2 pass).
## Acceptance done
- Matrix C1/C8/C10 jax serial bitwise vs main (cache written and loaded); gpu1 C1 1.4e-13, C8 1.4e-11 (GPU round-off;
  two GPU compiles of the same program differ 3.8e-15; base-vs-base gpu1 not rerun).
## NOT done
- zoomy_core pytest, zoomy_jax full suite (vs main failures), QR suite.
- Program-size reduction (template: one kernel batch for O2 + MOOD fallback; stages as a loop) — proposal only.
- Caching is active only with JAX_COMPILATION_CACHE_DIR set; no default dir chosen (user decision).
- Possible merge conflict: dev/wavespeed-frame edits riemann_solvers.py near _model_eval.
