"""zoomy_jax — JAX backend for Zoomy.

Sets the process-global JAX float precision (x64 by default) and the persistent XLA compilation cache at package import, since both must be configured before the first array or compile.
"""

import os

import jax  # type: ignore[reportMissingImports]


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no"}


# Controlled by env so backend/precision benchmarks can compare float32 vs
# float64.  Default remains x64 for stability/parity.
ZOOMY_JAX_ENABLE_X64 = _env_on("ZOOMY_JAX_ENABLE_X64")
jax.config.update("jax_enable_x64", ZOOMY_JAX_ENABLE_X64)

# REQ-179 (2): jax's persistent XLA compilation cache, ON by default — same
# standing intent as REQ-163's symbolic cache: caching is the default, opt out
# via env.  Configured HERE for the same reason as x64: the dir must be set
# before the first jitted function compiles, and this module runs on every
# `zoomy_jax.*` import.  Skips the XLA compile of an already-seen jitted step
# across processes — the ML-FullVAM setup that motivated this pays 20+ min of
# lowering+jit on a cold process (part (1), the lambdify tier, is core's).
#
# ⚠ SEGREGATED PER HOST.  jax's CPU cache stores an AOT executable compiled for
# the WRITER's CPU features; loading it on a host with a different ISA warns
# "machine type ... doesn't match ... could lead to SIGILL" (verified: a warm
# read on a feature-mismatched host emits `cpu_aot_loader` errors).  On this
# cluster `~/.cache` is shared NFS across heterogeneous nodes, so a flat cache
# would let one node load another's kernel → SIGILL / silent miscompute.  Keying
# the dir by hostname means a node only ever loads what it compiled — which
# preserves exactly what gui asked for (same-host, cross-process reuse) and
# drops the unsafe cross-node reuse.  An explicit `JAX_COMPILATION_CACHE_DIR`
# always wins (the user takes responsibility); `ZOOMY_JAX_COMPILATION_CACHE=0`
# disables.
if _env_on("ZOOMY_JAX_COMPILATION_CACHE") and not os.environ.get(
        "JAX_COMPILATION_CACHE_DIR") and not jax.config.jax_compilation_cache_dir:
    _host = os.uname().nodename or "unknown-host"
    _xla_cache = os.path.expanduser(
        os.environ.get("ZOOMY_JAX_COMPILATION_CACHE_DIR",
                       os.path.join("~/.cache/zoomy/xla", _host)))
    jax.config.update("jax_compilation_cache_dir", _xla_cache)
    # jax's default already skips compiles under 1.0s (not worth a disk hit);
    # the solver-setup compiles this targets are far above that bar.  Explicit
    # for intent.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
