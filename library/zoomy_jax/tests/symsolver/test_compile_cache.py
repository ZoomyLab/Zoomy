"""`compile_cache`: a program with host callbacks is written to and loaded from the persistent
compilation cache (jax alone writes no entry for it), the loaded executable runs its callbacks
and gives the compiled one's result bitwise (ANALYSIS 65)."""
import jax
import jax.numpy as jnp
import numpy as np

from zoomy_jax.symsolver import compile_cache as cc


def _make(seen):
    """A fresh jitted function (same name, same program) whose ordered and unordered callbacks
    append to `seen`."""
    def march(x):
        jax.debug.callback(lambda v: seen.append(("ordered", float(v))), x.sum(), ordered=True)
        y = jnp.sin(x) * 1.5 + x ** 3
        jax.lax.cond(y.sum() > 0, lambda: jax.debug.callback(
            lambda v: seen.append(("cond", float(v))), y.max()), lambda: None)
        return y
    return jax.jit(march)


def test_callback_program_round_trips(tmp_path):
    old = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", str(tmp_path))
    try:
        x = jnp.linspace(0.1, 1.0, 17)
        hits, writes = cc.stats["hits"], cc.stats["writes"]
        seen1, seen2 = [], []
        with cc.callback_cache():
            y1 = _make(seen1)(x)
        jax.effects_barrier()
        assert cc.stats["writes"] == writes + 1
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(cc._PREFIX)]
        with cc.callback_cache():
            y2 = _make(seen2)(x)
        jax.effects_barrier()
        assert cc.stats["hits"] == hits + 1
        assert np.array_equal(np.asarray(y1), np.asarray(y2))
        assert seen1 == seen2 and len(seen2) == 2
    finally:
        jax.config.update("jax_compilation_cache_dir", old)


def test_inactive_without_cache_dir():
    old = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", None)
    try:
        writes = cc.stats["writes"]
        with cc.callback_cache():
            _make([])(jnp.ones(3))
        assert cc.stats["writes"] == writes
    finally:
        jax.config.update("jax_compilation_cache_dir", old)
