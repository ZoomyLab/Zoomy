"""The persistent compilation cache for programs with host callbacks.

jax (0.9) writes no persistent-cache entry for an executable with Python host callbacks
(`jax.debug.callback`, `io_callback`, `pure_callback`: ordered or not, every one is an
`xla_ffi_python_*_callback` custom call whose callable is bound to the executable at load), so
the march of the symbolic solvers, which carries the CFL warning, the linear-solve accept gate,
`check.march` and `io.write`, compiled anew in every process (ANALYSIS 65).  The executable itself
does not depend on the callables: the custom call names its callback by an index into the list
handed to the compiler, and jaxlib's `deserialize_executable` takes that list.  So inside
`callback_cache()` an executable with callbacks is cached as jax caches any other one (the same
key, `compilation_cache.get_cache_key`, in `jax_compilation_cache_dir`), and a hit is loaded with
the callbacks of the present lowering, which binds them in the same order.

Active only with a persistent cache directory configured (`jax_compilation_cache_dir`,
`JAX_COMPILATION_CACHE_DIR`) and within `callback_cache()` (the solvers' march); opt out with
`ZOOMY_JAX_CALLBACK_CACHE=0`.  Any failure (another jax version, an unreadable entry) falls back
to jax's own compile.

References:
    Bradbury et al. 2018, https://github.com/jax-ml/jax
"""
from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import os
import tempfile
import zlib

logger = logging.getLogger(__name__)

_ACTIVE = contextvars.ContextVar("zoomy_callback_cache", default=False)
_PARAMS = ["backend", "computation", "devices", "compile_options", "host_callbacks",
           "executable_devices"]
_PREFIX = "zoomy-cb-"
_installed = []
stats = dict(hits=0, writes=0)


@contextlib.contextmanager
def callback_cache():
    """Compiles inside this context also cache executables with host callbacks."""
    install()
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _cache_dir():
    """The persistent cache directory (a local path), else `None`."""
    from jax._src import config
    d = config.compilation_cache_dir.value
    if not d or "://" in str(d):
        return None
    return str(d)


def install():
    """Wrap jax's `compiler.compile_or_get_cached` once (no-op on an unknown signature)."""
    if _installed or os.environ.get("ZOOMY_JAX_CALLBACK_CACHE", "1") == "0":
        return
    _installed.append(True)
    try:
        from jax._src import compilation_cache, compiler
        orig = compiler.compile_or_get_cached
        if list(inspect.signature(orig).parameters)[:len(_PARAMS)] != _PARAMS:
            logger.debug("compile_cache: unknown compile_or_get_cached signature, not installed")
            return
    except Exception as e:                      # noqa: BLE001  (another jax layout)
        logger.debug(f"compile_cache: not installed ({e})")
        return

    def compile_or_get_cached(backend, computation, devices, compile_options, host_callbacks,
                              executable_devices, *args, **kwargs):
        """jax's compile, plus the persistent cache for executables with host callbacks."""
        d = _cache_dir() if (host_callbacks and _ACTIVE.get()) else None
        key = None
        if d is not None:
            try:
                key = compilation_cache.get_cache_key(computation, devices, compile_options,
                                                      backend)
            except Exception as e:              # noqa: BLE001
                logger.debug(f"compile_cache: no key ({e})")
        if key is None:
            return orig(backend, computation, devices, compile_options, host_callbacks,
                        executable_devices, *args, **kwargs)
        path = os.path.join(d, _PREFIX + key)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    blob = zlib.decompress(f.read())
                exe = backend.deserialize_executable(blob, executable_devices, compile_options,
                                                     host_callbacks=list(host_callbacks))
                stats["hits"] += 1
                logger.debug(f"compile_cache: hit {key}")
                return exe
            except Exception as e:              # noqa: BLE001  (a torn or stale entry)
                logger.debug(f"compile_cache: unreadable entry {key} ({e})")
        exe = compiler.backend_compile_and_load(backend, computation, executable_devices,
                                                compile_options, host_callbacks)
        try:
            blob = zlib.compress(backend.serialize_executable(exe))
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-" + _PREFIX)
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)                # atomic: concurrent ranks read whole files
            stats["writes"] += 1
        except Exception as e:                  # noqa: BLE001
            logger.debug(f"compile_cache: not written ({e})")
        return exe

    compiler.compile_or_get_cached = compile_or_get_cached
