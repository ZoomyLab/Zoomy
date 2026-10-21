"""Function-level derivation cache: memoizes expensive derivation stages on (function source,
argument content). In-memory by default with an optional pickle tier on disk; decorate leaf stages,
not the orchestrator.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import pickle
from pathlib import Path

from zoomy_core.model.derivation.cache_keys import CACHE_VERSION, argument_key

logger = logging.getLogger(__name__)

# function-id → { call_key: result }
_STORE: dict[str, dict[str, object]] = {}


def _cache_enabled() -> bool:
    """Return whether derivation caching is enabled (see derivation_cache_enabled)."""
    from zoomy_core.model.derivation.cache_keys import derivation_cache_enabled
    return derivation_cache_enabled()


def _persist_dir() -> Path:
    """Return the on-disk cache directory ($ZOOMY_CACHE_DIR/derivation_cache or
    ~/.cache/zoomy/derivation_cache), creating it.
    """
    base = os.environ.get("ZOOMY_CACHE_DIR")
    root = str(Path(base) / "derivation_cache") if base else None
    p = Path(root) if root else Path.home() / ".cache" / "zoomy" / "derivation_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _persist_path(fid: str, key: str) -> Path:
    """Return the pickle path of one (function id, call key) entry."""
    safe = fid.replace(os.sep, "_").replace(".", "_")
    return _persist_dir() / f"{safe}-{key}.pkl"


class CacheStats:
    """Hit, miss and wrapped-body call counters of one decorated function."""

    def __init__(self):
        """Zero all counters."""
        self.hits = 0
        self.misses = 0
        self.calls = 0          # wrapped-body executions (= misses, kept distinct for clarity)

    def __repr__(self):
        """Counters as text."""
        return f"CacheStats(hits={self.hits}, misses={self.misses}, calls={self.calls})"


def _fn_source(fn) -> str:
    """Return the source of a function for the key, or its qualified name if no source is available."""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        logger.warning(
            "derivation_cache: no source for %s; keying on qualname only "
            "(edits to the body will NOT invalidate — pass version= to bump).",
            getattr(fn, "__qualname__", fn))
        return f"<no-source:{fn.__module__}.{getattr(fn, '__qualname__', fn)}>"


def _call_key(source: str, args, kwargs, version) -> str:
    """Return the hash of a call from CACHE_VERSION, version, function source and argument keys."""
    parts = [CACHE_VERSION, f"version={version!r}", source]
    parts += [argument_key(a) for a in args]
    parts += [f"{k}={argument_key(v)}" for k, v in sorted(kwargs.items())]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def derivation_cache(fn=None, *, version=0, verify=False, persist=False):
    """Memoize a derivation stage, keyed on function source and argument content; usable bare or
    with arguments.

    Args:
        version: Bump to invalidate the entries without editing the body.
        verify: On a hit re-run the body and assert an identical srepr (test aid).
        persist: Add a cross-session pickle tier; unpicklable results stay in memory.

    Returns:
        The wrapped function with .stats, .cache_clear() and .__wrapped__.
    """
    def decorate(func):
        """Wrap one function with the cache."""
        fid = f"{func.__module__}.{func.__qualname__}"
        source = _fn_source(func)
        _STORE.setdefault(fid, {})
        stats = CacheStats()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Return the cached result for the call, else compute and store it."""
            if not _cache_enabled():
                stats.misses += 1
                stats.calls += 1
                return func(*args, **kwargs)
            key = _call_key(source, args, kwargs, version)
            store = _STORE[fid]
            if key in store:
                stats.hits += 1
                cached = store[key]
                if verify:
                    fresh = func(*args, **kwargs)
                    _assert_same(fresh, cached, fid)
                return cached
            if persist:                              # cross-session disk tier
                path = _persist_path(fid, key)
                if path.exists():
                    try:
                        cached = pickle.loads(path.read_bytes())
                    except Exception as exc:         # corrupt / stale pickle
                        logger.warning(
                            "derivation_cache[%s]: unreadable disk entry (%s); "
                            "recomputing.", fid, exc)
                    else:
                        store[key] = cached
                        stats.hits += 1
                        return cached
            stats.misses += 1
            stats.calls += 1
            result = func(*args, **kwargs)
            store[key] = result
            if persist:
                try:
                    _persist_path(fid, key).write_bytes(
                        pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
                except Exception as exc:             # unpicklable → memory-only
                    logger.warning(
                        "derivation_cache[%s]: result not picklable (%s); "
                        "keeping in memory only.", fid, exc)
            return result

        wrapper.stats = stats
        wrapper.cache_clear = lambda: _STORE.get(fid, {}).clear()
        wrapper._cache_id = fid
        wrapper._persist = persist
        return wrapper

    return decorate if fn is None else decorate(fn)


def _assert_same(fresh, cached, fid):
    """Raise AssertionError if the fresh and cached results differ in srepr."""
    import sympy as sp
    try:
        a, b = sp.srepr(fresh), sp.srepr(cached)
    except Exception:
        a, b = repr(fresh), repr(cached)
    if a != b:
        raise AssertionError(
            f"derivation_cache[{fid}]: verify=True found a STALE entry — fresh "
            f"result differs from cached. The cache key is missing a "
            f"result-affecting argument.")


def clear_derivation_cache(fn=None):
    """Clear all entries, or those of one decorated function."""
    if fn is not None:
        _STORE.get(getattr(fn, "_cache_id", None), {}).clear()
    else:
        _STORE.clear()


def cache_size() -> int:
    """Return the total number of cached entries."""
    return sum(len(d) for d in _STORE.values())


__all__ = [
    "derivation_cache",
    "clear_derivation_cache",
    "cache_size",
    "CacheStats",
]
