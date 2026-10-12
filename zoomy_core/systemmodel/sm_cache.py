"""Cache of built ``SystemModel`` objects (memory, user directory, shipped ``_prebuilt``), keyed on the
model spec and the source of every class in its MRO; a model whose source cannot be read is never cached.
Off by default; ``ZOOMY_DERIVATION_CACHE=1`` enables it and ``ZOOMY_DERIVATION_REBUILD=1`` rebuilds.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os
import pickle
from pathlib import Path

_MEMORY: dict[str, bytes] = {}

_LOG = logging.getLogger(__name__)

# Model classes we've already warned are uncacheable (REPL/exec-defined) — the
# warning fires once per class, not once per derivation.
_UNCACHEABLE_WARNED: set[str] = set()


def _enabled() -> bool:
    """Whether cache reads are enabled (``ZOOMY_DERIVATION_CACHE``)."""
    from zoomy_core.model.derivation.cache_keys import derivation_cache_enabled
    return derivation_cache_enabled()


def _rebuild() -> bool:
    """Whether ``ZOOMY_DERIVATION_REBUILD`` forces a fresh build."""
    return os.environ.get("ZOOMY_DERIVATION_REBUILD", "").strip() in {"1", "true", "yes"}


def _user_dir() -> Path:
    """User cache directory (``$ZOOMY_CACHE_DIR`` or ``~/.cache/zoomy``) for pickled
    ``SystemModel`` objects.
    """
    root = os.environ.get("ZOOMY_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "zoomy"
    return base / "systemmodels"


def _prebuilt_dir() -> Path:
    """Directory of the ``SystemModel`` pickles shipped inside the package."""
    return Path(__file__).resolve().parent / "_prebuilt"


def _src_hash(obj) -> str:
    """Short hash of the source of ``obj`` (``"nosrc"`` if unavailable)."""
    try:
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "nosrc"


def _mro_classes(cls):
    """The model's MRO classes minus ``object`` and ``abc`` plumbing."""
    for c in cls.__mro__:
        if c is object or c.__module__ == "abc":
            continue
        yield c


def _mro_source_terms(cls) -> str | None:
    """Identity and source hash of every MRO class, or ``None`` if one has no retrievable source."""
    parts = []
    for c in _mro_classes(cls):
        try:
            src = inspect.getsource(c)
        except (OSError, TypeError):
            _warn_uncacheable(cls, c)
            return None
        digest = hashlib.sha256(src.encode()).hexdigest()[:16]
        parts.append(f"{c.__module__}.{c.__qualname__}:{digest}")
    return "|".join(parts)


def _warn_uncacheable(model_cls, failed_cls) -> None:
    """Warn once per model class that it is uncacheable."""
    ident = f"{model_cls.__module__}.{model_cls.__qualname__}"
    if ident in _UNCACHEABLE_WARNED:
        return
    _UNCACHEABLE_WARNED.add(ident)
    _LOG.warning(
        "SystemModel cache DISABLED for %s: cannot read the source of %s.%s "
        "(REPL/exec-defined?). Deriving fresh every call — a source we cannot "
        "fingerprint could otherwise be served stale after an edit (REQ-188).",
        ident, failed_cls.__module__, failed_cls.__qualname__,
    )


def cache_key(model, builder) -> str | None:
    """Cache key of the built ``SystemModel`` of ``model`` (spec, MRO sources, builder source), or
    ``None`` if uncacheable.
    BCs and ICs are runtime data and are not part of the key.
    """
    from zoomy_core.model.derivation.cache_keys import CACHE_VERSION, model_spec_key
    mro = _mro_source_terms(type(model))
    if mro is None:
        return None                       # uncacheable: always derive fresh
    spec = model_spec_key(model)
    return hashlib.sha256(
        f"{CACHE_VERSION}|sm|{spec}|mro={mro}|b={_src_hash(builder)}".encode()
    ).hexdigest()


def fetch(key: str | None):
    """A fresh ``SystemModel`` unpickled for ``key``, or ``None`` on a miss, disabled cache or
    ``key is None``.
    """
    if key is None or not _enabled() or _rebuild():
        return None
    blob = _MEMORY.get(key)
    if blob is None:
        for root in (_user_dir(), _prebuilt_dir()):
            p = root / f"{key}.pkl"
            if p.is_file():
                try:
                    blob = p.read_bytes()
                except OSError:
                    continue
                _MEMORY[key] = blob
                break
    if blob is None:
        return None
    try:
        return pickle.loads(blob)
    except Exception as exc:
        # Corrupt / stale-format entry: falling through to a fresh build is
        # always CORRECT — but say so.  Swallowing this silently is how all 16
        # shipped ``_prebuilt`` entries became unreadable under param 2.3.3
        # (``'List' object has no attribute 'class_'`` — that slot is
        # ``item_type`` now) with nothing reporting it: every build that should
        # have hit the cache paid a full symbolic derivation instead, and only
        # a test that unpickled the tier by hand ever noticed.
        import warnings
        warnings.warn(
            f"sm_cache: cached SystemModel {key[:12]}… could not be unpickled "
            f"({type(exc).__name__}: {exc}); deriving fresh.  If this fires for "
            "every model, the shipped cache is stale for this environment — "
            "regenerate with "
            "`python -m zoomy_core.systemmodel.build_prebuilt_cache`.",
            RuntimeWarning, stacklevel=2)
        _MEMORY.pop(key, None)
        return None


def store(key: str | None, sm) -> None:
    """Persist ``sm`` under ``key`` in memory and the user directory; a no-op for ``key is None``
    or a disabled cache.
    """
    if key is None or not _enabled():   # uncacheable model: never persist
        return
    try:
        blob = pickle.dumps(sm)
    except Exception:
        return                    # unpicklable exotic model: memory-only skip
    _MEMORY[key] = blob
    try:
        d = _user_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.tmp"
        tmp.write_bytes(blob)
        tmp.replace(d / f"{key}.pkl")
    except OSError:
        pass                      # read-only FS: memory tier still works
