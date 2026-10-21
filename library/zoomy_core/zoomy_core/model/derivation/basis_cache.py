"""Manifest-based on-disk cache for basis-function matrices, plus an in-memory wrapper. Sits under
the moment/Galerkin projection: consumers call get_basis_matrices instead of building a
SymbolicIntegrator.
"""

import hashlib
import json
import os
import time

import numpy as np
import sympy
from sympy.abc import z

from zoomy_core.misc.misc import get_main_directory


CACHE_ROOT = ".cache/basismatrices"
MANIFEST_NAME = "manifest.json"
MATRIX_NAMES = ["phib", "M", "A", "B", "D", "Dxi", "Dxi2", "DD", "D1", "DT"]


def _basis_fingerprint(basis):
    """Return a string that changes whenever the basis name, level, bounds, weight or polynomials
    change.
    """
    parts = [
        basis.name,
        str(basis.level),
        str(basis.bounds()),
        str(basis.weight(z)),
    ]
    for k in range(basis.level + 1):
        parts.append(str(basis.get(k)))
    return "|".join(parts)


def _compute_hash(fingerprint):
    """Return the 16-hex-digit SHA-256 prefix of a fingerprint string."""
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


class BasisMatrixCache:
    """Hash-keyed .npz cache of the basis matrices, indexed by a JSON manifest. Stale entries are
    detected by comparing the stored basis fingerprint.
    """

    def __init__(self, cache_root=CACHE_ROOT):
        """Set the cache directory, relative to the project root."""
        self.cache_root = cache_root
        self._abs_root = os.path.join(get_main_directory(), cache_root)

    def _manifest_path(self):
        """Path of the manifest file."""
        return os.path.join(self._abs_root, MANIFEST_NAME)

    def _load_manifest(self):
        """Read the manifest, or return an empty dict if absent."""
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest):
        """Write the manifest, creating the cache directory."""
        os.makedirs(self._abs_root, exist_ok=True)
        path = self._manifest_path()
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _entry_dir(self, hash_key):
        """Directory of one hash entry."""
        return os.path.join(self._abs_root, hash_key)

    def _npz_path(self, hash_key):
        """Path of the matrices archive of one hash entry."""
        return os.path.join(self._entry_dir(hash_key), "matrices.npz")

    def get_or_compute(self, basis, level, compute_fn):
        """Return the cached matrices of a basis or compute and store them.

        Args:
            basis: Basisfunction instance.
            level: Polynomial level.
            compute_fn: Called as compute_fn(level), returns {name: ndarray}.

        Returns:
            Dict {name: ndarray} of basis matrices.
        """
        fingerprint = _basis_fingerprint(basis)
        hash_key = _compute_hash(fingerprint)

        cached = self._try_load(hash_key, fingerprint)
        if cached is not None:
            return cached

        matrices = compute_fn(level)
        self._save(hash_key, fingerprint, basis, level, matrices)
        return matrices

    def _try_load(self, hash_key, fingerprint):
        """Return the stored matrices if the manifest fingerprint matches and the archive loads,
        else None.
        """
        manifest = self._load_manifest()
        entry = manifest.get(hash_key)
        if entry is None:
            return None
        if entry.get("fingerprint") != fingerprint:
            return None
        npz_path = self._npz_path(hash_key)
        if not os.path.exists(npz_path):
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            return {name: data[name] for name in MATRIX_NAMES if name in data}
        except Exception:
            return None

    def _save(self, hash_key, fingerprint, basis, level, matrices):
        """Write the matrices archive and add its manifest entry."""
        entry_dir = self._entry_dir(hash_key)
        os.makedirs(entry_dir, exist_ok=True)

        npz_path = self._npz_path(hash_key)
        np.savez(npz_path, **matrices)

        manifest = self._load_manifest()
        manifest[hash_key] = {
            "basis_name": basis.name,
            "level": level,
            "fingerprint": fingerprint,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matrices": list(matrices.keys()),
        }
        self._save_manifest(manifest)

    def list_entries(self):
        """Return the manifest dict."""
        return self._load_manifest()

    def clear(self):
        """Delete the whole on-disk cache."""
        import shutil
        if os.path.exists(self._abs_root):
            shutil.rmtree(self._abs_root)


# ── One-stop entrypoint for downstream pipelines ───────────────────
#
# Combines in-memory + on-disk caching: compute once per (basis, level)
# in a Python session, persist via BasisMatrixCache for cross-session
# reuse.  SME / VAM / MLSME and any other consumer should reach for
# this — they don't construct a SymbolicIntegrator themselves.

_INMEM_BASIS_MATRICES: dict = {}
_DEFAULT_CACHE: "BasisMatrixCache | None" = None


def _default_cache() -> "BasisMatrixCache":
    """Return the lazily created process-wide BasisMatrixCache."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = BasisMatrixCache()
    return _DEFAULT_CACHE


def get_basis_matrices(basis, level=None, *, cache=None):
    """Return {phib, M, A, B, D, Dxi, Dxi2, DD, D1, DT} for (basis, level), memoized in memory and
    on disk.

    Args:
        basis: Basisfunction exposing name, level, bounds(), weight(z), get(k).
        level: Polynomial level; defaults to basis.level.
        cache: Override of the default disk cache.
    """
    if level is None:
        level = basis.level
    key = (basis.name, int(level), _basis_fingerprint(basis))
    cached = _INMEM_BASIS_MATRICES.get(key)
    if cached is not None:
        return cached

    # Lazy import to avoid circulars during package init.
    from zoomy_core.model.derivation.symbolic_integrator import SymbolicIntegrator

    integrator = SymbolicIntegrator(basis)
    disk_cache = cache if cache is not None else _default_cache()
    matrices = disk_cache.get_or_compute(
        basis, level, integrator.compute_all_matrices
    )
    _INMEM_BASIS_MATRICES[key] = matrices
    return matrices


def clear_basis_matrix_cache(*, in_memory: bool = True, disk: bool = False):
    """Drop the in-memory cache and optionally the on-disk cache."""
    if in_memory:
        _INMEM_BASIS_MATRICES.clear()
    if disk:
        _default_cache().clear()
