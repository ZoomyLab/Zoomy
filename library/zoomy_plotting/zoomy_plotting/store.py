"""Canonical in-memory representation of a simulation result for plotting.

``SimulationStore`` holds the mesh and field-name index eagerly, but defers
reading individual field values to a callable installed by a reader. This
keeps opening a multi-GB HDF5 timeline cheap — only the cells needed for a
given plot call are ever read from disk.

Axis convention (enforced by ``__post_init__``): ``vertices.shape == (n_vertices, dim)``
and ``cells.shape == (n_cells, k)``. Readers are responsible for transposing
HDF5's ``dim x n`` layout into this canonical form exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from functools import cached_property
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------
# Zstruct: attribute-and-string indexing over a {name: int} mapping.
# --------------------------------------------------------------------------
class Zstruct:
    """Attribute-access wrapper around a dict of ``{name: index}``.

    Supports both ``z.name`` and ``z["name"]`` access. Iterable via ``keys``.
    """

    def __init__(self, mapping: Optional[dict] = None):
        self.__dict__.update(mapping or {})

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self):
        return f"Zstruct({sorted(self.__dict__)})"


# --------------------------------------------------------------------------
# FieldReader signature.
# --------------------------------------------------------------------------
# (time_step: int, field_index: int) -> 1-D numpy array of length n_cells
# (or n_vertices for point fields).
FieldReader = Callable[[int, int], np.ndarray]


# --------------------------------------------------------------------------
# SimulationStore
# --------------------------------------------------------------------------
@dataclass
class SimulationStore:
    """Lightweight handle to a simulation result.

    Eager attributes (mesh + metadata) are small enough to keep in memory for
    any reasonable simulation. Field values are fetched lazily through
    :py:meth:`get_cell` / :py:meth:`get_point`, which delegate to reader
    closures installed by :py:mod:`zoomy_plotting.readers`.

    Notes
    -----
    * ``field`` maps a cell-data variable name to its integer index along
      the variable axis. For HDF5 inputs, ``Q`` and ``Qaux`` are concatenated
      at read time, so ``store.field.h == 0`` regardless of whether ``h`` was
      originally a primary or auxiliary variable.
    * ``point_field`` has the same role for point data (VTK only in PR 1's
      horizon; left empty by the HDF5 reader).
    """

    # --- Eager mesh ------------------------------------------------------
    dim: int
    cell_type: str
    vertices: np.ndarray              # (n_vertices, dim)
    cells: np.ndarray                 # (n_cells, k)
    n_inner_cells: Optional[int] = None

    # --- Eager metadata --------------------------------------------------
    times: Optional[np.ndarray] = None          # (n_snaps,)
    field: Zstruct = _field(default_factory=Zstruct)        # cell-field name -> index
    point_field: Zstruct = _field(default_factory=Zstruct)  # point-field name -> index

    # --- Lazy field access (installed by the reader) --------------------
    _cell_reader: Optional[FieldReader] = None
    _point_reader: Optional[FieldReader] = None

    # --- Bookkeeping -----------------------------------------------------
    source_path: Optional[str] = None
    extras: dict = _field(default_factory=dict)

    # Opaque handle to any file/resource that needs to live as long as the
    # store (e.g. an h5py.File). Closed by ``close()`` / ``__del__``.
    _resource: object = None

    # ------------------------------------------------------------------
    # Validation & simple invariants
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3; got {self.dim}")
        v = self.vertices
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(
                f"vertices must have shape (n_vertices, dim={self.dim}); "
                f"got {v.shape}. Readers must transpose HDF5 (dim, n)."
            )
        c = self.cells
        if c.ndim != 2:
            raise ValueError(f"cells must be 2-D (n_cells, k); got {c.shape}")

    # ------------------------------------------------------------------
    # Lazy field access
    # ------------------------------------------------------------------
    def get_cell(self, time_step: int, field) -> np.ndarray:
        """Return the requested cell-field at the requested time step.

        ``field`` may be an ``int`` index, a ``str`` variable name, or a
        ``sympy.Symbol`` (duck-typed via its ``.name`` attribute).
        """
        if self._cell_reader is None:
            raise RuntimeError(
                "store has no cell reader installed — did you construct the "
                "SimulationStore by hand? Use zoomy_plotting.read*() or install "
                "a reader closure on _cell_reader."
            )
        idx = self._resolve(field, self.field)
        return self._cell_reader(int(time_step), idx)

    def get_point(self, time_step: int, field) -> np.ndarray:
        """Return the requested point-field at the requested time step."""
        if self._point_reader is None:
            raise RuntimeError(
                "store has no point reader installed — the HDF5 reader does "
                "not populate point fields (VTK reader arrives in PR 2)."
            )
        idx = self._resolve(field, self.point_field)
        return self._point_reader(int(time_step), idx)

    @staticmethod
    def _resolve(spec, ztable: Zstruct) -> int:
        """Normalise ``spec`` to an integer field index using ``ztable``."""
        if isinstance(spec, (int, np.integer)):
            return int(spec)
        if isinstance(spec, str):
            try:
                return int(ztable[spec])
            except KeyError as e:
                raise KeyError(
                    f"field {spec!r} not in {list(ztable.keys())}"
                ) from e
        # Duck-type sympy.Symbol to avoid importing sympy at this layer.
        name = getattr(spec, "name", None)
        if isinstance(name, str):
            return SimulationStore._resolve(name, ztable)
        raise TypeError(
            f"unsupported field spec: expected int | str | sympy.Symbol, "
            f"got {type(spec).__name__}"
        )

    # ------------------------------------------------------------------
    # Derived mesh helpers (cheap, cached)
    # ------------------------------------------------------------------
    @cached_property
    def cell_centers(self) -> np.ndarray:
        """Cell geometric centers, shape ``(n_cells, dim)``."""
        return self.vertices[self.cells].mean(axis=1)

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_snapshots(self) -> int:
        return 1 if self.times is None else int(len(self.times))

    # ------------------------------------------------------------------
    # Resource lifecycle
    # ------------------------------------------------------------------
    def close(self):
        """Release any underlying resource (e.g. HDF5 file handle)."""
        r = self._resource
        self._resource = None
        if r is not None:
            closer = getattr(r, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Best-effort; ignore everything during interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass
