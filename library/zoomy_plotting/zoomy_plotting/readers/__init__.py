"""Readers that populate a :class:`~zoomy_plotting.store.SimulationStore`.

PR 1 ships only the HDF5 reader; VTK arrives in PR 2.
"""

from __future__ import annotations

import os

from .hdf5 import read_hdf5

__all__ = ["read", "read_hdf5"]


def read(path, **kwargs):
    """Dispatch to the appropriate reader based on file extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".h5", ".hdf5"):
        return read_hdf5(path, **kwargs)
    raise ValueError(
        f"no reader registered for extension {ext!r}; "
        "supported so far: .h5 / .hdf5"
    )
