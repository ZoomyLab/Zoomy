"""Zoomy plotting — publication-ready matplotlib visualization for
HDF5 / VTK simulation outputs with lazy field access."""

from ._version import __version__
from .store import SimulationStore, Zstruct, FieldReader
from .readers import read, read_hdf5
from .plot import (
    CONFIG,
    PlotConfig,
    apply_style,
    BasePlotter,
    MatplotlibPlotter,
)

__all__ = [
    "__version__",
    "SimulationStore",
    "Zstruct",
    "FieldReader",
    "read",
    "read_hdf5",
    "CONFIG",
    "PlotConfig",
    "apply_style",
    "BasePlotter",
    "MatplotlibPlotter",
]
