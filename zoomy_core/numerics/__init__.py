"""Numerical layer: ``NumericalSystemModel`` wraps a ``SystemModel`` with the reconstruction,
Riemann solver, diffusion scheme and regularization the solver templates consume.
Pipeline: Model -> SystemModel -> NumericalSystemModel -> solver template -> backend runtime.
"""

from zoomy_core.numerics.numerical_system_model import (
    DiffusionSpec,
    NumericalSystemModel,
    ReconstructionSpec,
    to_numerical_system_model,
)

__all__ = [
    "DiffusionSpec",
    "NumericalSystemModel",
    "ReconstructionSpec",
    "to_numerical_system_model",
]
