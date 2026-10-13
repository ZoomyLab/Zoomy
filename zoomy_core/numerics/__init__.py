"""Numerical layer: `NumericalSystemModel` wraps a `SystemModel` with the reconstruction,
Riemann solver, diffusion scheme and regularization the solver templates consume.
Pipeline: Model -> SystemModel -> NumericalSystemModel -> solver template -> backend runtime.
"""

from zoomy_core.numerics.numerical_system_model import (
    BedReconstruction,
    DiffusionSpec,
    Limiter,
    NumericalSystemModel,
    ReconstructedVariables,
    ReconstructionSpec,
    to_numerical_system_model,
)

__all__ = [
    "BedReconstruction",
    "DiffusionSpec",
    "Limiter",
    "NumericalSystemModel",
    "ReconstructedVariables",
    "ReconstructionSpec",
    "to_numerical_system_model",
]
