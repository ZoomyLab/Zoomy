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
    SourceTreatment,
    to_numerical_system_model,
)

__all__ = [
    "BedReconstruction",
    "DiffusionSpec",
    "Limiter",
    "NumericalSystemModel",
    "ReconstructedVariables",
    "ReconstructionSpec",
    "SourceTreatment",
    "to_numerical_system_model",
]

#: The Riemann numerics an NSM takes as `riemann=` (defined in `zoomy_core.fvm.riemann_solvers`;
#: re-exported here, imported on first use: that module imports the model layer)
RIEMANN = ("Rusanov", "HLL", "HLLC", "PositiveRusanov", "NonconservativeRusanov",
           "WellBalancedNonconservativeRusanov", "PositiveHLL", "PositiveNonconservativeRusanov",
           "NonconservativeRoe", "PositiveNonconservativeHLL", "QuasilinearRusanov",
           "PositiveQuasilinearRusanov", "QuasilinearRoe")


def __getattr__(name):
    if name in RIEMANN:
        from zoomy_core.fvm import riemann_solvers
        return getattr(riemann_solvers, name)
    raise AttributeError(f"module 'zoomy_core.numerics' has no attribute {name!r}")
