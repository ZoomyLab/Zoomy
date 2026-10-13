"""Generic linear-analysis tools for shallow-water-family PDE systems.

The unified analysis entry point is `SystemModel` (in
`zoomy_core.systemmodel.system_model`).  Every analysis routine
in this package consumes a SystemModel and never inspects
model-specific attributes.

Package contents:

* Linearisation around a base state (`linearise`) — operates on a
  SystemModel, returns a SystemModel of perturbation fields.
* Plane-wave dispersion analysis (`ω(k)` solutions).
* Generalised-eigenvalue ("pencil") form for systems with constraints.
* Numerical eigenvalue sampling for hyperbolicity over a parameter
  cube.

The library knows nothing about VAM, SME, ML-SWE, etc. — every
model-specific bit is in tutorials/.
"""
from .linearisation import linearise
from .plane_wave import plane_wave_dispersion, plane_wave_matrix
from .pencil import (
    extract_quasilinear_pencil,
    generalised_eigenvalues,
    sample_generalised_eigenvalues,
    symbolic_eigenvalues_at,
)
from .reduce_pencil import reduce_singular_pencil
from .stability import (
    NumericPencil,
    temporal_branch,
    spatial_branch,
    spatial_dispersion,
    spatial_cutoff,
    growth_cutoff,
    critical_parameter,
    viscous_operator,
)
from .hyperbolicity import (
    is_hyperbolic_at,
    sample_hyperbolicity,
)
from .plotting import (
    plot_dispersion,
    plot_hyperbolic_region_2d,
)

__all__ = [
    "linearise",
    "plane_wave_dispersion",
    "plane_wave_matrix",
    "extract_quasilinear_pencil",
    "generalised_eigenvalues",
    "sample_generalised_eigenvalues",
    "symbolic_eigenvalues_at",
    "reduce_singular_pencil",
    "NumericPencil",
    "temporal_branch",
    "spatial_branch",
    "spatial_dispersion",
    "spatial_cutoff",
    "growth_cutoff",
    "critical_parameter",
    "viscous_operator",
    "is_hyperbolic_at",
    "sample_hyperbolicity",
    "plot_dispersion",
    "plot_hyperbolic_region_2d",
]
