"""Mesh hierarchy BaseMesh -> FVMMesh -> LSQMesh shared by every backend."""

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_mesh import LSQMesh
from zoomy_core.mesh.mesh_ic import (
    initial_conditions_from_mesh,
    interpolate_node_data_to_cells,
)


def ensure_lsq_mesh(mesh, model) -> LSQMesh:
    """Promote `mesh` to an LSQMesh whose stencil degree is taken from `model`.

    Args:
        mesh: BaseMesh, FVMMesh or LSQMesh; an LSQMesh of lower degree is rebuilt in place.
        model: Model, SystemModel or NumericalSystemModel; its resolved LSQ degree sets the stencil.
    Returns:
        The LSQMesh (or `mesh` untouched if it is not a BaseMesh).
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    if model is None:
        raise TypeError(
            "ensure_lsq_mesh now requires a model argument — the LSQ "
            "polynomial degree is derived from the model's "
            "NumericalSystemModel.  Pass the Model / SystemModel / NSM "
            "that the solver will consume."
        )

    # Resolve the required degree once, via the NSM.  Importing NSM
    # at module level would create a mesh → numerics cycle.
    from zoomy_core.numerics import NumericalSystemModel
    if isinstance(model, NumericalSystemModel):
        nsm = model
    else:
        nsm = NumericalSystemModel.from_system_model(model)
    required_degree = nsm.resolved_lsq_degree()

    if isinstance(mesh, LSQMesh):
        if mesh._current_lsq_degree() < required_degree:
            mesh._build_lsq_stencil(required_degree)
        return mesh

    if isinstance(mesh, FVMMesh):
        lsq = LSQMesh.from_fvm(mesh)
    elif isinstance(mesh, BaseMesh):
        lsq = LSQMesh.from_fvm(FVMMesh.from_base(mesh))
    else:
        # Fallback: old Mesh class without an LSQ hierarchy — return
        # untouched.  Solvers that hit this raise downstream.
        return mesh

    if required_degree > 1:
        lsq._build_lsq_stencil(required_degree)
    return lsq
