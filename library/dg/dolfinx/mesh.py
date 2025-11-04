import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio

# import gmsh
# import dolfinx
from dolfinx import fem
import basix
import ufl

# import pyvista
# import sys
# from dolfinx import mesh
from dolfinx import mesh as dolfinx_mesh

import numpy.typing as npt


from library.zoomy_core.mesh.mesh import Mesh


def compute_facet_distances(mesh: mesh.Mesh) -> fem.Function:
    """
    Compute distance from each facet center to its adjacent cell center.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The DOLFINx mesh.

    Returns
    -------
    distance : dolfinx.fem.Function
        A DG0 function defined on facets (fdim) with the distances.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    coords = mesh.geometry.x

    # --- Cell centers ---
    cell2v = mesh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    cell_coords = coords[cell2v, :]  # (num_cells, tdim+1, tdim)
    cell_centers = cell_coords.mean(axis=1)  # (num_cells, tdim)

    # --- Facet centers ---
    facet2v = mesh.topology.connectivity(fdim, 0).array.reshape(-1, tdim)
    facet_coords = coords[facet2v, :]  # (num_facets, tdim, tdim)
    facet_centers = facet_coords.mean(axis=1)  # (num_facets, tdim)

    # --- Map each facet to a neighboring cell ---
    facet2cell_conn = mesh.topology.connectivity(fdim, tdim)
    facet_to_cell = np.array(
        [c[0] if len(c) > 0 else -1 for c in facet2cell_conn.links()]
    )
    valid_facets = facet_to_cell >= 0

    # --- Compute distances ---
    facet_distances = np.zeros(facet_centers.shape[0], dtype=np.float64)
    facet_distances[valid_facets] = np.linalg.norm(
        facet_centers[valid_facets] - cell_centers[facet_to_cell[valid_facets]], axis=1
    )

    # --- Store in DG0 function ---
    V_facets = fem.FunctionSpace(mesh, ("DG", 0))
    distance = fem.Function(V_facets, name="distance")
    distance.x.array[:] = facet_distances
    distance.x.scatter_forward()

    return distance


def load_mesh(path_to_mesh):
    mesh = Mesh.from_gmsh(path_to_mesh)
    min_inradius = np.min(mesh.cell_inradius)
    tags = [int(v) for v in mesh.boundary_conditions_sorted_physical_tags]
    tags.sort()
    map_tag_to_id = {v: i for i, v in enumerate(tags)}

    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        path_to_mesh, MPI.COMM_WORLD, 0, gdim=2
    )
    unique_facet_tags = np.unique(facet_tags.values)
    facet_boundary_function_id = np.array(
        [map_tag_to_id[tag] for tag in facet_tags.values[:]]
    )
    return (
        mesh,
        cell_tags,
        facet_tags,
        unique_facet_tags,
        facet_boundary_function_id,
        min_inradius,
    )


def evaluate_on_all_interior_facets_midpoint(
    expr: ufl.core.expr.Expr, domain: dolfinx_mesh.Mesh, side: str = "+"
) -> tuple[
    npt.NDArray[np.floating],  # X_mid  (N, gdim)
    npt.NDArray[np.inexact],
]:  # values (N, …)
    """
    Mid-point evaluation of `expr` on *all* interior facets owned by this
    MPI rank.

    Parameters
    ----------
    expr   : UFL expression (may contain u("+"), u("-"), jump(u), …)
    domain : dolfinx mesh
    side   : "+"  – take value from the first adjacent cell  (default)
             "-"  – take value from the second adjacent cell
             "avg" – arithmetic mean of + and -

    Returns
    -------
    X_mid  : coordinates of facet mid-points
    values : expression evaluated according to `side`
    """

    assert side in {"+", "-", "avg"}

    tdim, fdim = domain.topology.dim, domain.topology.dim - 1

    # --- identify interior facets -------------------------------------
    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    n_loc_facets = domain.topology.index_map(fdim).size_local

    interior_facets = np.array(
        [f for f in range(n_loc_facets) if len(f_to_c.links(f)) == 2], dtype=np.int32
    )

    if interior_facets.size == 0:
        # nothing on this rank
        shape_val = expr.ufl_shape or ()
        return (np.empty((0, domain.geometry.dim)), np.empty((0, *shape_val)))

    # --- reference mid-point of one facet -----------------------------
    facet_type = basix.cell.subentity_types(domain.basix_cell())[fdim][0]
    xi_facet_mid = basix.cell.geometry(facet_type).mean(axis=0)

    xi_cell = np.zeros((1, tdim))
    xi_cell[0, : xi_facet_mid.shape[0]] = xi_facet_mid

    expr_comp = fem.Expression(expr, xi_cell)
    coord_comp = fem.Expression(ufl.SpatialCoordinate(domain), xi_cell)

    # --- (cell, local_facet) pairs for interior facets ----------------
    cell_facet = fem.compute_integration_domains(
        fem.IntegralType.interior_facet, domain.topology, interior_facets, fdim
    )

    # --- evaluate (+ and - in one call) --------------------------------
    values_2 = expr_comp.eval(domain, cell_facet)  # shape (2N, …)
    X_2 = coord_comp.eval(domain, cell_facet)  # shape (2N, gdim)

    # UFL/FFC order: first cell → “+”, second cell → “-”
    val_plus = values_2[0::2]
    val_minus = values_2[1::2]
    X_mid = X_2[0::2]  # same coordinates twice

    if side == "+":
        values = val_plus
    elif side == "-":
        values = val_minus
    else:  # "avg"
        values = 0.5 * (val_plus + val_minus)

    return X_mid, values


def evaluate_on_all_exterior_facets_midpoint(
    expr: ufl.core.expr.Expr, domain: dolfinx_mesh.Mesh
) -> tuple[
    npt.NDArray[np.floating],  # X_mid  (N, gdim)
    npt.NDArray[np.inexact],
]:  # values (N, …)
    """
    Mid-point evaluation of `expr` on all boundary (exterior) facets
    owned by this rank.
    """

    tdim, fdim = domain.topology.dim, domain.topology.dim - 1

    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    n_loc_facets = domain.topology.index_map(fdim).size_local

    exterior_facets = np.array(
        [f for f in range(n_loc_facets) if len(f_to_c.links(f)) == 1], dtype=np.int32
    )

    if exterior_facets.size == 0:
        shape_val = expr.ufl_shape or ()
        return (np.empty((0, domain.geometry.dim)), np.empty((0, *shape_val)))

    facet_type = basix.cell.subentity_types(domain.basix_cell())[fdim][0]
    xi_facet_mid = basix.cell.geometry(facet_type).mean(axis=0)

    xi_cell = np.zeros((1, tdim))
    xi_cell[0, : xi_facet_mid.shape[0]] = xi_facet_mid

    expr_comp = fem.Expression(expr, xi_cell)
    coord_comp = fem.Expression(ufl.SpatialCoordinate(domain), xi_cell)

    cell_facet = fem.compute_integration_domains(
        fem.IntegralType.exterior_facet, domain.topology, exterior_facets, fdim
    )

    values = expr_comp.eval(domain, cell_facet)  # shape (N, …)
    X_mid = coord_comp.eval(domain, cell_facet)  # shape (N, gdim)

    return X_mid, values


def evaluate_on_all_facets_midpoint(
    expr: ufl.core.expr.Expr, domain: dolfinx_mesh.Mesh, side: str = "+"
) -> tuple[
    npt.NDArray[np.floating],  # X_mid_all  (N_tot, gdim)
    npt.NDArray[np.inexact],
]:  # values_all (N_tot, …)
    """
    Mid-point evaluation on *all* facets owned by this rank.

    • interior facets – take value according to `side` ("+", "-", "avg")
    • exterior facets – the only available value

    Returns concatenated arrays of coordinates and values.
    """

    X_int, val_int = evaluate_on_all_interior_facets_midpoint(expr, domain, side)
    X_ext, val_ext = evaluate_on_all_exterior_facets_midpoint(expr, domain)

    X_all = np.vstack((X_int, X_ext))
    values = np.concatenate((val_int, val_ext), axis=0)

    return X_all, values
