import numpy as np
from dolfinx import fem, mesh

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
    cell_coords = coords[cell2v, :]            # (num_cells, tdim+1, tdim)
    cell_centers = cell_coords.mean(axis=1)    # (num_cells, tdim)

    # --- Facet centers ---
    facet2v = mesh.topology.connectivity(fdim, 0).array.reshape(-1, tdim)
    facet_coords = coords[facet2v, :]          # (num_facets, tdim, tdim)
    facet_centers = facet_coords.mean(axis=1)  # (num_facets, tdim)

    # --- Map each facet to a neighboring cell ---
    facet2cell_conn = mesh.topology.connectivity(fdim, tdim)
    facet_to_cell = np.array([c[0] if len(c) > 0 else -1 for c in facet2cell_conn.links()])
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
