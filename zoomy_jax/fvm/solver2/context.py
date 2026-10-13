"""`prepare_mesh` / `build_operators` — the two translation blocks.

Design §2:

    prepare_mesh(mesh, nsm)          -> MeshRT
    build_operators(kernels, MeshRT) -> Ops

These are the ONLY blocks that touch the backend's own containers; everything
downstream speaks `(Q, Qaux, p, t, dt)`.  Both are pure re-packaging of the
arrays and callables the existing jax solver already builds — no new physics,
no new kernels.  In particular:

* the per-face index arrays are exactly the ones `solver_jax.get_flux_operator`
  precomputes (`interior_faces` by set-difference, `bf_face_idx`,
  the periodic-seam mask of REQ-116, the boundary face-to-cell distance);
* `Ops` holds the *unchanged* `JaxRuntime`
  slots (`numerical_flux`, `numerical_fluctuations`, `eigenvalues`,
  `source`, `boundary_conditions`, `nonconservative_matrix`,
  `update_variables`, `update_aux_variables`) and the *unchanged*
  reconstruction objects from `reconstruction_jax`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from zoomy_core.model.boundary_conditions import boundary_face_branches


# ── Shu-Osher stage tableaus (core data; §2 "tableau") ──────────────────────
# Q^(k) = alpha_k * Q^n + (1 - alpha_k) * (Q^(k-1) + beta_k * dt * L(Q^(k-1)))
TABLEAU_EULER = ((0.0, 1.0),)                       # explicit Euler
TABLEAU_SSPRK2 = ((0.0, 1.0), (0.5, 1.0))           # SSP-RK2 (Heun)


@dataclass(frozen=True)
class MeshRT:
    """Runtime mesh view (design §2).  Plain python object: it is *closed over*
    by the jitted step, so its jnp arrays become trace-time constants."""

    dimension: int
    n_cells: int
    n_faces: int
    n_boundary_faces: int

    # face topology
    face_owner: Any          # (F,)  owner cell of every face
    face_neigh: Any          # (F,)  neighbour cell; == owner on boundary faces
    interior_faces: Any      # (F_int,)
    boundary_faces: Any      # (F_bnd,)  face index of each boundary face
    iA_int: Any
    iB_int: Any
    iInner_bnd: Any

    # geometry
    face_normals: Any        # (3, F)
    face_volumes: Any        # (F,)
    face_centers: Any        # (F, 3)
    cell_volumes: Any        # (C,)
    cell_centers: Any        # (3, C)
    inradius_f: Any          # (F,)  min inradius of the two adjacent cells

    # boundary bookkeeping
    boundary_face_cells: Any
    bf_branch: Any           # BC-kernel branch of each boundary face (by tag name)
    bf_distance: Any
    periodic_mask: Any
    has_periodic: bool

    mesh: Any = None         # the jax mesh, kept for io / aux gradients
    mesh_np: Any = None      # the source LSQMesh, kept for hdf5 output


@dataclass
class Ops:
    """Operator bundle (design §2).  Every entry is an EXISTING kernel."""

    flux_face: Any               # runtime.numerical_flux
    fluct_face: Any              # runtime.numerical_fluctuations
    bc_face: Any                 # runtime.boundary_conditions
    eigenvalues: Any             # runtime.eigenvalues
    reconstruct: Any             # reconstruction object (order-dependent)
    reconstruct_o1: Any          # ConstantReconstruction — the MOOD fallback
    source: Any                  # runtime.source (or None)
    nonconservative_matrix: Any  # runtime.nonconservative_matrix (or None)
    update_variables: Any        # runtime.update_variables (or None)
    update_aux_variables: Any    # runtime.update_aux_variables (or None)
    aux_registry_walk: Any       # HyperbolicSolver._walk_derivative_aux
    sm: Any
    tableau: Any
    order: int
    use_interior_ncp: bool
    h_index: Optional[int]
    n_state: int


def prepare_mesh(mesh, nsm, mesh_np=None) -> MeshRT:
    """Translate a `MeshJAX` into the contract's `MeshRT`."""
    fc0 = np.asarray(mesh.face_cells[0])
    fc1 = np.asarray(mesh.face_cells[1])
    n_faces = int(mesh.n_faces)
    n_bf = int(getattr(mesh, "n_boundary_faces", 0))
    bf_face_idx = (np.asarray(mesh.boundary_face_face_indices)
                   if n_bf > 0 else np.zeros(0, dtype=np.int32))
    bf_set = {int(f) for f in bf_face_idx}
    interior_faces = np.array(
        [f for f in range(n_faces) if f not in bf_set], dtype=np.int32)

    # Boundary faces have no second cell inside the inner block; use the OWNER
    # on both sides so a per-face two-sided sweep (dt_pass) never gathers
    # out-of-bounds.  ``jnp`` would clamp such an index to the last cell — an
    # arbitrary state that would silently enter the CFL minimum.
    face_neigh = fc1.copy()
    if n_bf > 0:
        face_neigh[bf_face_idx] = fc0[bf_face_idx]

    inradius = np.asarray(mesh.cell_inradius)
    inradius_f = np.minimum(inradius[fc0], inradius[face_neigh])

    # Periodic seam (REQ-116): ``boundary_face_cells`` was remapped to the
    # PARTNER cell, so it differs from the this-side cell exactly at the seam.
    periodic_np = (np.asarray(mesh.boundary_face_cells) != fc0[bf_face_idx]
                   if n_bf > 0 else np.zeros(0, dtype=bool))

    if n_bf > 0:
        d_face = np.asarray([
            np.linalg.norm(
                np.asarray(mesh.face_centers)[bf_face_idx[i], :]
                - np.asarray(mesh.cell_centers)[:, mesh.boundary_face_cells[i]])
            for i in range(n_bf)])
    else:
        d_face = np.zeros(0)

    return MeshRT(
        dimension=int(mesh.dimension),
        n_cells=int(mesh.n_inner_cells),
        n_faces=n_faces,
        n_boundary_faces=n_bf,
        face_owner=jnp.asarray(fc0),
        face_neigh=jnp.asarray(face_neigh),
        interior_faces=jnp.asarray(interior_faces),
        boundary_faces=jnp.asarray(bf_face_idx, dtype=jnp.int32),
        iA_int=jnp.asarray(fc0[interior_faces]),
        iB_int=jnp.asarray(fc1[interior_faces]),
        iInner_bnd=jnp.asarray(fc0[bf_face_idx]),
        face_normals=jnp.asarray(mesh.face_normals),
        face_volumes=jnp.asarray(mesh.face_volumes),
        face_centers=jnp.asarray(mesh.face_centers),
        cell_volumes=jnp.asarray(mesh.cell_volumes),
        cell_centers=jnp.asarray(mesh.cell_centers),
        inradius_f=jnp.asarray(inradius_f),
        boundary_face_cells=jnp.asarray(mesh.boundary_face_cells,
                                        dtype=jnp.int32),
        bf_branch=jnp.asarray(
            boundary_face_branches(mesh, getattr(nsm.sm, "_bc_source", None)),
            dtype=jnp.int32),
        bf_distance=jnp.asarray(d_face),
        periodic_mask=jnp.asarray(periodic_np),
        has_periodic=bool(periodic_np.any()),
        mesh=mesh,
        mesh_np=mesh_np,
    )


def build_operators(runtime, MeshRT_, *, reconstruct, order, h_index,
                    aux_registry_walk) -> Ops:
    """Bundle the EXISTING runtime slots + reconstruction into `Ops`."""
    from zoomy_jax.fvm.reconstruction_jax import ConstantReconstruction

    rt_ncm = getattr(runtime, "nonconservative_matrix", None)
    # A class opts in by providing its OWN reconstruct_with_grad, or by
    # declaring supports_grad_recon in its own class body.  INHERITING either
    # is not opting in: a subclass whose positivity / wet-dry treatment lives
    # only in __call__ would otherwise route the order>=2 interior-NCP path
    # through the untreated base reconstruction (measured on production:
    # min h_face = -3.1e-03, negative depth).  Must stay identical to the
    # guard at solver_jax.py:592-601 — a hasattr() test here silently accepts
    # exactly the case production rejects.
    _cls = type(reconstruct)
    _grad_recon_ok = ("reconstruct_with_grad" in _cls.__dict__
                      or bool(_cls.__dict__.get("supports_grad_recon", False)))
    use_interior_ncp = bool(order >= 2 and rt_ncm is not None
                            and _grad_recon_ok)
    if order >= 2 and rt_ncm is not None and not use_interior_ncp:
        # Silently dropping the interior NCP at order >= 2 loses
        # well-balancing for NCP-bearing models — same guard as solver_jax.
        raise NotImplementedError(
            f"order-{order} with a nonzero nonconservative_matrix requires a "
            f"reconstruction exposing reconstruct_with_grad (limited cell "
            f"gradient) for the cell-interior NCP integral; "
            f"{type(reconstruct).__name__} does not.")

    return Ops(
        flux_face=runtime.numerical_flux,
        fluct_face=runtime.numerical_fluctuations,
        bc_face=runtime.boundary_conditions,
        eigenvalues=runtime.eigenvalues,
        reconstruct=reconstruct,
        reconstruct_o1=ConstantReconstruction(MeshRT_.mesh, MeshRT_.dimension),
        source=getattr(runtime, "source", None),
        nonconservative_matrix=rt_ncm,
        update_variables=getattr(runtime, "update_variables", None),
        update_aux_variables=getattr(runtime, "update_aux_variables", None),
        aux_registry_walk=aux_registry_walk,
        sm=getattr(runtime, "sm", None),
        tableau=(TABLEAU_SSPRK2 if order >= 2 else TABLEAU_EULER),
        order=int(order),
        use_interior_ncp=use_interior_ncp,
        h_index=h_index,
        n_state=int(runtime.n_state),
    )
