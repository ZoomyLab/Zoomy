"""The finite-volume vocabulary of the hyperbolic solver.

Each class is one mesh operation the hyperbolic solver needs; every backend
implements a callable of the same name.  Arrays are ``(rows, cells)`` or
``(rows, faces)``; ``faces`` means ALL faces (interior and boundary) in mesh
order.  ``mesh`` is an argument everywhere, so a backend can feed geometry as
a traced, differentiable input.

Ground truth: JAX ``solver_jax.HyperbolicSolver.get_flux_operator`` and
foam ``numerics.H`` / ``numerics_o2.H``.
"""
from zoomy_core.symsolver.core import Block, Effect


# ── kernels on cells and faces ──────────────────────────────────────────────
class cells(Block):
    """``(kernel, Q, Qaux, p, time, dt, mesh)`` -> kernel in every cell.
    ``time``, ``dt`` and the cell centre reach the kernel when it declares
    them."""


class faces(Block):
    """``(kernel, QL, QR, AL, AR, p, mesh)`` -> kernel on every face with that
    face's unit normal (owner -> neighbour; outward at a boundary)."""


class face_aux(Block):
    """``(Qaux, mesh)`` -> ``(AL, AR)``: owner and neighbour aux per face; a
    boundary face takes its inner cell on both sides."""


class face_wavespeed(Block):
    """``(kernel, Q, Qaux, p, mesh)`` -> max |eigenvalue| per face over the
    two adjacent cells (the inner cell twice at a boundary face)."""


class face_inradius(Block):
    """``(mesh)`` -> per-face cell size: min inradius of the adjacent cells."""


# ── boundary ────────────────────────────────────────────────────────────────
# Two BC kernels, two values (never mixed):
#   GHOST  ``nsm.call.boundary_ghost_state`` (decision D5) -- the value at the
#          mirror image of the inner cell centre (2 v - Q for a face value v,
#          Q + 2 d g for a normal gradient g, the reflection / partner as is);
#          evaluated at the foot of the perpendicular with the normal
#          distance d.  Consumers: gradient, limiter bounds, derivative aux,
#          diffusion.
#   FACE   ``nsm.call.boundary_conditions`` -- the outer state ON the face
#          from the inner state there; evaluated at the face centre.
#          Consumer: the Riemann problem (boundary_traces).
class boundary_states(Block):
    """``(ghost, Q, Qaux, p, time, mesh)`` -> ``(Qg, Qauxg)``: the GHOST per
    boundary face (kernel ``ghost`` = ``boundary_ghost_state``, branch by
    tag; from the inner cell, at the foot of the perpendicular with the
    normal distance) and the inner cell's aux."""


class boundary_ghosts(Block):
    """``(ghost, Q, Qaux, p, time, mesh)`` -> the GHOST per boundary face, as
    :class:`boundary_states` without the aux."""


class boundary_traces(Block):
    """``(bc, Q, QL, QR, Qaux, p, time, mesh)`` -> ``QR`` whose boundary
    faces hold the outer FACE state ``bc(QL)`` (kernel ``bc`` =
    ``boundary_conditions``, at the face centre; on a periodic seam: the
    partner face's ``QL``, the partner cell's trace at the seam)."""


# ── reconstruction and assembly ─────────────────────────────────────────────
class pointwise(Block):
    """``(kernel, X, Xaux, p)`` -> the kernel in every column of ``X`` (cells,
    faces or boundary faces); no time, step or position."""


class cell_traces(Block):
    """``(Q, mesh)`` -> ``(QL, QR)``: the owner's and the neighbour's cell
    value per face; ``QR`` on a boundary face is ``QL`` (a placeholder for
    :class:`boundary_traces`)."""


class gradient(Block):
    """``(W, Wg, mesh)`` -> the LSQ gradient of every row of ``W`` per cell
    (backend layout).  A boundary face adds the sample ``Wg`` (its ghost) at
    the mirror image of the cell centre."""


class minmod(Block):
    """``(rows, W, Wg, G, mesh)`` -> ``(rows of W, cells)`` limiter factors
    in [0, 1], 1 off ``rows`` (static): the largest that keeps every face
    extrapolation within the bounds of the face neighbours and the boundary
    ghosts ``Wg`` (at the ghost position, as in :class:`gradient`)."""


class venkatakrishnan(Block):
    """As :class:`minmod`, the smooth Venkatakrishnan factor (eps^2 = h^2)."""


class barth_jespersen(Block):
    """As :class:`minmod`, the Barth-Jespersen factor."""


class extrapolate(Block):
    """``(W, G, phi, mesh)`` -> ``(WL, WR)``: each cell's ``W`` moved to the
    face centre by ``phi * G`` (``phi`` per row and cell, per cell, or a
    number); ``WR`` on a boundary face is ``WL`` (a placeholder for
    :class:`boundary_traces`)."""


class divergence(Block):
    """``(Wm, Wp, mesh)`` -> per cell ``sum_f |f| Wm / |V|`` over the faces it
    owns plus ``|f| Wp / |V|`` over the faces it neighbours (``Wm, Wp`` the
    two rows of ``numerical_face_sides``)."""


class edge_source(Block):
    """``(kernel, Q, QL, QR, Qaux, p, mesh)`` -> per cell the interior
    nonconservative integral over each of its edges (order >= 2)."""


# ── whole-domain operations ─────────────────────────────────────────────────
class derivative_aux(Block):
    """``(spec, Q, Qaux, mesh)`` -> Qaux with the derivative rows of ``spec``
    (a system's ``aux_derivatives``) refilled from mesh gradients; a boundary
    face's sample is the inner cell value (zero normal gradient).
    ``(spec, Q, Qaux, mesh, Qg, rows)``: only the derivatives of state
    ``rows``, a boundary face's sample the GHOST ``Qg`` (see
    :class:`boundary_ghosts`) at the mirror image of the cell centre."""


class put(Block):
    """``(Q, rows, values)`` -> Q with state rows ``rows`` (static) replaced."""


class take(Block):
    """``(Q, rows)`` -> the state rows ``rows`` (static) of ``Q``."""


class zeros(Block):
    """``(n, mesh)`` -> an ``(n, cells)`` array of zeros."""


class diffusion(Block):
    """``(kernel, bc, Q, Qaux, p, time, mesh)`` -> per cell div(A : grad Q)
    of the diffusion tensor ``kernel`` (explicit; RHS sign, A > 0 diffuses).
    Two-point face fluxes ``n.A.n (Q_out - Q_in) / d``: A the mean of the
    two cells' tensors, ``d`` the normal distance between the two values.
    At a boundary face ``Q_out`` is the ghost of the BC ghost kernel ``bc``
    (``nsm.call.boundary_ghost_state``) at ``time``, at the mirror image of
    the cell centre (A the cell's, ``d`` twice its normal distance to the
    face); across a periodic seam the partner cell; a cut face
    (distributed) carries no flux."""


class implicit_diffusion(Block):
    """``(kernel, bc, Q, Qaux, p, time, dt, mesh)`` -> one Crank-Nicolson
    step from ``time`` to ``time + dt`` of the diffusion tensor ``kernel``
    (as :class:`diffusion`; the ghosts of ``Q`` at ``time``, of the new
    state at ``time + dt``)."""


class inadmissible(Block):
    """``(kernel, Q, Qaux, p)`` -> cell mask: an admissibility row < 0 or a
    non-finite value."""


class clean_round_off(Block):
    """``(Q, rows, tol)`` -> ``Q`` with the entries of the static ``rows`` in
    ``[-tol, 0)`` set to 0: round-off below zero of a quantity the model
    declares non-negative (its depth role)."""


class halo(Block):
    """``(X, mesh)`` -> ``X`` with the copies of other ranks' cells current
    (a distributed run); serial: ``X``.  After every stencil update."""


class minimum(Block):
    """``(a, b)`` -> elementwise minimum, NaN propagating.  For graph-valued
    arguments: ``sp.Min`` sorts its arguments by node count, which walks a
    shared graph as a tree.  Real (a time step compares with it)."""
    is_commutative = True
    is_extended_real = True


class reduce_min(Block):
    """Global minimum (all ranks)."""


class reduce_max(Block):
    """``(X)`` -> the largest entry (NaN propagates)."""


class reduce_any(Block):
    """Global any (all ranks)."""


class select(Block):
    """``(mask, A, B)`` -> A where mask else B, per cell or per face."""


# ── coupling geometry (preCICE) ─────────────────────────────────────────────
class coupled_faces(Block):
    """``(mesh)`` -> face mask of the coupled boundary faces."""


class interface_states(Block):
    """``(Q, mesh)`` -> states of the cells next to the coupled faces."""


class interface_flux(Block):
    """``(sides, aux, side, Q, Qaux, peer, p, mesh)`` -> per face this
    participant's row of ONE Riemann problem posed low -> high (``sides`` =
    ``numerical_face_sides``; row 0 for the low side, ``side = 0``, row 1 for
    the high side); zero off the coupled faces.  Both states take their aux
    from the pointwise kernel ``aux`` alone (``false``: none), so both sides
    pose the same problem."""


# ── effects ─────────────────────────────────────────────────────────────────
class io:
    write = Effect()             # (time, Q, Qaux)


class couple:                    # one preCICE participant
    init = Effect(1)             # (q_interface) -> coupling ongoing
    max_dt = Effect(1)           # () -> time left in the window
    read = Effect(1)             # () -> peer interface states
    write = Effect()             # (q_interface)
    advance = Effect(1)          # (dt) -> coupling ongoing


class check:
    cfl = Effect()               # (dt_cfl, dt): warn once if dt > dt_cfl
