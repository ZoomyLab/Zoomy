"""The finite-volume vocabulary of the hyperbolic solver: one :class:`Block` per mesh
operation, implemented under the same name by every backend. Arrays are
``(rows, cells)`` or ``(rows, faces)``, ``faces`` meaning all faces in mesh order,
and ``mesh`` is always an argument.
"""
from zoomy_core.symsolver.core import Block, Effect


# ── kernels on cells and faces ──────────────────────────────────────────────
class cells(Block):
    """``(kernel, Q, Qaux, p, time, dt, mesh)`` -> kernel in every cell, with ``time``,
    ``dt`` and the cell centre passed when the kernel declares them.
    """


class faces(Block):
    """``(kernel, QL, QR, AL, AR, p, mesh)`` -> kernel on every face with its unit
    normal (owner to neighbour, outward at a boundary).
    """


class face_aux(Block):
    """``(Qaux, mesh)`` -> ``(AL, AR)``, the owner and neighbour aux per face (the inner
    cell on both sides at a boundary).
    """


class face_wavespeed(Block):
    """``(kernel, Q, Qaux, p, mesh)`` -> max |eigenvalue| per face over its two cells.
    """


class face_inradius(Block):
    """``(mesh)`` -> per-face cell size: min inradius of the adjacent cells."""


# ── boundary ────────────────────────────────────────────────────────────────
# Two BC kernels, two values (never mixed):
#   GHOST  ``nsm.call.boundary_ghost_state`` (decision D5) -- the value at the
#          mirror image of the inner cell centre (2 v - Q for a face value v,
#          Q + 2 d g for a normal gradient g, the reflection / partner as is);
#          evaluated at the foot of the perpendicular with the normal
#          distance d.  Consumers: gradient, limiter bounds, derivative aux,
#          diffusion.  The LSQ fits read a Dirichlet- / Neumann-type row's
#          ghost as the datum of a constraint at the foot point (ANALYSIS
#          40, zoomy_core.numerics.boundary_closure).
#   FACE   ``nsm.call.boundary_conditions`` -- the outer state ON the face
#          from the inner state there; evaluated at the face centre.
#          Consumer: the Riemann problem (boundary_traces).
class boundary_states(Block):
    """``(ghost, Q, Qaux, p, time, mesh)`` -> ``(Qg, Qauxg)``: the ghost state per
    boundary face (kernel ``boundary_ghost_state``) and the inner cell's aux.
    """


class boundary_ghosts(Block):
    """``(ghost, Q, Qaux, p, time, mesh)`` -> the ghost per boundary face, as
    :class:`boundary_states` without the aux.
    """


class boundary_traces(Block):
    """``(bc, Q, QL, QR, Qaux, p, time, mesh)`` -> ``QR`` holding the outer face state
    ``bc(QL)`` on boundary faces (the partner face's ``QL`` on a periodic seam).
    """


# ── reconstruction and assembly ─────────────────────────────────────────────
class pointwise(Block):
    """``(kernel, X, Xaux, p)`` -> the kernel in every column of ``X`` (cells, faces or
    boundary faces); no time, step or position.
    """


class cell_traces(Block):
    """``(Q, mesh)`` -> ``(QL, QR)``, the owner's and neighbour's cell value per face;
    ``QR`` on a boundary face is a placeholder for :class:`boundary_traces`.
    """


class gradient(Block):
    """``(W, Wg, mesh)`` -> LSQ gradient of every row of ``W`` per cell, boundary faces
    closed from the ghost values ``Wg``.

    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """


class minmod(Block):
    """``(rows, W, Wg, G, mesh)`` -> limiter factors in [0, 1] per row and cell (1 off
    ``rows``): the largest keeping every face extrapolation within the bounds of the
    face neighbours and boundary ghosts ``Wg``.
    """


class venkatakrishnan(Block):
    """As :class:`minmod`, the smooth Venkatakrishnan factor (eps^2 = h^2)."""


class barth_jespersen(Block):
    """As :class:`minmod`, the Barth-Jespersen factor.

    References:
        Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
    """


class extrapolate(Block):
    """``(W, G, phi, mesh)`` -> ``(WL, WR)``: each cell's ``W`` moved to the face centre
    by ``phi * G``; ``WR`` on a boundary face is a placeholder for
    :class:`boundary_traces`.

    References:
        van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
    """


class divergence(Block):
    """``(Wm, Wp, mesh)`` -> per cell ``sum_f |f| Wm / |V|`` over the faces it owns plus
    ``|f| Wp / |V|`` over the faces it neighbours.
    """


class edge_source(Block):
    """``(kernel, Q, QL, QR, Qaux, p, mesh)`` -> per cell the interior nonconservative
    integral over each of its edges (order >= 2).

    References:
        Pares 2006, https://doi.org/10.1137/050628052
    """


# ── whole-domain operations ─────────────────────────────────────────────────
class derivative_aux(Block):
    """``(spec, Q, Qaux, mesh[, Qg, rows])`` -> Qaux with the derivative rows of
    ``spec`` refilled from mesh gradients; with ``rows``, only those states'
    derivatives, their boundary faces taken from the ghosts ``Qg``.
    """


class put(Block):
    """``(Q, rows, values)`` -> Q with the static state rows ``rows`` replaced."""


class take(Block):
    """``(Q, rows)`` -> the static state rows ``rows`` of ``Q``."""


class zeros(Block):
    """``(n, mesh)`` -> an ``(n, cells)`` array of zeros."""


class diffusion(Block):
    """``(kernel, bc, Q, Qaux, p, time, mesh)`` -> per cell div(A grad Q) of the
    diffusion tensor ``kernel`` (explicit; RHS sign, A > 0 diffuses), by two-point
    face fluxes. The outer value at a boundary face is the ghost of ``bc``, and a cut
    face (distributed) carries no flux.
    """


class implicit_diffusion(Block):
    """``(kernel, bc, Q, Qaux, p, time, dt, mesh)`` -> one Crank-Nicolson step of
    :class:`diffusion` from ``time`` to ``time + dt``.
    """


class inadmissible(Block):
    """``(kernel, Q, Qaux, p)`` -> cell mask: an admissibility row < 0 or a non-finite
    value.
    """


class clean_round_off(Block):
    """``(Q, rows, tol)`` -> ``Q`` with the entries of the static ``rows`` in
    ``[-tol, 0)`` set to 0 (round-off below zero of a quantity declared
    non-negative).
    """


class halo(Block):
    """``(X, mesh)`` -> ``X`` with the copies of other ranks' cells current (serial:
    ``X``); after every stencil update.
    """


class minimum(Block):
    """``(a, b)`` -> elementwise minimum, NaN propagating, for graph-valued arguments
    (``sp.Min`` would walk a shared graph as a tree).
    """
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
    """``(sides, aux, side, Q, Qaux, peer, p, mesh)`` -> per face this participant's row
    of one Riemann problem posed low -> high (row 0 for ``side = 0``), zero off the
    coupled faces; both sides pose the same problem.
    """


# ── effects ─────────────────────────────────────────────────────────────────
class io:
    """Output effects."""
    write = Effect()             # (time, Q, Qaux)


class couple:                    # one preCICE participant
    """Effects of one preCICE participant.

    References:
        Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
    """
    init = Effect(1)             # (q_interface) -> coupling ongoing
    max_dt = Effect(1)           # () -> time left in the window
    read = Effect(1)             # () -> peer interface states
    write = Effect()             # (q_interface)
    advance = Effect(1)          # (dt) -> coupling ongoing


class check:
    """Run-time checks."""
    cfl = Effect()               # (dt_cfl, dt): warn once if dt > dt_cfl
