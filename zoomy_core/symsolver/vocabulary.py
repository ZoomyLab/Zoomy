"""The finite-volume vocabulary of the hyperbolic solver: one `Block` per mesh
operation, implemented under the same name by every backend. Arrays are
`(rows, cells)` or `(rows, faces)`, `faces` meaning all faces in mesh order,
and `mesh` is always an argument.
"""
import math

from zoomy_core.symsolver.core import Block, Effect


# ── kernels on cells and faces ──────────────────────────────────────────────
class cells(Block):
    """`(kernel, Q, Qaux, p, time, dt, mesh)` -> kernel in every cell, with `time`,
    `dt` and the cell centre passed when the kernel declares them.
    """


class faces(Block):
    """`(kernel, QL, QR, AL, AR, p, mesh)` -> kernel on every face with its unit
    normal (owner to neighbour, outward at a boundary).
    """


class face_aux(Block):
    """`(Qaux, mesh)` -> `(AL, AR)`, the owner and neighbour aux per face (the inner
    cell on both sides at a boundary).
    """


class face_wavespeed(Block):
    """`(kernel, Q, Qaux, p, mesh)` -> max `|eigenvalue|` per face over its two cells.
    """


class face_inradius(Block):
    """`(mesh)` -> per-face cell size: min inradius of the adjacent cells."""


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
    """`(ghost, Q, Qaux, p, time, mesh)` -> `(Qg, Qauxg)`: the ghost state per
    boundary face (kernel `boundary_ghost_state`) and the inner cell's aux.
    """


class boundary_ghosts(Block):
    """`(ghost, Q, Qaux, p, time, mesh)` -> the ghost per boundary face, as
    `boundary_states` without the aux.
    """


class boundary_traces(Block):
    """`(bc, Q, QL, QR, Qaux, p, time, mesh)` -> `QR` holding the outer face state
    `bc(QL)` on boundary faces (the partner face's `QL` on a periodic seam).
    """


# ── reconstruction and assembly ─────────────────────────────────────────────
class pointwise(Block):
    """`(kernel, X, Xaux, p)` -> the kernel in every column of `X` (cells, faces or
    boundary faces); no time, step or position.
    """


class cell_traces(Block):
    """`(Q, mesh)` -> `(QL, QR)`, the owner's and neighbour's cell value per face;
    `QR` on a boundary face is a placeholder for `boundary_traces`.
    """


class gradient(Block):
    """`(W, Wg, mesh)` -> LSQ gradient of every row of `W` per cell, boundary faces
    closed from the ghost values `Wg`.  `(W, Wg, (eta, h), eps, mesh)`: in row `eta` a
    neighbour `k` with `W[h, k] <= eps` enters cell `i`'s stencil as `min(W[eta, k],
    W[eta, i])` (the hydrostatic extension of the free surface, `reconstruction.
    shore_extension`).

    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """


class minmod(Block):
    """`(rows, W, Wg, G, mesh)` -> limiter factors in [0, 1] per row and cell (1 off
    `rows`): the largest keeping every face extrapolation within the bounds of the
    face neighbours and boundary ghosts `Wg`.  `(rows, W, Wg, G, (eta, h), eps, mesh)`:
    the bounds of row `eta` from the neighbour values of `gradient`'s shore form.

    References:
        Sweby 1984, https://doi.org/10.1137/0721062
    """


class venkatakrishnan(Block):
    """As `minmod`, the smooth Venkatakrishnan factor (eps^2 = h^2).

    References:
        Venkatakrishnan 1995, https://doi.org/10.1006/jcph.1995.1084
    """


class barth_jespersen(Block):
    """As `minmod`, the Barth-Jespersen factor.

    References:
        Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
    """


class front_cells(Block):
    """`(Q, G, phi, (depth, w), tol, mesh)` -> cell mask: the cells whose depth
    `h_f = Q[depth] + phi[depth] G[w] . r_f` falls below `-(1/tol - 1) Q[depth]` at a face
    `f` (`G` the gradients of the reconstruction variables, `phi` their limiter factors,
    `r_f` from the cell centre to the face centre), i.e. whose positivity factor
    `h / (h - min_f h_f)` is below `tol`.

    References:
        Gallardo, Pares, Castro 2007, https://doi.org/10.1016/j.jcp.2007.08.007
    """


class extrapolate(Block):
    """`(W, G, phi, mesh)` -> `(WL, WR)`: each cell's `W` moved to the face centre
    by `phi * G`; `WR` on a boundary face is a placeholder for
    `boundary_traces`.

    References:
        van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
    """


class divergence(Block):
    """`(Wm, Wp, mesh)` -> per cell `sum_f |f| Wm / |V|` over the faces it owns plus
    `|f| Wp / |V|` over the faces it neighbours.
    """


class edge_source(Block):
    """`(kernel, Q, QL, QR, Qaux, p, mesh)` -> per cell the interior nonconservative
    integral over each of its edges (order >= 2).

    References:
        Pares 2006, https://doi.org/10.1137/050628052
    """


# ── whole-domain operations ─────────────────────────────────────────────────
class derivative_aux(Block):
    """`(spec, Q, Qaux, mesh[, Qg, rows])` -> Qaux with the derivative rows of
    `spec` refilled from mesh gradients; with `rows`, only those states'
    derivatives, their boundary faces taken from the ghosts `Qg`.
    """


class put(Block):
    """`(Q, rows, values)` -> Q with the static state rows `rows` replaced."""


class take(Block):
    """`(Q, rows)` -> the static state rows `rows` of `Q`."""


class zeros(Block):
    """`(n, mesh)` -> an `(n, cells)` array of zeros."""


class diffusion(Block):
    """`(kernel, bc, Q, Qaux, p, time, mesh)` -> per cell div(A grad Q) of the
    diffusion tensor `kernel` (explicit; RHS sign, A > 0 diffuses), face flux
    `n.A.grad_f`: the two-point flux plus a correction, `grad_f` the mean of the
    two cells' LSQ gradients with the two-point difference as its part along the
    cell-to-cell vector (boundary faces: the ghost of `bc`, the tangential part from
    the cell gradient). A cut face (distributed) carries no flux.

    References:
        Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
    """


class diffusion_tpfa(Block):
    """`(kernel, bc, Q, Qaux, p, time, mesh)` -> per cell the two-point part of
    `diffusion`: per face `n.A.n (Q_out - Q_in) / d`, `A` the mean of the two cells'
    tensors, `Q_out` the BC ghost across a boundary face, the partner cell across a
    periodic seam.

    References:
        Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
    """


class diffusion_correction(Block):
    """`(kernel, bc, Q, Qaux, p, time, mesh)` -> per cell the rest of `diffusion`
    beyond `diffusion_tpfa` (the tangential and non-orthogonal part of the face flux;
    zero in 1-D), so that `diffusion = diffusion_tpfa + diffusion_correction`.

    References:
        Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
    """


class implicit_diffusion(Block):
    """`(kernel, bc, Q, Qaux, p, time, dt, mesh)` -> one Crank-Nicolson step of
    the two-point part of `diffusion` from `time` to `time + dt`, its
    correction explicit at `Q` (deferred correction).

    References:
        Crank, Nicolson 1947, https://doi.org/10.1017/s0305004100023197
    """


class inadmissible(Block):
    """`(kernel, Q, Qaux, p)` -> cell mask: an admissibility row < 0 or a non-finite
    value.
    """


class clean_round_off(Block):
    """`(Q, rows, tol)` -> `Q` with the entries of the static `rows` in
    `[-tol, 0)` set to 0 (round-off below zero of a quantity declared
    non-negative).
    """


class halo(Block):
    """`(X, mesh)` -> `X` with the copies of other ranks' cells current (serial:
    `X`); after every stencil update.
    """


class minimum(Block):
    """`(a, b)` -> elementwise minimum, NaN propagating, for graph-valued arguments
    (`sp.Min` would walk a shared graph as a tree).
    """
    is_commutative = True
    is_extended_real = True


class reduce_min(Block):
    """Global minimum (all ranks)."""


class reduce_max(Block):
    """`(X)` -> the largest entry (NaN propagates)."""


class reduce_any(Block):
    """Global any (all ranks)."""


class select(Block):
    """`(mask, A, B)` -> A where mask else B, per cell or per face."""


# ── coupling geometry (preCICE) ─────────────────────────────────────────────
class coupled_faces(Block):
    """`(mesh)` -> face mask of the coupled boundary faces."""


class interface_states(Block):
    """`(Q, mesh)` -> states of the cells next to the coupled faces."""


class interface_flux(Block):
    """`(sides, aux, side, Q, Qaux, peer, p, mesh)` -> per face this participant's row
    of one Riemann problem posed low -> high (row 0 for `side = 0`), zero off the
    coupled faces; both sides pose the same problem.
    """


class lifted_interface_flux(Block):
    """`(sides, aux, n_aux, side, own, peer, rows, p, mesh)` -> per face the rows `rows` of
    this participant's side of one Riemann problem posed low -> high in the model of `sides`
    (the interface model) on the interface states `own`, `peer` (`(rows of that model, coupled
    faces)`), the aux rebuilt from each state alone (`aux` on a zero aux of `n_aux` rows; `aux`
    False: the zero aux); zero off the coupled faces. Both sides pose the same problem.
    """


class characteristic_ghost(Block):
    """`(A, own, Aown, peer, p, mesh)` -> per coupled face the transmission state towards a
    peer of another model: with `A_n = sum_d A(own, Aown)[:, :, d] n_d` (`A` the quasilinear
    matrix kernel, `n` the outward normal) the characteristics `l_k . q` that enter
    (`lambda_k < 0`) from `peer`, those that leave from `own`; `peer` where the eigenvectors
    of `A_n` are singular, `own` where `peer` is all zero (no peer data yet). `own`, `Aown`,
    `peer` and the result are `(rows, coupled faces)`.
    """


class ghost_flux(Block):
    """`(sides, aux, own, Aown, ghost, p, mesh)` -> per face the own cell's row of the
    Riemann problem `(own, ghost)` posed along the outward normal of the coupled faces, zero
    elsewhere; `Aown` on the own side, `aux(ghost)` on the other (the pointwise aux formula on
    a zero aux; `aux` False: `Aown`). `own`, `Aown`, `ghost` are `(rows, coupled faces)`.
    """


class replace_row(Block):
    """`(A, i, B)` -> `A` with row `i` taken from `B`."""


class coupled_ghosts(Block):
    """`(Qg, own, face, mesh)` -> the boundary ghosts `Qg` with the coupled faces' replaced by
    `2 face - own` (`own`, `face`: `(rows, coupled faces)`)."""


# ── effects ─────────────────────────────────────────────────────────────────
class io:
    """Output effects; `save_state` / `load_state`: the restart pair of the march state."""
    write = Effect()             # (time, Q, Qaux)
    save_state = Effect()        # (time, Q, Qaux): write the march state
    load_state = Effect(3)       # (time, Q, Qaux) -> the saved state when restarting, else these


class couple:                    # one preCICE participant
    """Effects of one preCICE participant.

    References:
        Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
    """
    init = Effect(1)             # (q_interface[, qaux_interface]) -> coupling ongoing
    max_dt = Effect(1)           # () -> time left in the window
    read = Effect(1)             # () -> peer interface states
    write = Effect()             # (q_interface[, qaux_interface])
    advance = Effect(1)          # (dt) -> coupling ongoing
    window_complete = Effect(1)  # () -> the last advance completed a time window
    read_round_trip = Effect(1)  # () -> the own interface states as the peer reads them
                                 #    (in the interface model of a lifted shared face)


class check:
    """Run-time checks."""
    cfl = Effect()               # (dt_cfl, dt): warn once if dt > dt_cfl
    # (time, time_end, dt, n_step, max |Q|) after the march: raise unless time >= time_end
    # with a finite state (the loop stops at a step that does not advance the time)
    march = Effect()


def march_error(time, time_end, dt, n_step, q_max):
    """The message of a failed `check.march`, or None: the march stopped before `time_end` (a
    step that does not advance the time) or ended with a non-finite state (`q_max = max |Q|`).
    The C++ runtimes print the same text (`rt_common.H`).
    """
    time, time_end, dt, q_max = map(float, (time, time_end, dt, q_max))
    if time >= time_end and math.isfinite(q_max):
        return None
    if not time >= time_end:
        state = (f"the state at t is finite (max |Q| = {q_max:.6g}): its CFL step is not a "
                 "positive number (a NaN or infinite wave speed)" if math.isfinite(q_max) else
                 f"the state at t is not finite (max |Q| = {q_max}): "
                 + (f"step {int(n_step) - 1} produced it" if n_step > 1 else
                    "it is the initial state"))
        return (f"zoomy march stopped at t = {time:.9g} (t_end = {time_end:.9g}): step "
                f"{int(n_step)} has dt = {dt:.6g}, which does not advance the time; {state}")
    return (f"zoomy march reached t = {time:.9g} after {int(n_step)} steps with a non-finite "
            f"state (max |Q| = {q_max})")
