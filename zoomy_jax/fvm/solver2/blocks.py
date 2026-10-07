"""The v6 march blocks, with the design's names and signatures.

Every block below is a thin SHELL over kernels that already exist in
``zoomy_jax`` — the JaxRuntime slots and the ``reconstruction_jax`` classes.
No scheme decision is hand-written here: well-balancing, positivity and
truncation all live in the symbolic/core layer (the WB face transform inside
``numerical_fluctuations``, the KP ``hinv`` sweep inside ``update_aux``, the
limiter inside the reconstruction object).

Block list (design §2 + v5 amendments + v6 dt ruling)::

    dt_pass(Q, Qaux, p, MeshRT, Ops)        -> lam_lo_f(F), lam_hi_f(F)
    reduce_dt(lam_lo_f, lam_hi_f, inradius_f, dt_max, t_remaining|dt_window)
    halo_bc(Q, Qaux, p, MeshRT, t, Ops)     -> Q, Qaux, bf_values
    reconstruct(Q, Qaux, MeshRT, Ops, bf_values, o1) -> Q_L, Q_R, lim_grad
    flux_pass(Q, Qaux, p, MeshRT, Ops, t_stage, faces) -> Fface, Dp, Dm
    gather_update(Q0, Fface, Dp, Dm, MeshRT, dt, a_stage, source, cell_term)
                                            -> Q_cand, troubled(C)
    mood_resolve(troubled, Q0, Q_cand, Qaux, p, t, dt, MeshRT, Ops) -> Q_next
    update_variables(Q, Qaux, p, time, dt, MeshRT, Ops)
    update_aux(Q, Qaux, p, time, dt, MeshRT, Ops)

Signature deviations from the design text, all minor and deliberate, are
listed in the package ``__init__`` docstring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ══ STEP HEAD: dt (v6 §1 — computed BEFORE the stage loop, then FROZEN) ═════

def dt_pass(Q, Qaux, p, MeshRT, Ops):
    """Evaluate the EXISTING ``eigenvalues`` slot at the CELL states on both
    sides of every face and store TWO scalars per face.

    Returns ``(lam_lo_f, lam_hi_f)``, each ``(n_faces,)`` — the signed lower
    and upper wave-speed bounds of the face.  That is exactly what Rusanov
    (``|lam|max``), HLL (``S_L``/``S_R``) and Kurganov-Petrova (``a±``) need.
    Roe-class schemes needing full eigenvector matrices are not served by the
    store and recompute internally (none in production).

    Boundary faces carry the OWNER cell on both sides (see
    :func:`~zoomy_jax.fvm.solver2.context.prepare_mesh`), so no gather ever
    leaves the inner block.
    """
    iA, iB = MeshRT.face_owner, MeshRT.face_neigh
    normal = MeshRT.face_normals
    evA = Ops.eigenvalues(Q[:, iA], Qaux[:, iA], p, normal)
    evB = Ops.eigenvalues(Q[:, iB], Qaux[:, iB], p, normal)
    if evA.ndim > 1:                       # reduce the eigenvalue axis, keep F
        red = tuple(range(evA.ndim - 1))
        lo = jnp.minimum(jnp.min(evA, axis=red), jnp.min(evB, axis=red))
        hi = jnp.maximum(jnp.max(evA, axis=red), jnp.max(evB, axis=red))
    else:
        lo = jnp.minimum(evA, evB)
        hi = jnp.maximum(evA, evB)
    return lo, hi


def reduce_dt(lam_lo_f, lam_hi_f, inradius_f, dt_max, *,
              CFL, dimension, degree=0, t_remaining=None, dt_window=None):
    """The single dt reduction (design §2, v6 §1).

    ``dt <= CFL * 2 r_in / (d (2k+1) |lam|max)`` per face, minimised over
    faces, then capped by ``dt_max`` and by EITHER the remaining time to
    ``t_end`` OR the coupling window remainder — **never both**: D7 /
    amendment 7 (``dt_window`` REPLACES the ``t_end`` clamp; the min-combined
    form is the measured preCICE abort).

    A wave-free face has ``|lam|max == 0`` and therefore a local limit of
    ``+inf`` — it drops out of the minimum instead of imposing a floor
    (REQ-190); an all-dry domain steps at exactly ``dt_max``.

    The CFL safety factor is the USER LAW (0.9 in 1-D, 0.45 in 2-D) and is
    passed in, never adjusted here.  There is no dt-halving and no retry.
    """
    if t_remaining is not None and dt_window is not None:
        raise ValueError(
            "reduce_dt takes t_remaining XOR dt_window (design D7 / amendment "
            "7): dt_window REPLACES the t_end clamp, it is never min-combined.")

    amax = jnp.maximum(jnp.abs(lam_lo_f), jnp.abs(lam_hi_f))
    h = 2.0 * inradius_f
    dt_local = CFL * h / (float(dimension) * float(2 * degree + 1) * amax)
    dt = jnp.min(dt_local)
    if dt_max is not None:
        dt = jnp.minimum(dt, dt_max)
    clamp = t_remaining if t_remaining is not None else dt_window
    if clamp is not None:
        dt = jnp.minimum(dt, clamp)
    return dt


def assert_dt_admissible(dt, lam_lo_f, lam_hi_f, inradius_f, *, time,
                         iteration):
    """March honesty guard (v6 §1).  ``dt <= 0`` or non-finite is **FATAL**.

    Host-side on purpose: the failure mode this exists to kill is an amrex run
    that spun 3800+ zero-progress steps and exited 0.  Aborts with a
    diagnostic naming the offending face.  Never a silent spin, never a
    dt-halving retry, never a CFL reduction.
    """
    dt_h = float(dt)
    if np.isfinite(dt_h) and dt_h > 0.0:
        return dt_h
    lo = np.asarray(lam_lo_f)
    hi = np.asarray(lam_hi_f)
    r = np.asarray(inradius_f)
    amax = np.maximum(np.abs(lo), np.abs(hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        local = 2.0 * r / amax
    finite = np.isfinite(local)
    bad = (int(np.argmin(np.where(finite, local, np.inf))) if finite.any()
           else int(np.argmax(np.nan_to_num(amax, nan=np.inf))))
    raise FloatingPointError(
        f"reduce_dt returned dt = {dt_h!r} at t = {float(time)} "
        f"(iteration {int(iteration)}) — FATAL. Worst face f = {bad}: "
        f"lam_lo = {float(lo[bad])!r}, lam_hi = {float(hi[bad])!r}, "
        f"inradius = {float(r[bad])!r}. The march aborts: a non-positive or "
        f"non-finite dt is a reported FINDING, never a silent spin.")


# ══ PER-STAGE BLOCKS ════════════════════════════════════════════════════════

def halo_bc(Q, Qaux, p, MeshRT, t, Ops):
    """Per-stage ghost synthesis (v5 amendment 3).

    On an unstructured single-device jax mesh there are no ghost CELLS: the
    boundary state is synthesised as per-boundary-face values from the indexed
    BC kernel.  Returns ``(Q, Qaux, bf_values)`` with ``bf_values`` of shape
    ``(n_state, n_boundary_faces)`` — consumed by :func:`reconstruct` for the
    limiter bounds and the boundary ``Q_R``.  An SPMD backend puts its
    ``ppermute`` halo exchange here; amrex puts its time-interpolated
    FillPatch here (``t`` is carried for exactly that reason).

    vmapped over the boundary-face axis: the generated BC kernel is a sympy
    ``Piecewise`` over ``i_bc_func`` and vectorises cleanly.  A python loop
    unrolls one copy of the whole kernel per boundary face (quadratic compile).
    """
    n_bf = MeshRT.n_boundary_faces
    if Ops.bc_face is None or n_bf == 0:
        return Q, Qaux, jnp.zeros((Q.shape[0], max(n_bf, 1)), dtype=Q.dtype)

    def _per_bf(i):
        cell = MeshRT.boundary_face_cells[i]
        fidx = MeshRT.boundary_faces[i]
        return Ops.bc_face(
            MeshRT.bf_branch[i], t,
            MeshRT.face_centers[fidx, :], MeshRT.bf_distance[i],
            Q[:, cell], Qaux[:, cell], p, MeshRT.face_normals[:, fidx])

    bf_values = jax.vmap(_per_bf)(jnp.arange(n_bf)).T
    return Q, Qaux, bf_values


def reconstruct(Q, Qaux, MeshRT, Ops, bf_values, *, o1=False):
    """Face-state reconstruction (design §2, amendment 10).

    Returns ``(Q_L, Q_R, lim_grad)`` with ``Q_L/Q_R`` of shape
    ``(n_state, n_faces)`` and ``lim_grad`` the LIMITED cell gradient
    ``(n_state, dim, n_cells)`` — or ``None`` at order 1 / on the MOOD
    fallback, where the slope is identically zero.

    ``o1=True`` forces the piecewise-constant object: that is the whole-step
    order-1 redo of :func:`mood_resolve`, and the reason the order-1 path
    needs no separate code.  The reconstruction OBJECT owns the limiter and
    any well-balanced / positivity variable change — none of that is decided
    here.
    """
    if o1 or Ops.order < 2:
        Q_L, Q_R = Ops.reconstruct_o1(Q, bf_values)
        return Q_L, Q_R, None
    if Ops.use_interior_ncp:
        Q_L, Q_R, lim_grad = Ops.reconstruct.reconstruct_with_grad(
            Q, bf_values)
        return Q_L, Q_R, lim_grad
    Q_L, Q_R = Ops.reconstruct(Q, bf_values)
    return Q_L, Q_R, None


def flux_pass(Q, Qaux, p, MeshRT, Ops, t_stage, faces):
    """Per-face numerical flux + fluctuations — **STORED** (design §1).

    ``faces = (Q_L, Q_R)`` from :func:`reconstruct`.  Returns three arrays of
    shape ``(n_state, n_faces)``::

        Fface  conservative numerical flux
        Dp     the +side fluctuation  (path-conservative NCP jump)
        Dm     the -side fluctuation

    Each face is evaluated exactly ONCE; the arrays live until
    :func:`gather_update` consumes them (a serial backend MAY fuse the two —
    the contract is that they are available).

    ``bc_face`` fires HERE, at ``t_stage`` (amendment 11): the boundary ``Q_R``
    is recomputed from the RECONSTRUCTED inner face state so limiter and BC
    stay consistent at the face centre, and stage 2 evaluates its BCs at
    ``t + dt``.  At a periodic seam ``Q_R`` is the partner cell's state
    instead (REQ-116) — the pass-through BC kernel fed ``Q_L`` would silently
    degrade the wrap to extrapolation.
    """
    Q_L, Q_R = faces
    n_state, n_faces = Q_L.shape
    normals = MeshRT.face_normals
    n_bf = MeshRT.n_boundary_faces

    # ── boundary faces: BC at the reconstructed face state, at t_stage ──
    if Ops.bc_face is not None and n_bf > 0:
        bfj = MeshRT.boundary_faces
        Q_L_bnd = Q_L[:, bfj]

        def _per_bf_R(i):
            cell = MeshRT.boundary_face_cells[i]
            fidx = bfj[i]
            return Ops.bc_face(
                MeshRT.bf_branch[i], t_stage,
                MeshRT.face_centers[fidx, :], MeshRT.bf_distance[i],
                Q_L_bnd[:, i], Qaux[:, cell], p, normals[:, fidx])

        Q_R_bnd = jax.vmap(_per_bf_R)(jnp.arange(n_bf)).T
        if MeshRT.has_periodic:
            Q_partner = Q[:, MeshRT.boundary_face_cells]
            Q_R_bnd = jnp.where(MeshRT.periodic_mask[None, :],
                                Q_partner, Q_R_bnd)
        Q_R = Q_R.at[:, bfj].set(Q_R_bnd)

    # ── ONE evaluation per face, over ALL faces ──────────────────────────
    iA, iB = MeshRT.face_owner, MeshRT.face_neigh
    qauxA = Qaux[:, iA]
    qauxB = Qaux[:, iB]
    Fface = Ops.flux_face(Q_L, Q_R, qauxA, qauxB, p, normals)
    fluct = Ops.fluct_face(Q_L, Q_R, qauxA, qauxB, p, normals)
    Dp, Dm = fluct[0], fluct[1]
    return Fface, Dp, Dm


def gather_update(Q0, Fface, Dp, Dm, MeshRT, dt, a_stage, source=None,
                  cell_term=None, h_index=None):
    """Consume the stored face arrays into a candidate cell state, flagging
    troubled cells **while writing** (design §1, the fused detection).

    ``Q0`` is the STAGE BASE (the state the residual was evaluated at, kept
    for the RK average and for the MOOD rollback); ``a_stage`` is the
    Shu-Osher ``beta`` coefficient::

        Q_cand = Q0 + a_stage * dt * ( -div(F + D) + cell_term + source )

    ``cell_term`` is the optional per-cell contribution of amendment 10 — the
    cell-interior non-conservative integral, WB-critical at order >= 2.
    ``troubled`` is ``(n_cells,)`` bool: ``h < 0`` (when the model declares an
    ``h`` row) or any non-finite component.  Detection costs one comparison in
    a pass that exists anyway.

    Owner and neighbour contributions are two scatters, boundary faces
    contributing to their owner only.  A GPU backend flips this to a gather
    over a face-of-cell table — the stored face arrays are what make that
    possible; the arithmetic is identical.
    """
    dQ = jnp.zeros_like(Q0)
    fv = MeshRT.face_volumes
    cv = MeshRT.cell_volumes

    iA_int, iB_int = MeshRT.iA_int, MeshRT.iB_int
    fint = MeshRT.interior_faces
    dQ = dQ.at[:, iA_int].subtract(
        (Fface[:, fint] + Dm[:, fint]) * fv[fint] / cv[iA_int])
    dQ = dQ.at[:, iB_int].subtract(
        (-Fface[:, fint] + Dp[:, fint]) * fv[fint] / cv[iB_int])

    if MeshRT.n_boundary_faces > 0:
        fbnd = MeshRT.boundary_faces
        ib = MeshRT.iInner_bnd
        dQ = dQ.at[:, ib].subtract(
            (Fface[:, fbnd] + Dm[:, fbnd]) * fv[fbnd] / cv[ib])

    if cell_term is not None:
        dQ = dQ + cell_term
    if source is not None:
        dQ = dQ.at[:, :source.shape[1]].add(source)

    Q_cand = Q0 + a_stage * dt * dQ

    return Q_cand, flag_troubled(Q_cand, h_index)


def flag_troubled(Q_cand, h_index):
    """PAD + CAD troubled predicate (core inventory: "troubled predicate").

    ``h < 0`` (physical admissibility) OR any non-finite component (computed
    admissibility).  No relaxed-DMP: it is invalid in the presence of a
    source.  Nothing is floored or clipped — a negative ``h`` is *detected*,
    never repaired in place (user law).
    """
    bad = ~jnp.isfinite(Q_cand).all(axis=0)
    if h_index is not None:
        bad = bad | (Q_cand[h_index] < 0.0)
    return bad


def mood_resolve(troubled, Q0, Q_cand, Qaux, p, t, dt, MeshRT, Ops,
                 stage_loop):
    """A-posteriori MOOD with **OUTCOME semantics** (amendment 5).

    Contract: the accepted state is positive and finite and exactly mass
    conservative.  The STRATEGY is the backend's choice; the sanctioned body
    used here is the **whole-step order-1 redo** — replay the entire stage
    loop with the piecewise-constant reconstruction.  It is conservative by
    construction (the same shared face flux) and positivity-preserving by the
    first-order XZS lemma; the per-cell variant was measured to leak 14 % of
    the mass in amrex, which is why the whole-step redo is the sanctioned one.

    ``stage_loop(Q0, Qaux, p, t, dt, o1)`` is the step's own stage loop,
    handed in so MOOD owns no scheme logic of its own.
    """
    return jax.lax.cond(
        jnp.any(troubled),
        lambda: stage_loop(Q0, Qaux, p, t, dt, True)[0],
        lambda: Q_cand,
    )


# ══ POST-STEP BLOCKS ════════════════════════════════════════════════════════

def update_variables(Q, Qaux, p, time, dt, MeshRT, Ops):
    """The model's per-cell ``update_variables`` slot (identity when the model
    declares none — the derived SWE does, which is exactly why it carries no
    wet/dry momentum cap)."""
    if Ops.update_variables is None:
        return Q
    return Ops.update_variables(Q, Qaux, p)


def update_aux(Q, Qaux, p, time, dt, MeshRT, Ops):
    """The aux leg: the model's algebraic ``update_aux_variables`` (e.g. the
    KP-desingularised ``hinv``) followed by the LSQ-gradient rows of
    ``aux_registry`` (amendment 9 gives this block ``dt``).

    Both legs are the EXISTING kernels; the derivative walk is
    ``HyperbolicSolver._walk_derivative_aux`` reused verbatim.  The
    full-length contract on the algebraic vector is enforced here for the
    same reason it is enforced in ``solver_jax``: a prefix write silently
    mis-places algebraic rows and degenerates SME(>=1) to SWE.
    """
    out = Qaux
    if Ops.update_aux_variables is not None:
        local = Ops.update_aux_variables(
            Q, Qaux, p, time=time,
            position=MeshRT.cell_centers[:, :Q.shape[-1]])
        if local.shape[0] != out.shape[0]:
            raise ValueError(
                f"update_aux_variables must be full-length ({out.shape[0]} aux "
                f"rows) so each algebraic aux lands on its own row — got "
                f"{local.shape[0]}.")
        out = local
    if Ops.aux_registry_walk is not None:
        out = Ops.aux_registry_walk(Ops.sm, out, Q, MeshRT.mesh)
    return out
