"""Time-stepping strategies for explicit FVM solvers.

All timestep functions return a `compute_dt` closure that is
compatible with both NumPy and JAX (JIT-safe: no Python `float()`
or `min()` on traced values).
"""

import numpy as np


def _safe_minimum(a, b):
    """Element-wise minimum for both NumPy arrays and JAX tracers.

    Dispatches on operand TYPE: when both operands are plain NumPy/Python
    numbers the reduction stays in NumPy (float64), and only a JAX operand
    (a tracer under `jax.jit` or a device array) routes through
    `jnp.minimum`.  The old unconditional `jnp.minimum` silently promoted a
    NumPy `float64` `dt` to a default-`float32` JAX array whenever jax was
    merely importable — harmless while the cap was rarely applied, but REQ-190
    now caps EVERY step, and a wet-case `dt` must stay bit-identical."""
    plain = (np.ndarray, np.generic, int, float)
    if isinstance(a, plain) and isinstance(b, plain):
        return np.minimum(a, b)
    import jax.numpy as jnp
    return jnp.minimum(a, b)


def constant(dt=0.1):
    """Fixed timestep."""
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        return dt
    return compute_dt


def apply_default_dt_max(compute_dt, dt_max):
    """Fill a timestep strategy's `dt_max` from the NSM default (REQ-190).

    Solvers call this once at setup with `nsm.dt_max`: a strategy that
    supports an NSM-default cap (`adaptive`) fills its cap **iff the caller
    did not pass one explicitly** (an explicit `dt_max=` on `adaptive(...)`
    always wins).  Strategies without the hook (`constant` dt, a user's own
    callable) are left untouched — capability detection, not a mismatch shim.
    """
    setter = getattr(compute_dt, "set_default_dt_max", None)
    if setter is not None and dt_max is not None:
        setter(dt_max)


def adaptive(CFL=0.9, nu=0.0, dimension=2, degree=0, dt_max=None):
    """Adaptive CFL-based timestep.

    Uses the classical hyperbolic / parabolic limits with the spatial
    dimension and DG polynomial degree factored *into the denominator*
    so the `CFL` knob is a single safety factor ∈ (0, 1]:

    .. math::

        \\Delta t_\\text{conv} &\\le \\mathrm{CFL} \\;
            \\frac{2\\,r_\\text{in}}{d \\, (2k+1) \\, |\\lambda|_\\text{max}}, \\\\
        \\Delta t_\\text{diff} &\\le \\mathrm{CFL} \\;
            \\frac{(2\\,r_\\text{in})^2}{2 \\, d \\, \\nu}.

    Here `r_in` is the cell inradius (so `2·r_in` is the conservative
    "diameter"), `d` the spatial dimension, `k` the DG polynomial
    degree (use `0` for FV).  The hyperbolic CFL constant `1/(2k+1)`
    is the SSP-RK(k+1,k+1) stability limit (Cockburn & Shu, 1991);
    the spatial dimension contributes the `1/d` factor in 2D/3D.

    Parameters
    ----------
    CFL : float in (0, 1]
        Safety factor on top of the theoretical limit (default 0.9).
    dt_max : float, optional
        Hard upper cap on dt — for explicit SOURCE stiffness (e.g. the
        regularized-Bingham plug viscosity or a bed-slip penalty) that
        the hyperbolic CFL cannot see, AND the timestep a WAVE-FREE domain
        takes: when every gated eigenvalue is 0 (fully dry / sub-`wet_dry_eps`)
        the CFL is `+inf` everywhere, so `dt` collapses to this cap rather
        than a hardcoded floor (REQ-190).  Backend-neutral (JIT-safe): prefer
        this over wrapping `compute_dt` in a Python `min(float(...))` lambda,
        which fails under `jax.jit`.  When `None` at construction the value
        is FILLED from the NSM's `dt_max` by the solver at setup
        (`apply_default_dt_max`); passing a number here is EXPLICIT and
        overrides the NSM default.
    nu : float
        Scalar viscosity for the parabolic CFL (default 0 → skip).
    dimension : int
        Spatial dimension of the mesh (default 2).
    degree : int
        DG polynomial degree (default 0 for FV).
    """
    deg_fac = float(2 * degree + 1)
    dim_fac = float(dimension)
    # Mutable cap.  An EXPLICIT ``dt_max`` is frozen here (``explicit=True``);
    # when ``None`` the solver may later fill it from the NSM's ``dt_max`` via
    # :func:`apply_default_dt_max` (explicit always wins).
    cap = {"dt_max": dt_max, "explicit": dt_max is not None}

    def compute_dt(Q, Qaux, parameters, inradius, compute_max_abs_eigenvalue):
        # LOCAL CFL: ``inradius`` is the per-face (or per-cell) inradius array
        # paired elementwise with the per-face local wave speed — each face is
        # limited by ITS OWN size and ITS OWN |λ|, and the global dt is the
        # minimum of those local limits.  A global-scalar radius (the old
        # behavior) silently paired the smallest cell anywhere with the
        # fastest wave anywhere — strictly over-restrictive on non-uniform
        # meshes.  A scalar still works (uniform mesh) and gives the same dt.
        ev_abs_max = compute_max_abs_eigenvalue(Q, Qaux, parameters)
        h = 2.0 * inradius                                    # conservative size
        # WAVE-FREE: a dry / sub-``wet_dry_eps`` face has ``ev_abs_max == 0`` so
        # its local limit is ``h/0 == +inf`` — it imposes NO CFL constraint and
        # drops out of the ``min`` below.  The divide-by-zero is intentional and
        # yields exactly ``+inf`` (not a floor); silence only the FP warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            dt_local = CFL * h / (dim_fac * deg_fac * ev_abs_max)
        if nu > 0:
            dt_local = _safe_minimum(dt_local, CFL * h ** 2 / (2.0 * dim_fac * nu))
        dt = dt_local.min()
        # If EVERY face is wave-free, ``dt`` is ``+inf`` here; the cap makes the
        # step exactly ``dt_max`` (REQ-190) instead of leaking ``inf`` downstream.
        dtm = cap["dt_max"]
        if dtm is not None:
            dt = _safe_minimum(dt, dtm)
        return dt

    def set_default_dt_max(value):
        """Fill `dt_max` from a solver-supplied default (the NSM's `dt_max`)
        IFF the caller did not pass one explicitly.  Explicit argument wins."""
        if not cap["explicit"] and value is not None:
            cap["dt_max"] = value

    compute_dt.set_default_dt_max = set_default_dt_max
    compute_dt.dt_max_is_explicit = cap["explicit"]
    return compute_dt
