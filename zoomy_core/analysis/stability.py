"""Linear-stability workflows on a SystemModel.

The intended reading of a case driver is four lines:

```
    P    = NumericPencil(sm)                                   # once per system
    mats = P.at_equilibrium(params, fixed={"b": 0, "h": 1},
                            guess=q0, drop=("b",))             # base state
    T    = temporal_branch(mats, k_grid, c_seed=c0)            # omega(k)
    S    = spatial_branch(mats, omega_grid, c_seed=c0)         # k(omega)
```

plus `growth_cutoff` / `critical_parameter` for derived scalars.

Everything is generic over the SystemModel: state size, parameter set and
layout come from the model; nothing here knows SME/VAM/Bingham.

Why `lambdify` lives HERE and not in a case: analysis needs the symbolic
pencil (from `extract_quasilinear_pencil`) as fast numeric callables
over `state + parameters`.  `sympy.lambdify` *is* sympy's numpy code
printer; this module is the single place it is applied, so cases never
touch it.  (Solver kernels go through the per-backend printers instead —
different target, same idea.)
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.optimize import brentq, fsolve

from zoomy_core.misc.misc import Zstruct
from .linearisation import linearise
from .pencil import extract_quasilinear_pencil

__all__ = [
    "NumericPencil",
    "temporal_branch",
    "spatial_branch",
    "spatial_dispersion",
    "spatial_cutoff",
    "growth_cutoff",
    "critical_parameter",
    "viscous_operator",
]


def _aux_list(sm):
    """The SystemModel's aux symbols (gradient fields etc.), [] when absent."""
    aux = getattr(sm, "aux_variables", None)
    if aux is None:
        return []
    try:
        return list(aux.get_list())
    except AttributeError:
        return list(aux)


class NumericPencil:
    """Numeric evaluator of a SystemModel's quasilinear pencil.

    Lambdifies `(M_t, A, J_S, source)` over `state + parameters` once at
    construction; `at_equilibrium` then produces the base-state
    matrices for the branch solvers.  Construct it once per system and reuse
    it across parameter sweeps (e.g. inside `critical_parameter`).
    """

    def __init__(self, sm):
        M_t, M_xa, M_0 = extract_quasilinear_pencil(sm)
        aux = _aux_list(sm)
        args = list(sm.state) + aux + list(sm.parameters)
        self.sm = sm
        self.state_names = [str(s) for s in sm.state]
        self._n_aux = len(aux)
        self._A = sp.lambdify(args, M_xa[0], "numpy")       # d(F+P)/dQ + B
        self._JS = sp.lambdify(args, -M_0, "numpy")         # dS/dQ
        self._Mt = sp.lambdify(args, sp.Matrix(M_t), "numpy")
        self._S = sp.lambdify(args, list(sp.Matrix(sm.source)), "numpy")

    def _pvals(self, params):
        # aux variables (gradient fields etc.) vanish at the uniform base
        # state — passed as zeros between state and parameters.
        return [0.0] * self._n_aux + [float(params[str(p)])
                                      for p in self.sm.parameters]

    def equilibrium(self, params, fixed, guess=None):
        """Uniform base state: source rows of the FREE states vanish.

        `fixed` pins states by name (e.g. `{"b": 0.0, "h": 1.0}`); the
        remaining (free) states are solved with `fsolve` from `guess`
        (zeros by default — pass a guess only if the solve demonstrably
        needs one).  Returns `(Q_full, max_residual)`.
        """
        pv = self._pvals(params)
        free = [i for i, n in enumerate(self.state_names) if n not in fixed]
        if guess is None:
            guess = np.zeros(len(free))

        def q_full(qf):
            Q = [fixed[n] for n in self.state_names if n in fixed]  # ordered below
            out = np.empty(len(self.state_names))
            for i, n in enumerate(self.state_names):
                out[i] = fixed[n] if n in fixed else np.nan
            out[free] = qf
            return out

        def res(qf):
            Q = q_full(qf)
            return np.asarray(self._S(*(list(Q) + pv)), float).ravel()[free]

        if not free:                       # fully pinned base state (e.g. SWE)
            return q_full(np.empty(0)), 0.0
        qf = fsolve(res, np.asarray(guess, float))
        return q_full(qf), float(np.abs(res(qf)).max())

    def at_equilibrium(self, params, fixed, guess=None, drop=()):
        """Base-state matrices for the branch solvers.

        Solves the equilibrium, evaluates `A, J_S, M_t` there and drops the
        rows/columns of the states named in `drop` (static fields, e.g. the
        bed).  Returns `Zstruct(A, JS, Mt, Q, Qfree, residual)`.
        """
        Q, rmax = self.equilibrium(params, fixed, guess)
        pv = self._pvals(params)
        args = list(Q) + pv
        keep = [i for i, n in enumerate(self.state_names) if n not in drop]
        ix = np.ix_(keep, keep)
        free = [i for i, n in enumerate(self.state_names) if n not in fixed]
        return Zstruct(
            A=np.asarray(self._A(*args), float)[ix],
            JS=np.asarray(self._JS(*args), float)[ix],
            Mt=np.asarray(self._Mt(*args), float)[ix],
            Q=Q, Qfree=Q[free], residual=rmax)


def _track(eigs, seed_row, seed_idx):
    """Follow one mode through an eigenvalue table by nearest-continuity.

    `eigs`: (n_grid, n_modes) complex table; the branch is seeded at
    `eigs[seed_row, seed_idx]` and continued forward and backward.
    """
    tr = np.zeros(len(eigs), complex)
    tr[seed_row] = eigs[seed_row, seed_idx]
    for i in range(seed_row + 1, len(eigs)):
        tr[i] = eigs[i, np.argmin(np.abs(eigs[i] - tr[i - 1]))]
    for i in range(seed_row - 1, -1, -1):
        tr[i] = eigs[i, np.argmin(np.abs(eigs[i] - tr[i + 1]))]
    return tr


def temporal_branch(mats, k, viscous=None, c_seed=None, anchor=3):
    """Temporal branch omega(k): real wavenumbers, complex frequencies.

    Solves `omega = -i eig(i k A - J_S + k^2 V, M_t)` over the grid `k`
    and tracks the mode whose phase speed at `k[anchor]` is nearest
    `c_seed` (defaults to the fastest-growing mode there).  Returns
    `Zstruct(k, omega, sigma, c, cg, alpha)` with growth rate `sigma`,
    phase speed `c`, group speed `cg` and Gaster spatial growth
    `alpha = sigma / cg`.
    """
    k = np.asarray(k, float)
    om = np.zeros((len(k), mats.A.shape[0]), complex)
    for i, kk in enumerate(k):
        M = 1j * kk * mats.A - mats.JS
        if viscous is not None:
            M = M + kk ** 2 * viscous
        om[i] = -1j * eig(M, mats.Mt, right=False)
    if c_seed is not None:
        seed = int(np.argmin(np.abs(om[anchor].real / k[anchor] - c_seed)))
    else:
        seed = int(np.argmax(om[anchor].imag))
    tr = _track(om, anchor, seed)
    wr, sig = tr.real, tr.imag
    cg = np.gradient(wr, k)
    return Zstruct(k=k, omega=wr, sigma=sig, c=wr / k, cg=cg, alpha=sig / cg)


def _spatial_eig_table(mats, omega):
    """Complex-k eigenvalue table (n_omega × n_mode) of the spatial pencil.

    Since k enters the pencil linearly the spatial problem is again a
    generalized eigenproblem with k as the eigenvalue:
    `k = -i eig(i omega M_t + J_S, A)` — no root chasing.  Accepts
    complex `omega` (the round-trip validation feeds a complex temporal
    frequency back in; physical spatial analysis uses real omega)."""
    omega = np.asarray(omega, complex)
    return np.array([-1j * eig(1j * w * mats.Mt + mats.JS, mats.A, right=False)
                     for w in omega])


def _spatial_select(k_table, omega, c_seed):
    """Seed on the downstream mode (Re k > 0) nearest `c_seed` at
    `omega[0]` and continue it through the table by nearest-continuity.
    Shared by the numeric and analytic spatial branches so both return the
    SAME mode.  Returns the complex tracked branch."""
    k_table = np.asarray(k_table, complex)
    w0 = np.asarray(omega, complex)[0].real
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = np.abs(w0 / k_table[0].real
                      - (c_seed if c_seed is not None else 0.0))
    seed = int(np.argmin(cost + 1e3 * (k_table[0].real <= 0)))
    return _track(k_table, 0, seed)


def spatial_branch(mats, omega, c_seed=None):
    """Spatial branch k(omega) from pre-evaluated base-state matrices.

    Lower-level numeric primitive; prefer `spatial_dispersion` for the
    SystemModel-in entry point.  Returns `Zstruct(omega, k, alpha)` with
    spatial growth `alpha = -Im k`.
    """
    omega = np.asarray(omega, float)
    tr = _spatial_select(_spatial_eig_table(mats, omega), omega, c_seed)
    return Zstruct(omega=omega, k=tr.real, alpha=-tr.imag)


# ---------------------------------------------------------------------------
# Unified spatial-dispersion entry point: SystemModel in, results out.
# ONE entry point, ANALYTIC by default, ``numeric=True`` for the eigen path
# (the only feasible route for SME(N>=2), where the symbolic determinant
# blows up).  Both paths solve the SAME quasilinear pencil
# ``det(-i omega M_t + i k A + M_0) = 0`` for the complex wavenumber k.
# ---------------------------------------------------------------------------

def _norm_base(sm, base_state):
    name2sym = {str(s): s for s in sm.state}
    by_sym, by_name = {}, {}
    for key, val in base_state.items():
        s = name2sym[key] if isinstance(key, str) else key
        by_sym[s], by_name[str(s)] = val, val
    return by_sym, by_name


def _analytic_k_table(sm, base_state, omega, *, params, axis, drop):
    """Symbolic k(omega): build the quasilinear pencil from
    `extract_quasilinear_pencil`, solve `det = 0` for k, substitute
    the parameters and evaluate over the omega grid.  Returns
    `(k_table, k_symbolic)`."""
    by_sym, _ = _norm_base(sm, base_state)
    missing = set(sm.state) - set(by_sym)
    if missing:
        raise ValueError(
            f"analytic spatial_dispersion needs every state pinned in "
            f"base_state; missing {sorted(map(str, missing))}. Pin them, or "
            f"pass numeric=True (free moments are solved only numerically).")
    sm_lin = linearise(sm, by_sym)
    M_t, M_xa, M_0 = extract_quasilinear_pencil(sm_lin)
    keep = [i for i, s in enumerate(sm.state) if str(s) not in drop]
    ix = keep
    Mt = sp.Matrix(M_t)[ix, ix]
    Mx = sp.Matrix(M_xa[axis])[ix, ix]
    M0 = sp.Matrix(M_0)[ix, ix]
    om_s, k_s = sp.Symbol("omega"), sp.Symbol("k")
    pencil = -sp.I * om_s * Mt + sp.I * k_s * Mx + M0
    det = pencil.det(method="berkowitz")
    k_sym = sp.solve(sp.Eq(det, 0), k_s)
    if not k_sym:
        raise ValueError("symbolic spatial solve returned no k(omega) roots; "
                         "use numeric=True.")
    # Match params/base values by symbol NAME (assumptions on the pencil's
    # symbols, e.g. positive=True, differ from a bare Symbol(name)).
    pmap = {str(p): v for p, v in (params or {}).items()}
    k_num = [s.subs({sym: pmap[str(sym)] for sym in s.free_symbols
                     if str(sym) in pmap}) for s in k_sym]
    leftover = set().union(*(s.free_symbols for s in k_num)) - {om_s}
    if leftover:
        raise ValueError(
            f"k(omega) still depends on {sorted(map(str, leftover))} after "
            f"substituting params — provide numeric values for them (params "
            f"must cover every parameter and any symbolic base-state value).")
    funcs = [sp.lambdify(om_s, s, "numpy") for s in k_num]
    table = np.array([[complex(f(w)) for f in funcs]
                      for w in np.asarray(omega, complex)])
    return table, k_sym


def spatial_dispersion(sm, base_state, omega, *, params=None, numeric=False,
                       axis=0, drop=(), c_seed=None, guess=None):
    """Spatial dispersion branch k(omega) of a SystemModel — the real-omega,
    complex-k companion of `temporal_branch`.

    One entry point; the `numeric` switch selects how the SAME quasilinear
    pencil `det(-i omega M_t + i k A + M_0) = 0` is solved for k:

    * `numeric=False` (DEFAULT, analytic): the plane-wave ansatz stays
      symbolic — solve the determinant for k(omega) in closed form, then
      evaluate over the grid.  Feasible for small systems (SWE, SME(1)).
    * `numeric=True`: evaluate the pencil at the base state and take k as
      the generalized eigenvalue.  The only feasible route once the symbolic
      determinant blows up (SME(N>=2)).

    Parameters
    ----------
    sm : SystemModel                 the (non-linearised) system.
    base_state : dict                state (symbol or name) -> value.  The
        analytic path needs every state pinned; the numeric path pins these
        and solves the remaining (free) states at equilibrium.
    omega : array                    real frequencies (complex is accepted
        for the temporal->spatial round-trip validation).
    params : dict                    parameter (name) -> numeric value; for
        the analytic path must also cover any symbolic base-state value.
    numeric : bool                   analytic (False) or eigen (True).
    axis, drop, c_seed, guess        spatial axis; static states to drop
        (e.g. the bed); phase-speed seed for mode selection; equilibrium
        guess (numeric path).

    Returns
    -------
    Zstruct(omega, k, alpha, c, k_symbolic, residual)
        complex `k` (tracked branch), spatial growth `alpha = -Im k`,
        phase speed `c = Re(omega)/Re(k)`; `k_symbolic` is the closed
        form (analytic only), `residual` the equilibrium residual
        (numeric only).
    """
    om_arr = np.asarray(omega, complex)
    k_symbolic, residual = None, None
    if numeric:
        _, by_name = _norm_base(sm, base_state)
        mats = NumericPencil(sm).at_equilibrium(
            params or {}, fixed=by_name, guess=guess, drop=drop)
        table = _spatial_eig_table(mats, om_arr)
        residual = mats.residual
    else:
        table, k_symbolic = _analytic_k_table(
            sm, base_state, om_arr, params=params, axis=axis, drop=drop)
    kc = _spatial_select(table, om_arr, c_seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = om_arr.real / kc.real
    return Zstruct(omega=om_arr, k=kc, alpha=-kc.imag, c=c,
                   k_symbolic=k_symbolic, residual=residual)


def spatial_cutoff(branch):
    """Spatial cutoff frequency: first `omega` where the spatial growth
    `alpha` crosses from + to - (amplification -> decay) on the tracked
    branch of a `spatial_dispersion` result.  `nan` if it never
    restabilizes."""
    a = np.asarray(branch.alpha, float)
    w = np.asarray(branch.omega)
    w = w.real if np.iscomplexobj(w) else w
    zc = np.where((a[:-1] > 0) & (a[1:] <= 0))[0]
    return float(w[zc[0]]) if len(zc) else np.nan


def growth_cutoff(branch):
    """First frequency where the temporal growth `sigma` crosses + -> -.

    Takes a `temporal_branch` result; returns `omega` at the
    crossing, or `nan` if the branch never restabilizes.
    """
    s = branch.sigma
    zc = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    return float(branch.omega[zc[0]]) if len(zc) else np.nan


def critical_parameter(build, lo, hi, k=None, xtol=0.03):
    """Neutral value of a scalar parameter: max-over-k temporal growth = 0.

    `build(x)` maps the parameter value to `(mats, viscous_or_None)`
    (typically a closure over a reusable `NumericPencil`).  Root via
    `brentq` on `max_k max_modes Im omega`; `nan` when the bracket
    does not change sign.
    """
    k = np.linspace(1e-3, 4.0, 400) if k is None else np.asarray(k, float)

    def maxgrow(x):
        mats, V = build(x)
        return max(np.max((-1j * eig(
            1j * kk * mats.A - mats.JS + (kk ** 2 * V if V is not None else 0.0),
            mats.Mt, right=False)).imag) for kk in k)

    try:
        return brentq(maxgrow, lo, hi, xtol=xtol)
    except ValueError:
        return np.nan


def viscous_operator(mats_or_sm, params=None, Q=None):
    """Second-order (diffusive) pencil contribution `V` from a SystemModel
    whose derivation retains the viscous terms (`diffusion_matrix` set):
    plane waves turn `d_x(D d_x Q)` into `-k^2 D`, so `V = -D(Q)`.

    Models that drop the streamwise stress in the derivation (pre-REQ-176(4))
    have no `diffusion_matrix`; their cases must build `V` from closure
    knowledge and pass it to `temporal_branch` directly.
    """
    sm = mats_or_sm
    D = getattr(sm, "diffusion_matrix", None)
    if D is None:
        raise ValueError(
            "SystemModel has no diffusion_matrix — the derivation dropped the "
            "viscous terms (REQ-176(4)); build V case-side and pass it to "
            "temporal_branch(viscous=...) instead.")
    aux = _aux_list(sm)
    args = list(sm.state) + aux + list(sm.parameters)
    arr = np.asarray(D.tolist() if hasattr(D, "tolist") else D, dtype=object)
    n = arr.shape[0]
    Dxx = arr.reshape(n, arr.shape[1], -1)[:, :, 0]     # first direction pair
    fD = sp.lambdify(args, sp.Matrix([[sp.sympify(e) for e in row]
                                      for row in Dxx]), "numpy")
    pv = [0.0] * len(aux) + [float(params[str(p)]) for p in sm.parameters]
    return -np.asarray(fD(*(list(Q) + pv)), float)
