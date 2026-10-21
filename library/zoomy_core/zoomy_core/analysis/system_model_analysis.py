"""SystemModel-based analysis: linearise + dispersion + eigenvalues.

Operates directly on `zoomy_core.systemmodel.system_model.SystemModel`
— no `PDESystem` wrapper, no `linearise`-on-equation-tree machinery.
The model's stored sympy matrices already encode the operator surface;
analysis is straightforward symbolic substitution + eigenvalue solve.

Three entry points:

* `linearise_system_model` — substitute a base state into the
  flux / NCP / source / hydrostatic-pressure / mass matrices and
  return a new `SystemModel` whose matrices are constant in the
  state Symbols (only depend on coordinates and parameters).

* `plane_wave_dispersion` — given a constant-coefficient
  linearised SystemModel, build the principal-symbol pencil
  `(M_t, M_x[axis], M_0)` and return the symbolic eigenvalues
  `ω(k)` (or equivalently `λ = ω/k` solving
  `det(M_x − λ M_t) = 0`).

* `hyperbolic_eigenvalues` — same pencil but returning the
  generalized eigenvalues of `(M_x[axis], M_t)` directly via
  `zoomy_core.analysis.pencil.generalised_eigenvalues`.

For shapes where the analyst wants `Sum`/`Indexed` form (linear
stability with symbolic `k`, parameter-sweep hyperbolicity, etc.)
the substitution dict accepts any sympy values — `base_state[h] = h0`
keeps `h0` symbolic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import sympy as sp

from zoomy_core.systemmodel.system_model import SystemModel


def linearise_system_model(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    parameters: Optional[Dict[Any, Any]] = None,
) -> SystemModel:
    """Substitute `base_state` into every operator matrix; return a
    new `SystemModel` whose matrices are constant in the state.

    Parameters
    ----------
    sm : SystemModel
        The source operator-form system.
    base_state : dict
        Map of state Symbol → base value.  Missing entries are left
        symbolic.  Aux-state Symbols may also be substituted.
    parameters : dict, optional
        Additional parameter substitutions (e.g. `{ez: 1, g: 9.81}`).
        Combined with `base_state` for the substitution.
    """
    sub = dict(base_state)
    if parameters:
        sub.update(parameters)

    def _sub(M):
        if isinstance(M, sp.Matrix):
            return sp.Matrix(
                M.shape[0], M.shape[1],
                lambda i, j: sp.simplify(M[i, j].xreplace(sub)),
            )
        # NDimArray (NCP)
        out = sp.MutableDenseNDimArray.zeros(*M.shape)
        for idx in _iter_indices(M.shape):
            out[idx] = sp.simplify(M[idx].xreplace(sub))
        return out

    return SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=_sub(sm.flux),
        hydrostatic_pressure=_sub(sm.hydrostatic_pressure),
        nonconservative_matrix=_sub(sm.nonconservative_matrix),
        source=_sub(sm.source),
        mass_matrix=_sub(sm.mass_matrix),
        boundary_conditions=sm.boundary_conditions,
        history=list(sm.history) + [
            {"name": "linearise", "description": f"base state {dict(base_state)}"}
        ],
    )


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


def plane_wave_dispersion(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    axis: int = 0,
    parameters: Optional[Dict[Any, Any]] = None,
    return_omega_k: bool = True,
) -> Dict[str, Any]:
    """Compute the dispersion of `sm` linearised at `base_state`.

    Two output modes are supported (controlled by `return_omega_k`):

    * `return_omega_k=True` (default, **fixes G7** from the plan):
      assemble the full plane-wave matrix `M(ω, k)` from the
      linearised system and solve `det M(ω, k) = 0` for the true
      `ω(k)` curves — symbolic functions of `k` rather than just
      eigenvalues at a fixed base state.
    * `return_omega_k=False` (legacy): return only generalised
      eigenvalues of `(M_x, M_t)` — phase velocities at the base
      state, no `k`-dependence.

    Returns a dict with:

    * `M_t` — mass matrix at base state.
    * `M_x` — quasilinear matrix at base state in direction `axis`.
    * `M_0` — `-∂S/∂Q` at base state.
    * `eigenvalues` — generalised eigenvalues (legacy, always present).
    * `omega_solutions` (only when `return_omega_k=True`) — list of
      symbolic `ω(k)` solutions of `det M(ω, k) = 0`.
    * `phase_velocity_solutions` (only when `return_omega_k=True`)
      — `[ω/k for ω in omega_solutions]`.
    * `k` (only when `return_omega_k=True`) — the wavenumber symbol.
    * `omega` (only when `return_omega_k=True`) — the angular‑frequency
      symbol.
    """
    sub = dict(base_state)
    if parameters:
        sub.update(parameters)

    # Quasilinear / source-Jacobian on the FULL state-symbolic matrices,
    # then substitute the base state into the result.
    n_eq = sm.n_equations
    n_st = sm.n_state
    qm_sym = sm.quasilinear_matrix
    M_x = sp.Matrix(
        n_eq,
        n_st,
        lambda i, j: sp.simplify(qm_sym[i, j, axis].xreplace(sub)),
    )

    src_jac_sym = sm.source_jacobian_wrt_variables
    M_0 = sp.Matrix(
        n_eq,
        n_st,
        lambda i, j: -sp.simplify(src_jac_sym[i, j].xreplace(sub)),
    )

    M_t = sp.Matrix(
        n_eq,
        n_st,
        lambda i, j: sp.simplify(sm.mass_matrix[i, j].xreplace(sub)),
    )

    # Legacy: generalised eigenvalues of (M_x, M_t) — wave speeds.
    lam = sp.Symbol("lambda")
    char_poly = (M_x - lam * M_t).det(method="berkowitz")
    eigenvalues = sp.solve(sp.Eq(sp.simplify(char_poly), 0), lam)

    out = {
        "M_t": M_t,
        "M_x": M_x,
        "M_0": M_0,
        "eigenvalues": eigenvalues,
    }

    if return_omega_k:
        # Full ω(k) dispersion: plane-wave ansatz q = q̂ exp(i(k·x − ωt)).
        # The linearised system M_t ∂_t q + M_x ∂_x q + M_0 q = 0
        # under the plane-wave substitution becomes
        #     M(ω, k) q̂ = (-iω M_t + ik M_x + M_0) q̂ = 0,
        # and dispersion solutions are the roots of det M(ω, k) = 0.
        omega = sp.Symbol("omega", real=True)
        k = sp.Symbol("k", real=True)
        I = sp.I
        M_disp = -I * omega * M_t + I * k * M_x + M_0
        det_disp = sp.simplify(M_disp.det(method="berkowitz"))
        try:
            omega_solutions = sp.solve(sp.Eq(det_disp, 0), omega)
        except Exception:
            omega_solutions = []
        out["omega"] = omega
        out["k"] = k
        out["dispersion_matrix"] = M_disp
        out["dispersion_determinant"] = det_disp
        out["omega_solutions"] = omega_solutions
        try:
            out["phase_velocity_solutions"] = [
                sp.simplify(s / k) for s in omega_solutions
            ]
        except Exception:
            out["phase_velocity_solutions"] = []

    return out


def hyperbolic_eigenvalues(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    axis: int = 0,
    parameters: Optional[Dict[Any, Any]] = None,
) -> List[sp.Expr]:
    """Convenience: return only the eigenvalues from
    `plane_wave_dispersion`.

    Useful for hyperbolicity sampling: real eigenvalues = hyperbolic;
    complex conjugate pairs = elliptic mode = ill-posed initial-value
    problem.
    """
    return plane_wave_dispersion(
        sm, base_state, axis=axis, parameters=parameters
    )["eigenvalues"]


__all__ = [
    "linearise_system_model",
    "plane_wave_dispersion",
    "hyperbolic_eigenvalues",
]
