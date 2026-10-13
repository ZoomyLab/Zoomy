"""Module `zoomy_core.fvm.jvp_numpy`."""

import numpy as np

from zoomy_core.mesh.lsq_reconstruction import compute_derivatives


def fd_jvp(residual_fn, Q, V, eps=1e-7):
    """Fd jvp."""
    return (residual_fn(Q + eps * V) - residual_fn(Q)) / eps


def _dqaux_action_from_specs(symbolic_model, V, mesh, dt):
    """Internal helper `_dqaux_action_from_specs`."""
    n_aux, n_cells = symbolic_model.n_aux_variables, V.shape[1]
    dQaux = np.zeros((n_aux, n_cells), dtype=float)
    if not hasattr(symbolic_model, "derivative_specs"):
        return dQaux
    if not symbolic_model.derivative_specs:
        return dQaux

    var_keys = symbolic_model.variables.keys()
    field_to_index = {name: i for i, name in enumerate(var_keys)}

    for spec in symbolic_model.derivative_specs:
        i_aux = symbolic_model.derivative_key_to_index[spec.key]
        i_q = field_to_index[spec.field]
        v_field = V[i_q]
        n_t = sum(a == "t" for a in spec.axes)
        n_x = sum(a == "x" for a in spec.axes)
        if len(spec.axes) != (n_t + n_x):
            raise NotImplementedError("Only axes in {'t','x'} are supported.")
        if n_t > 1:
            raise NotImplementedError("Only first-order time derivatives are supported.")

        data = v_field
        if n_t == 1:
            data = v_field / max(float(dt), 1e-14)

        if n_x == 0:
            dQaux[i_aux] = data
            continue

        dQaux[i_aux] = compute_derivatives(
            data,
            mesh,
            derivatives_multi_index=[[n_x]],
            u_boundary_face="extrapolation",
        )[:, 0]
    return dQaux


def analytic_source_jvp(
    runtime_model,
    symbolic_model,
    Q,
    Qaux,
    V,
    mesh,
    dt,
    parameters,
    include_chain_rule=True,
):
    """
    Analytic Jv for source residual S(Q, Qaux(Q)).

    - Without chain rule: Jv = (dS/dQ) v
    - With chain rule:    Jv = (dS/dQ) v + (dS/dQaux) (dQaux/dQ v)

    `parameters` is the NUMERIC parameter vector the solver holds
    (`self._sim_parameters`).  REQ-144: it MUST be passed in — since REQ-163
    the model's parameters stay FREE SYMBOLS through derivation/lowering, so
    reconstructing them from `symbolic_model.parameters.values()` yields
    Symbols and `float(Symbol)` raised ``TypeError: Cannot convert expression
    to float``, crashing the whole `source_mode='global'` implicit path.
    """
    parameters = np.asarray(parameters, dtype=float)
    Jq = runtime_model.source_jacobian_wrt_variables(Q, Qaux, parameters)
    # expected shape: (n_var, n_var, n_cells)
    jv = np.einsum("ijc,jc->ic", Jq, V)

    if not include_chain_rule or symbolic_model.n_aux_variables == 0:
        return jv

    Ja = runtime_model.source_jacobian_wrt_aux_variables(Q, Qaux, parameters)
    # current SymPy derive_by_array ordering yields (n_aux, n_var, n_cells)
    # fallback for possible (n_var, n_aux, n_cells)
    if Ja.shape[0] == symbolic_model.n_aux_variables:
        Ja_aux_var = Ja
    elif Ja.shape[1] == symbolic_model.n_aux_variables:
        Ja_aux_var = np.transpose(Ja, (1, 0, 2))
    else:
        raise ValueError(f"Unexpected source_jacobian_wrt_aux_variables shape: {Ja.shape}")

    dQaux = _dqaux_action_from_specs(symbolic_model, V, mesh, dt)
    jv += np.einsum("aic,ac->ic", Ja_aux_var, dQaux)
    return jv
