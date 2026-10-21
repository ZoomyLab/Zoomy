"""Analytic Jacobian-vector product of the source term."""

import jax.numpy as jnp


def analytic_source_jvp_jax(runtime_model, Q, Qaux, V, parameters, dQaux=None):
    """`dS = dS/dQ . V + dS/dQaux . dQaux` from the lowered symbolic source
    Jacobians.

    Parameters
    ----------
    runtime_model : JaxRuntime
        Supplies `source_jacobian_wrt_variables` and, when `dQaux` is
        given, `source_jacobian_wrt_aux_variables` (`(n_eq, n_aux)` per
        cell).
    Q, Qaux : `(n_state, n_cells)`, `(n_aux, n_cells)`
        The state the Jacobians are evaluated at.
    V : `(n_state, n_cells)`
        The state direction.
    parameters : `(n_parameters,)`
        The runtime parameter vector (traced or concrete).
    dQaux : `(n_aux, n_cells)`, optional
        The aux direction induced by `V` (`d aux(Q) . V`).  `None` or a
        model without aux: the state block alone (aux frozen).
    """
    Jq = runtime_model.source_jacobian_wrt_variables(Q, Qaux, parameters)
    jv = jnp.einsum("ijc,jc->ic", Jq, V)
    if dQaux is None or dQaux.shape[0] == 0:
        return jv
    Ja = runtime_model.source_jacobian_wrt_aux_variables(Q, Qaux, parameters)
    return jv + jnp.einsum("iac,ac->ic", Ja, dQaux)
