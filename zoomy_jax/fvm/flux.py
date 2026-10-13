"""Module `zoomy_jax.fvm.flux`."""

import jax.numpy as jnp
import jax
from functools import partial


class Flux:
    """Flux. (class)."""
    def get_flux_operator(self, model):
        """Get flux operator."""
        pass

class Zero(Flux):
    """Zero. (class)."""
    def get_flux_operator(self, model):
        @jax.jit
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """Compute."""
            return jnp.zeros_like(Qi)

        return compute


class CenteredFlux(Flux):
    """Centered (non-dissipative) conservative flux.

    Computes F_num = 0.5*(F(Q_L)+F(Q_R)).n

    No Rusanov dissipation -- that is handled by the nonconservative
    fluctuation operator.  This mirrors the NumPy NonconservativeRusanov
    where `get_viscosity_identity_flux()` returns zero.
    """

    def get_flux_operator(self, model):
        flux_fn = model.flux
        dim = model.dimension

        def _single(qi, qj, qauxi, qauxj, parameters, normal):
            """Single-face centered conservative flux (no dissipation)."""
            Fi = flux_fn(qi, qauxi, parameters)     # (n_vars, dim)
            Fj = flux_fn(qj, qauxj, parameters)
            n = normal[:dim]
            Fn_i = Fi[:, :dim] @ n                   # (n_vars,)
            Fn_j = Fj[:, :dim] @ n
            return 0.5 * (Fn_i + Fn_j)

        @partial(jax.named_call, name="centered_flux")
        @jax.jit
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """Vectorised centered conservative flux."""
            F_num = jax.vmap(
                _single,
                in_axes=(1, 1, 1, 1, None, 1),
            )(Qi, Qj, Qauxi, Qauxj, parameters, normal)
            return F_num.T  # (n_vars, n_faces)

        return compute


class Rusanov(Flux):
    """Full Rusanov (Lax-Friedrichs) conservative flux with dissipation.

    Computes F_num = 0.5*(F(Q_L)+F(Q_R)).n - 0.5*sM*(Q_R - Q_L)
    """

    def get_flux_operator(self, model):
        flux_fn = model.flux
        eig_fn = model.eigenvalues
        dim = model.dimension

        def _single(qi, qj, qauxi, qauxj, parameters, normal):
            """Single-face Rusanov flux."""
            Fi = flux_fn(qi, qauxi, parameters)
            Fj = flux_fn(qj, qauxj, parameters)
            n = normal[:dim]
            Fn_i = Fi[:, :dim] @ n
            Fn_j = Fj[:, :dim] @ n
            ev_i = eig_fn(qi, qauxi, parameters, normal)
            ev_j = eig_fn(qj, qauxj, parameters, normal)
            sM = jnp.max(jnp.maximum(jnp.abs(ev_i), jnp.abs(ev_j)))
            return 0.5 * (Fn_i + Fn_j) - 0.5 * sM * (qj - qi)

        @partial(jax.named_call, name="rusanov_flux")
        @jax.jit
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """Vectorised Rusanov conservative flux."""
            F_num = jax.vmap(
                _single,
                in_axes=(1, 1, 1, 1, None, 1),
            )(Qi, Qj, Qauxi, Qauxj, parameters, normal)
            return F_num.T

        return compute

