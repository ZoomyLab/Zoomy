"""``newton_root`` stopped at ``maxiter`` above ``tol`` logs a warning, under
``jit`` and ``jvp`` alike, and leaves the result unchanged.

Stiff toy: ``R(x) = x + k x^3 - b`` cell-wise, ``k = 1e3``, from ``x0 = 1``;
one Newton update is far from the root.
"""
import pytest
import jax
import jax.numpy as jnp

from zoomy_core.misc.logger_config import logger
from zoomy_jax.fvm.implicit_ad import cellwise_solve, newton_root

K = 1e3


def _root(b, maxiter):
    def residual(x):
        return x + K * x ** 3 - b

    def jvp(x, v):
        return jax.jvp(residual, (x,), (v,))[1]

    return newton_root(residual, jnp.ones_like(b), jvp, cellwise_solve,
                       tol=1e-10, maxiter=maxiter)


@pytest.fixture
def warnings_logged():
    """The text of every WARNING the zoomy (loguru) logger emits, flushed
    through ``jax.effects_barrier`` and cleared on each call."""
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")

    def read():
        jax.effects_barrier()
        out = "".join(msgs)
        msgs.clear()
        return out

    yield read
    logger.remove(sink)


@pytest.mark.small
@pytest.mark.jax
def test_newton_root_warns_when_unconverged(warnings_logged):
    b = jnp.full((1, 3), 2.0)

    x_conv = jax.jit(lambda b: _root(b, 50))(b)
    assert "newton_root: not converged" not in warnings_logged()
    assert float(jnp.max(jnp.abs(x_conv + K * x_conv ** 3 - b))) < 1e-10

    x1 = jax.jit(lambda b: _root(b, 1))(b)
    err = warnings_logged()
    assert "newton_root: not converged after 1 of 1" in err
    assert "tol = 1.000e-10" in err

    # same iterate as the unwarned loop would return: one Newton update
    x_ref = 1.0 - (1.0 + K - 2.0) / (1.0 + 3 * K)
    assert jnp.allclose(x1, x_ref, rtol=1e-14, atol=0)

    _, dx = jax.jvp(lambda b: _root(b, 1), (b,), (jnp.ones_like(b),))
    assert "newton_root: not converged" in warnings_logged()
    assert bool(jnp.all(jnp.isfinite(dx)))

    # reverse mode, inside the checkpointed loop the reverse time march uses
    eqxi = pytest.importorskip("equinox.internal")

    def J(b):
        def body(c):
            k, x = c
            return k + 1, _root(b + x, 1)

        _, x = eqxi.while_loop(lambda c: c[0] < 2, body, (0, jnp.zeros_like(b)),
                               kind="checkpointed", checkpoints=1)
        return jnp.sum(x)

    gb = jax.grad(J)(b)
    assert "newton_root: not converged" in warnings_logged()
    assert bool(jnp.all(jnp.isfinite(gb)))
