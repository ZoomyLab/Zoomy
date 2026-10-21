"""`zoomy_jax.fvm.eigvals.eigvals_device` (LAPACK dgeev's eigenvalue path, batched on the
device) agrees with LAPACK to round-off on random, structured and degenerate matrices; its
derivative is `jnp.linalg.eigvals`'.  Runs the device algorithm on whatever backend is there.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from zoomy_jax.fvm.eigvals import eigvals, eigvals_device  # noqa: E402

pytestmark = [pytest.mark.small, pytest.mark.jax]

_RNG = np.random.default_rng(7)


def _matched_error(ev, ref, A):
    """Max over matrices of the eigenvalue error (multisets matched greedily) / max |A|."""
    worst = 0.0
    for e, r, a in zip(ev.reshape(-1, A.shape[-1]), ref.reshape(-1, A.shape[-1]),
                       A.reshape(-1, *A.shape[-2:])):
        r = list(r)
        for x in e:
            k = int(np.argmin([abs(x - y) for y in r]))
            worst = max(worst, abs(x - r.pop(k)) / max(np.abs(a).max(), 1e-300))
    return worst


def _cases():
    n = 14
    vam = _RNG.standard_normal((40, n, n))
    vam[:, 10:, :] = 0.0                       # zero rows: isolated eigenvalues (pressure rows)
    vam[:, 0, :] = 0.0
    holes = _RNG.standard_normal((40, n, n))
    holes[:, 3, :] = 0.0
    holes[:, :, 7] = 0.0
    sym = _RNG.standard_normal((40, n, n))
    Q, _ = np.linalg.qr(_RNG.standard_normal((n, n)))
    return {
        "random14": _RNG.standard_normal((60, n, n)),
        "random3": _RNG.standard_normal((60, 3, 3)),
        "random2": _RNG.standard_normal((60, 2, 2)),
        "random1": _RNG.standard_normal((5, 1, 1)),
        "vam_like": vam,
        "zero_rows_cols": holes,
        "triangular": np.triu(_RNG.standard_normal((20, n, n))),
        "zero": np.zeros((3, n, n)),
        "symmetric": sym + np.swapaxes(sym, 1, 2),
        "real_spectrum": (Q @ np.diag(np.linspace(-2, 2, n)) @ Q.T)[None],
        "badly_scaled": _RNG.standard_normal((40, n, n)) * np.logspace(-6, 6, n)[None, :, None],
        "rotation": np.array([[[0.0, -1.0], [1.0, 0.0]]]),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_eigvals_device_matches_lapack(name):
    A = _cases()[name]
    ev = np.asarray(jax.jit(eigvals_device)(jnp.asarray(A)))
    ref = np.linalg.eigvals(A)
    assert np.isfinite(ev).all()
    assert _matched_error(ev, ref, A) <= 1e-13
    # the CFL quantity: max |Re lambda| per matrix
    r, d = np.abs(ref.real).max(-1), np.abs(ev.real).max(-1)
    assert np.all(np.abs(r - d) <= 1e-13 * np.maximum(r, np.abs(A).max((-2, -1))))


def test_eigvals_device_batch_independent():
    """A matrix's eigenvalues do not depend on the batch it is computed in (bitwise), for
    batches of two or more (XLA compiles a batch of one with other contractions)."""
    A = _cases()["vam_like"]
    f = jax.jit(eigvals_device)
    whole = np.asarray(f(jnp.asarray(A)))
    for i in (0, 7, len(A) - 2):
        assert np.array_equal(np.asarray(f(jnp.asarray(A[i:i + 2]))), whole[i:i + 2])
    assert np.array_equal(np.asarray(f(jnp.asarray(np.concatenate([A, A]))))[:len(A)], whole)


def test_eigvals_device_nan_input_gives_nan():
    A = _cases()["random14"][:2].copy()
    A[1, 2, 3] = np.nan
    ev = np.asarray(jax.jit(eigvals_device)(jnp.asarray(A)))
    assert np.isfinite(ev[0]).all() and np.isnan(ev[1]).all()


def test_eigvals_device_jvp_is_lapack_jvp():
    A = jnp.asarray(_RNG.standard_normal((4, 5, 5)))
    dA = jnp.asarray(_RNG.standard_normal((4, 5, 5)))
    f = lambda a: jnp.max(jnp.abs(jnp.real(eigvals_device(a))), axis=-1)   # noqa: E731
    g = lambda a: jnp.max(jnp.abs(jnp.real(jnp.linalg.eigvals(a))), axis=-1)   # noqa: E731
    _, t1 = jax.jvp(f, (A,), (dA,))
    _, t2 = jax.jvp(g, (A,), (dA,))
    np.testing.assert_allclose(np.asarray(t1), np.asarray(t2), rtol=1e-12)


def test_eigvals_dispatch_cpu_is_lapack():
    """On the CPU `eigvals` is `jnp.linalg.eigvals` itself (bitwise)."""
    if jax.default_backend() != "cpu":
        pytest.skip("CPU dispatch")
    A = jnp.asarray(_cases()["random14"])
    assert np.array_equal(np.asarray(jax.jit(eigvals)(A)),
                          np.asarray(jax.jit(jnp.linalg.eigvals)(A)))


def test_eigvals_device_under_vmap():
    """Per cell under `vmap` (the runtimes' numerical-eigenvalue fallbacks) = the batched call
    (to round-off: the vmapped prelude is compiled with other contractions)."""
    A = _cases()["vam_like"]
    per_cell = np.asarray(jax.jit(jax.vmap(eigvals_device))(jnp.asarray(A)))
    whole = np.asarray(jax.jit(eigvals_device)(jnp.asarray(A)))
    assert _matched_error(per_cell, whole, A) <= 1e-13
