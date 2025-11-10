import numpy as np
import pytest

from zoomy_core.model.initial_conditions import *


@pytest.mark.critical
def test_default():
    Q = np.zeros((10, 3))
    x = np.linspace(0, 1, 10)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    X = np.vstack((x, y, z)).T
    Q = Constant(lambda n_variables: np.ones(n_variables)).apply(Q, X)
    assert np.allclose(Q, np.ones_like(Q))


@pytest.mark.critical
def test_RP():
    def fl(n: int) -> FArray:
        return np.ones(n, dtype=float)

    def fr(n: int) -> FArray:
        return 2 * np.ones(n, dtype=float)

    Q = np.zeros((10, 3))
    x = np.linspace(-1, 1, 10)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    X = np.vstack((x, y, z)).T
    Q = RP().apply(X, Q)
    assert np.allclose(Q[:5, :], np.ones((5, 3)))
    assert np.allclose(Q[5:, :], 2 * np.ones((5, 3)))


@pytest.mark.critical
def test_UserFunction():
    def f(x: FArray) -> FArray:
        if x[0] < 0:
            return np.ones(x.shape[0])
        return 2 * np.ones(x.shape[0])

    Q = np.zeros((10, 3))
    x = np.linspace(-1, 1, 10)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    X = np.vstack((x, y, z)).T
    Q = UserFunction(function=f).apply(X, Q)
    assert np.allclose(Q[:5, :], np.ones((5, 3)))
    assert np.allclose(Q[5:, :], 2 * np.ones((5, 3)))


if __name__ == "__main__":
    test_default()
    test_RP()
    test_UserFunction()
