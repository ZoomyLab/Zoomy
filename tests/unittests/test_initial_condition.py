import numpy as np

from library.initial_condition import *


def test_default():
    Q = np.zeros((10, 3))
    x = np.linspace(0, 1, 10)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    X = np.vstack((x, y, z)).T
    Q = Default().apply(Q, X)
    assert np.allclose(Q, np.ones_like(Q))


def test_RP():
    def f(n: int) -> FArray:
        return np.ones(n, dtype=float)

    Q = np.zeros((10, 3))
    x = np.linspace(-1, 1, 10)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    X = np.vstack((x, y, z)).T
    Q = RP(left=f).apply(Q, X)
    assert np.allclose(Q[:5, :], np.ones((5, 3)))
    assert np.allclose(Q[5:, :], 2 * np.ones((5, 3)))


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
    Q = UserFunction(function=f).apply(Q, X)
    assert np.allclose(Q[:5, :], np.ones((5, 3)))
    assert np.allclose(Q[5:, :], 2 * np.ones((5, 3)))


if __name__ == "__main__":
    test_default()
    test_RP()
    test_UserFunction()
