import numpy as np
import os
import sys

from library.solver.baseclass import BaseYaml  # nopep8

main_dir = os.getenv("SMPYTHON")


class Quasilinear(BaseYaml):
    yaml_tag = "!Quasilinear"

    def set_default_parameters(self):
        self.scheme = None

    def evaluate(self, A, time, Q, **kwargs):
        if self.scheme is None:
            return Q
        return getattr(sys.modules[__name__], self.scheme)(A, time, Q, **kwargs)


def price_lf(A, time, Q, **kwargs):
    dt = kwargs["dt"]
    dx = kwargs["dx"]
    Qplus = 1 / 2 * (Q[:, 3:-1] + Q[:, 1:-3])
    Qminus = Q[:, 3:-1] - Q[:, 1:-3]
    Qnew = Qplus - 1 / 2 * dt / dx[:, 2:-2] * np.einsum(
        "jk..., k...->j...", (A(Qplus)), (Qminus)
    )
    Q2 = np.array(Q)
    Q2[:, 2:-2] = Qnew
    return Q2


def price_lw(A, time, Q, **kwargs):
    dt = kwargs["dt"]
    dx = kwargs["dx"]
    Q2 = np.array(Q)
    QLWplus = 1 / 2 * (Q[:, 2:-2] + Q[:, 3:-1]) - 1 / 2 * dt / dx[:, 2:-2] * np.einsum(
        "jk..., k...->j...",
        A(1.0 / 2.0 * (Q[:, 2:-2] + Q[:, 3:-1])),
        Q[:, 3:-1] - Q[:, 2:-2],
    )
    QLWminus = 1 / 2 * (Q[:, 2:-2] + Q[:, 1:-3]) - 1 / 2 * dt / dx[:, 2:-2] * np.einsum(
        "jk...,k...->j...",
        A(1.0 / 2.0 * (Q[:, 2:-2] + Q[:, 1:-3])),
        (Q[:, 2:-2] - Q[:, 1:-3]),
    )
    Qnew = Q[:, 2:-2] - dt / dx[:, 2:-2] * np.einsum(
        "jk..., k...->j...", A(Q[:, 2:-2]), QLWplus - QLWminus
    )
    Q2[:, 2:-2] = Qnew
    return Q2

