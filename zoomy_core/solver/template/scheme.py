"""Time schemes: core DATA, unrolled into statements at build time.

``ShuOsher``  explicit SSP schemes in Shu-Osher form

    Q^(0) = Q^n
    Q^(k) = a_k Q^n + (1 - a_k) [Q^(k-1) + beta_k dt L(Q^(k-1), t + c_k dt)]
    Q^{n+1} = Q^(s)

with stage times ``c_k`` (D3).  ``EULER``, ``SSPRK2`` (Heun) and ``SSPRK3``
(Shu-Osher 1988) are provided; the default for a reconstruction order is
:func:`ShuOsher.for_order`, but the scheme is independent of the order.

``Imex``  IMEX data: ``kind="lie"`` (D2 default: the implicit stages run after
the explicit RK step) or ``kind="ark"`` (per-stage implicit solves with an
IMEX-ARK Butcher pair, ``ARS232`` / ``ARS343``).  Lowered in sub-stage 1b.

``Compose``  operator splitting: a sequence of ``(part, dt_fraction)``; each
part is a scheme + stage list advanced over ``dt_fraction * dt``.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class ShuOsher:
    name: str
    a: tuple            # weight of Q^n per stage
    c: tuple            # stage time fraction per stage (D3)
    beta: tuple = ()    # dt scale of the stage increment (default 1)

    def __post_init__(self):
        a = tuple(float(x) for x in self.a)
        c = tuple(float(x) for x in self.c)
        beta = tuple(float(x) for x in self.beta) or (1.0,) * len(a)
        if not (len(a) == len(c) == len(beta)) or not a:
            raise ValueError("ShuOsher needs equally long, non-empty a, c, beta")
        if a[0] != 0.0:
            raise ValueError("the first Shu-Osher stage starts from Q^n (a_0 = 0)")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "beta", beta)

    @property
    def stages(self) -> tuple:
        """``((a_k, c_k), ...)``."""
        return tuple(zip(self.a, self.c))

    @property
    def n_stages(self) -> int:
        return len(self.a)

    def butcher(self):
        """Equivalent explicit Butcher tableau ``(A, b, c)`` (for checks)."""
        s = self.n_stages
        # Y_k = sum_j alpha_kj Y_j + beta_k dt L(Y_{k-1}),  Y_0 = Q^n
        # stage values X_k = Q^(k-1) at which L is evaluated (k = 1..s)
        A = np.zeros((s, s))
        b = np.zeros(s)
        # coefficient vectors (over L evaluations) of Q^(k), k = 0..s
        coeff = [np.zeros(s)]
        for k in range(s):
            prev = coeff[-1]
            new = (1.0 - self.a[k]) * prev.copy()
            new[k] += (1.0 - self.a[k]) * self.beta[k]
            coeff.append(new)
            if k + 1 < s:
                A[k + 1] = new
        b[:] = coeff[-1]
        cc = A.sum(axis=1)
        return A, b, cc

    def check_consistency(self, tol=1e-14):
        """The declared stage times equal the Butcher row sums."""
        _, b, cc = self.butcher()
        if abs(b.sum() - 1.0) > tol:
            raise ValueError(f"{self.name}: weights sum to {b.sum()}")
        if np.max(np.abs(cc - np.asarray(self.c))) > tol:
            raise ValueError(f"{self.name}: stage times {self.c} != row sums {cc}")
        return True

    @staticmethod
    def for_order(order: int) -> "ShuOsher":
        try:
            return {1: EULER, 2: SSPRK2, 3: SSPRK3}[int(order)]
        except KeyError:
            raise KeyError(f"no default SSP scheme for order {order}") from None


EULER = ShuOsher("euler", a=(0.0,), c=(0.0,))
SSPRK2 = ShuOsher("ssprk2", a=(0.0, 0.5), c=(0.0, 1.0))
SSPRK3 = ShuOsher("ssprk3", a=(0.0, 0.75, 1.0 / 3.0), c=(0.0, 1.0, 0.5))


@dataclasses.dataclass(frozen=True)
class ButcherTableau:
    A: tuple
    b: tuple
    c: tuple

    def __post_init__(self):
        A = tuple(tuple(float(x) for x in row) for row in self.A)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", tuple(float(x) for x in self.b))
        object.__setattr__(self, "c", tuple(float(x) for x in self.c))
        s = len(self.b)
        if len(A) != s or any(len(r) != s for r in A) or len(self.c) != s:
            raise ValueError("Butcher tableau dimensions disagree")

    @property
    def n_stages(self):
        return len(self.b)

    def as_arrays(self):
        return np.asarray(self.A), np.asarray(self.b), np.asarray(self.c)


@dataclasses.dataclass(frozen=True)
class ImexTableau:
    """IMEX-ARK pair: explicit and implicit Butcher tableaus with common c."""

    name: str
    order: int
    explicit: ButcherTableau
    implicit: ButcherTableau

    def order_conditions(self) -> dict:
        """Residuals of the classical and coupling order conditions up to
        ``order`` (all should be ~0)."""
        Ae, be, ce = self.explicit.as_arrays()
        Ai, bi, ci = self.implicit.as_arrays()
        res = {
            "c_explicit_rowsum": float(np.max(np.abs(Ae.sum(1) - ce))),
            "c_implicit_rowsum": float(np.max(np.abs(Ai.sum(1) - ci))),
            "c_equal": float(np.max(np.abs(ce - ci))),
            "b_e": float(abs(be.sum() - 1)), "b_i": float(abs(bi.sum() - 1)),
        }
        if self.order >= 2:
            res.update({
                "bc_e": float(abs(be @ ce - 0.5)),
                "bc_i": float(abs(bi @ ci - 0.5)),
                "be_ci": float(abs(be @ ci - 0.5)),
                "bi_ce": float(abs(bi @ ce - 0.5)),
            })
        if self.order >= 3:
            for name, b in (("e", be), ("i", bi)):
                res[f"bcc_{name}"] = float(abs(b @ (ce * ce) - 1 / 3))
            for nb, b in (("e", be), ("i", bi)):
                for nA, A in (("e", Ae), ("i", Ai)):
                    res[f"bAc_{nb}{nA}"] = float(abs(b @ A @ ce - 1 / 6))
        return res


def _ars232() -> ImexTableau:
    """ARS(2,3,2), Ascher-Ruuth-Spiteri 1997 sec. 2.6: gamma = 1 - 1/sqrt 2,
    delta = -2 sqrt 2 / 3; stiffly accurate."""
    g = 1.0 - 1.0 / math.sqrt(2.0)
    d = -2.0 * math.sqrt(2.0) / 3.0
    expl = ButcherTableau(A=((0, 0, 0), (g, 0, 0), (d, 1 - d, 0)),
                          b=(0, 1 - g, g), c=(0, g, 1))
    impl = ButcherTableau(A=((0, 0, 0), (0, g, 0), (0, 1 - g, g)),
                          b=(0, 1 - g, g), c=(0, g, 1))
    return ImexTableau("ARS232", 2, expl, impl)


def _ars343() -> ImexTableau:
    """ARS(3,4,3), Ascher-Ruuth-Spiteri 1997 sec. 2.7.

    gamma is the middle root of 6x^3 - 18x^2 + 9x - 1 (the L-stable SDIRK3
    value 0.4358665215...), the implicit tableau follows in closed form;
    the explicit free parameters a42 = a43 = 0.5529291479 are the published
    ones and a32 / a31 / a41 are solved from the row sums and the third-order
    condition b^T A_E c = 1/6 -- so every condition holds to roundoff
    (the published 10-digit coefficients satisfy them to ~1e-10)."""
    roots = np.roots([6.0, -18.0, 9.0, -1.0])
    g = float(sorted(r.real for r in roots if abs(r.imag) < 1e-12)[1])
    b1 = -1.5 * g * g + 4.0 * g - 0.25
    b2 = 1.5 * g * g - 5.0 * g + 1.25
    c = (0.0, g, (1.0 + g) / 2.0, 1.0)
    impl = ButcherTableau(
        A=((0, 0, 0, 0), (0, g, 0, 0), (0, (1 - g) / 2, g, 0), (0, b1, b2, g)),
        b=(0, b1, b2, g), c=c)
    a42 = a43 = 0.5529291479
    a41 = 1.0 - a42 - a43
    # b3 * a32 * c2 + b4 * (a42 c2 + a43 c3) = 1/6
    a32 = (1.0 / 6.0 - g * (a42 * c[1] + a43 * c[2])) / (b2 * c[1])
    a31 = c[2] - a32
    expl = ButcherTableau(
        A=((0, 0, 0, 0), (g, 0, 0, 0), (a31, a32, 0, 0), (a41, a42, a43, 0)),
        b=(0, b1, b2, g), c=c)
    return ImexTableau("ARS343", 3, expl, impl)


ARS232 = _ars232()
ARS343 = _ars343()


@dataclasses.dataclass(frozen=True)
class Imex:
    """IMEX scheme data (lowered in sub-stage 1b).

    ``kind="lie"`` (D2 default): explicit RK step with ``explicit`` (a
    :class:`ShuOsher`), then the implicit stages over the full dt.
    ``kind="ark"``: per-stage implicit solves with the ``tableau`` pair."""

    kind: str = "lie"
    explicit: Any = None
    tableau: Any = None

    def __post_init__(self):
        if self.kind not in ("lie", "ark"):
            raise ValueError("Imex kind must be 'lie' or 'ark'")
        if self.kind == "ark" and self.tableau is None:
            object.__setattr__(self, "tableau", ARS232)
        if self.kind == "ark" and not isinstance(self.tableau, ImexTableau):
            raise TypeError("Imex(kind='ark') needs an ImexTableau")


@dataclasses.dataclass(frozen=True)
class Compose:
    """Operator splitting: ``parts = ((part, dt_fraction), ...)``.  A part is
    any object the step builder accepts (a ``(scheme, stage)`` pair).  Lie
    splitting is ``((A, 1), (B, 1))``; Strang ``((A, .5), (B, 1), (A, .5))``."""

    parts: tuple

    def __post_init__(self):
        parts = tuple((p, float(f)) for p, f in self.parts)
        if not parts:
            raise ValueError("Compose needs at least one part")
        object.__setattr__(self, "parts", parts)


def strang(a, b) -> Compose:
    return Compose(((a, 0.5), (b, 1.0), (a, 0.5)))


def lie(*parts) -> Compose:
    return Compose(tuple((p, 1.0) for p in parts))


__all__ = ["ShuOsher", "EULER", "SSPRK2", "SSPRK3", "ButcherTableau",
           "ImexTableau", "ARS232", "ARS343", "Imex", "Compose", "strang",
           "lie"]
