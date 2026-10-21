"""Sample-based hyperbolicity test.

Given a linearised `SystemModel` (or pencil matrices already in
hand), evaluate the generalised eigenvalues at a grid / random sample
of base states and check the imaginary parts.  Reports the fraction of
samples that are hyperbolic (all eigenvalues real to within tolerance)
and a summary of any non-hyperbolic regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .pencil import (
    extract_quasilinear_pencil,
    sample_generalised_eigenvalues,
    _eval_matrix_numerically,
)


# ---------------------------------------------------------------------------
# Single-state hyperbolicity test
# ---------------------------------------------------------------------------

def is_hyperbolic_at(M_x: sp.Matrix, M_t: sp.Matrix, sample: Dict, *,
                     tol: float = 1e-9, drop_infinite: bool = True
                     ) -> Tuple[bool, np.ndarray]:
    """Evaluate `(M_x, M_t)` at `sample` and check eigenvalues.

    Returns `(hyperbolic, eigenvalues)`.  `hyperbolic` is `True`
    iff every finite eigenvalue has `|imag| < tol`.
    """
    eigs = sample_generalised_eigenvalues(M_x, M_t, [sample],
                                          drop_infinite=drop_infinite)[0]
    hyperbolic = bool(np.all(np.abs(np.imag(eigs)) < tol))
    return hyperbolic, eigs


# ---------------------------------------------------------------------------
# Sampling over a parameter cube
# ---------------------------------------------------------------------------

@dataclass
class HyperbolicitySample:
    sample: Dict[Any, float]
    eigenvalues: np.ndarray
    hyperbolic: bool


@dataclass
class HyperbolicityReport:
    samples: List[HyperbolicitySample]
    fraction_hyperbolic: float
    nonhyperbolic_samples: List[HyperbolicitySample] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.samples)
        nh = len(self.nonhyperbolic_samples)
        lines = [
            f"Hyperbolicity sample: {n} states, "
            f"{n - nh}/{n} hyperbolic "
            f"({100 * self.fraction_hyperbolic:.2f}%).",
        ]
        if self.notes:
            lines += ["  notes:"] + [f"    - {n}" for n in self.notes]
        if nh > 0:
            lines.append(f"  first {min(3, nh)} non-hyperbolic example(s):")
            for s in self.nonhyperbolic_samples[:3]:
                vals = ", ".join(f"{k}={v}" for k, v in s.sample.items())
                imag_max = float(np.max(np.abs(np.imag(s.eigenvalues))))
                lines.append(f"    {vals}  (max |Im λ| = {imag_max:.3e})")
        return "\n".join(lines)


def _normalise_ranges(parameter_ranges: Dict[Any, Sequence]) -> Dict[Any, Tuple[float, float]]:
    out = {}
    for sym, rng in parameter_ranges.items():
        if not (hasattr(rng, "__len__") and len(rng) == 2):
            raise TypeError(f"parameter_ranges[{sym}] must be a (lo, hi) pair")
        out[sym] = (float(rng[0]), float(rng[1]))
    return out


def sample_hyperbolicity(M_x: sp.Matrix, M_t: sp.Matrix,
                         parameter_ranges: Dict[Any, Sequence],
                         *, n_samples: int = 1000,
                         rng: Optional[np.random.Generator] = None,
                         tol: float = 1e-9,
                         drop_infinite: bool = True,
                         constraint_filter: Optional[Callable[[Dict], bool]] = None,
                         max_attempts: int = 10,
                         ) -> HyperbolicityReport:
    """Random-uniform sample over a hyper-rectangle of parameters.

    Args:
        M_x, M_t       : the pencil matrices (sympy).
        parameter_ranges: dict `{sympy_symbol: (lo, hi)}` for the
                          variables you want to sample.  Symbols not in
                          this dict must already be substituted-out in
                          the pencil (e.g. fixed at chosen values).
        n_samples      : number of *accepted* samples to draw.
        rng            : optional `np.random.Generator`.
        tol            : imaginary-part threshold for "real".
        drop_infinite  : drop infinite generalised eigenvalues (typical
                          for systems with constraints).
        constraint_filter: optional callable `sample_dict → bool`.  A
                          sample is rejected (and another drawn) if this
                          returns False — useful for excluding e.g.
                          h ≤ 0.
        max_attempts   : per accepted sample, how many draws before
                          giving up.

    Returns `HyperbolicityReport`.
    """
    if rng is None:
        rng = np.random.default_rng()

    parameter_ranges = _normalise_ranges(parameter_ranges)
    keys = list(parameter_ranges.keys())
    los = np.array([parameter_ranges[k][0] for k in keys], dtype=float)
    his = np.array([parameter_ranges[k][1] for k in keys], dtype=float)

    samples = []
    attempts = 0
    accepted = 0
    while accepted < n_samples and attempts < n_samples * max_attempts:
        attempts += 1
        u = rng.uniform(low=los, high=his)
        sub = {k: float(v) for k, v in zip(keys, u)}
        if constraint_filter is not None and not constraint_filter(sub):
            continue
        try:
            hyperbolic, eigs = is_hyperbolic_at(M_x, M_t, sub,
                                                tol=tol,
                                                drop_infinite=drop_infinite)
        except Exception as exc:                       # pragma: no cover
            # singular B that even pinv can't fix, etc. — skip
            continue
        samples.append(HyperbolicitySample(
            sample=sub, eigenvalues=eigs, hyperbolic=hyperbolic
        ))
        accepted += 1

    nonhyper = [s for s in samples if not s.hyperbolic]
    n = len(samples)
    frac = (n - len(nonhyper)) / n if n > 0 else 0.0
    notes = []
    if attempts > n_samples and accepted < n_samples:
        notes.append(
            f"Only {accepted} samples accepted in {attempts} attempts "
            f"(constraint_filter rejection rate)."
        )
    return HyperbolicityReport(
        samples=samples,
        fraction_hyperbolic=frac,
        nonhyperbolic_samples=nonhyper,
        notes=notes,
    )
