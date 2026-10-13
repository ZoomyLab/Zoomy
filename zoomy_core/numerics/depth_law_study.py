"""One-call `NumericalSystemModel` construction over the depth-law axes (regularization path,
`eps`, eigenvalue treatment, wet/dry guard) for parameter sweeps.
It measures nothing and asserts no physics; every regularization path uses the Kurganov-Petrova
reciprocal `kp_hinv`.

References:
    Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
"""
from __future__ import annotations

import functools
import inspect
import itertools
from typing import Any, Callable, Iterator, Optional, Union

from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.systemmodel.operations import _DEFAULT_REGULARIZATION_EPS
from zoomy_core.systemmodel.system_model import SystemModel

__all__ = [
    "AXES",
    "PATHS",
    "GUARDS",
    "EIGENVALUE_TREATMENTS",
    "DEFAULT_EPS",
    "build_nsm",
    "iter_axes",
    "axis_id",
]

#: Regularization paths.  ``"direct"`` and ``"aux"`` are the two study variants;
#: ``"legacy"`` and ``"none"`` are controls (the shipped behaviour and the
#: un-regularized system).
PATHS = ("direct", "aux", "legacy", "none")

#: Wet/dry eigenvalue guards.  NB ``"both"`` is byte-identical to ``"gate"`` —
#: ``gate_eigenvalues_dry`` carries the power guard internally (REQ-74: a
#: branchless ``conditional`` evaluates both arms, so the wet branch must stay
#: real) and the guard is idempotent.  REQ-181 pins that identity.  Both are
#: kept as separate axis points so the sweep records the composition explicitly
#: rather than assuming the identity still holds.
GUARDS = (None, "power", "gate", "both")

#: Spectrum treatment under the chosen path.
EIGENVALUE_TREATMENTS = ("regularize", "exclude")

#: Default regularization scale — the user's "start near the old working state"
#: (SWE / Malpasset shipped ``wet_dry_eps = 1e-2``).  A STUDY default on the
#: study paths only; the legacy path still reads ``wet_dry_eps``.
DEFAULT_EPS = _DEFAULT_REGULARIZATION_EPS

#: The sweep grid: the two study paths × the four guards × the two spectrum
#: treatments = 16 points.  ``eps`` is deliberately NOT part of the grid — it is
#: a continuous knob, so a sweep varies it as an outer loop over whichever of
#: these 16 structural points it cares about.
AXES = {
    "path": ("direct", "aux"),
    "guard": GUARDS,
    "eigenvalues": EIGENVALUE_TREATMENTS,
}


def iter_axes(axes: Optional[dict] = None) -> Iterator[dict]:
    """Yield one `{axis: value}` kwargs dict per point of the cartesian product of `axes`
    (default `AXES`), ready for `build_nsm`.
    """
    axes = AXES if axes is None else axes
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, values))


def axis_id(combo: dict) -> str:
    """Short stable label of an axis point for test ids and filenames, e.g.
    `"aux-noguard-evexclude"`.
    """
    guard = combo.get("guard") or "noguard"
    return (f"{combo.get('path', 'legacy')}-{guard}"
            f"-ev{combo.get('eigenvalues', 'regularize')}")


def build_nsm(
    model: Union[Any, Callable[[], Any]],
    *,
    path: str = "aux",
    eps: float = DEFAULT_EPS,
    guard: Optional[str] = None,
    eigenvalues: str = "regularize",
    normalize_normal: bool = True,
    riemann=None,
    **nsm_kwargs,
) -> NumericalSystemModel:
    """Build a derived `NumericalSystemModel` at one point of the depth-law axes.

    Args:
        model: a `Model`, a `SystemModel` or a zero-argument callable returning either; prefer
            a callable, since the NSM promotes a `SystemModel` in place.
        path: `"direct"` | `"aux"` | `"legacy"` | `"none"` (see `PATHS`).
        eps: regularization scale for `"direct"`/`"aux"`, not `wet_dry_eps`.
        guard: `None` | `"power"` | `"gate"` | `"both"`.
        eigenvalues: `"regularize"` | `"exclude"`; ignored by `"legacy"` and `"none"`.
        riemann: Riemann class, forwarded verbatim (`None` keeps the NSM default).
        **nsm_kwargs: forwarded to `NumericalSystemModel.from_system_model`.

    Returns:
        The NSM with its operators, spectrum and aux rows final.
    """
    if path not in PATHS:
        raise ValueError(
            f"path={path!r} is not a legal regularization path; expected one "
            f"of {list(PATHS)}.")
    if guard not in GUARDS:
        raise ValueError(
            f"guard={guard!r} is not a legal eigenvalue guard; expected one "
            f"of {list(GUARDS)}.")
    if eigenvalues not in EIGENVALUE_TREATMENTS:
        raise ValueError(
            f"eigenvalues={eigenvalues!r} is not a legal spectrum treatment; "
            f"expected one of {list(EIGENVALUE_TREATMENTS)}.")

    # A bare class or a zero-arg factory is invoked here, so a sweep can pass
    # ``lambda: SWE(dimension=2)`` and get a FRESH system per axis point.
    obj = model
    if isinstance(obj, type) or inspect.isroutine(obj) or isinstance(
            obj, functools.partial):
        obj = obj()

    # ``depth_regularization=None`` is how the NSM spells "legacy default".
    axes = dict(
        riemann=riemann,
        depth_regularization=None if path == "legacy" else path,
        regularization_eps=eps,
        eigenvalue_treatment=eigenvalues,
        eigenvalue_guard=guard,
        normalize_normal=normalize_normal,
        **nsm_kwargs,
    )
    if isinstance(obj, SystemModel):
        return NumericalSystemModel.from_system_model(obj, **axes)
    return NumericalSystemModel.from_model(obj, **axes)
