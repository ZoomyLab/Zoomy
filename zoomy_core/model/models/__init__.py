"""Canonical model derivation classes (lazy-imported to avoid cycles).

After the Model-class rewrite, primitives moved out of this package
into `zoomy_core/model/{basemodel,equation,state,operations}.py`.
This package retains only the derivation subclasses
(`SME`, `VAM`, …).

Lazy import (PEP 562) is used so that importing
`zoomy_core.model.derivation.basisfunctions` (e.g. from
`zoomy_core.model.operations`) does not trigger eager loading of
the derivation classes — which themselves depend on
`zoomy_core.model.operations` and would create a cycle.

Users get the eager-looking surface (``from zoomy_core.model.models
import SME``) without the cycle.
"""

__all__ = ["SWE", "MalpassetSWE", "SME", "ElderSME", "KESME", "QRKESME",
           "VAM", "MLSWE", "MLSME", "MLVAM", "ClosureState",
           # composable stress / interface closures (closures.py) — the ONLY
           # stress-closure path (the legacy MaterialModel was removed)
           "Closure", "Newtonian", "NavierSlip", "StressFree", "RoughWall",
           "WallFunctionBed", "Bingham", "KEpsilonViscosity", "QRViscosity",
           "ElderViscosity", "InterfaceFlux", "MeanInterface", "UpwindInterface"]


def __getattr__(name):
    if name == "SWE":
        from zoomy_core.model.models.swe import SWE
        return SWE
    if name == "MalpassetSWE":
        from zoomy_core.model.models.malpasset import MalpassetSWE
        return MalpassetSWE
    if name == "SME":
        from zoomy_core.model.models.sme import SME
        return SME
    if name == "ElderSME":
        from zoomy_core.model.models.turbulent_sme import ElderSME
        return ElderSME
    if name == "KESME":
        from zoomy_core.model.models.ke_sme import KESME
        return KESME
    if name == "QRKESME":
        from zoomy_core.model.models.qr_kesme import QRKESME
        return QRKESME
    if name == "VAM":
        from zoomy_core.model.models.vam import VAM
        return VAM
    if name == "MLSWE":
        from zoomy_core.model.models.ml_swe import MLSWE
        return MLSWE
    if name == "MLSME":
        from zoomy_core.model.models.ml_sme import MLSME
        return MLSME
    if name == "MLVAM":
        from zoomy_core.model.models.ml_vam import MLVAM
        return MLVAM
    if name == "ClosureState":
        from zoomy_core.model.models.material import ClosureState
        return ClosureState
    if name in ("Closure", "Newtonian", "NavierSlip", "StressFree", "RoughWall",
                "Bingham", "KEpsilonViscosity", "QRViscosity", "ElderViscosity",
                "WallFunctionBed",
                "InterfaceFlux", "MeanInterface", "UpwindInterface"):
        from zoomy_core.model.models import closures as _cl
        return getattr(_cl, name)
    raise AttributeError(f"module 'zoomy_core.model.models' has no attribute {name!r}")
