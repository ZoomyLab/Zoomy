"""Named turbulent Shallow Moment Equations models built on `SME`: the algebraic Elder
eddy-viscosity model (`ElderSME`).
The two-equation k-epsilon model lives in `ke_sme.py`.
"""
from __future__ import annotations

from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    ElderViscosity, RoughWall, StressFree)


class ElderSME(SME):
    """SME with the Elder parabolic eddy viscosity `nu_t = kappa u_star h zeta (1 - zeta)` and
    a turbulent rough-wall bed; algebraic, adds no state. Pass `closures=[...]` to override
    the default closure set.

    References:
        Elder 1959, https://doi.org/10.1017/s0022112059000374
    """

    def __init__(self, **params):
        """Install the Elder bulk stress, rough-wall bed and stress-free surface as default
        closures, then build the `SME`.
        """
        params.setdefault(
            "closures", [ElderViscosity(), RoughWall(), StressFree()])
        super().__init__(**params)


__all__ = ["ElderSME"]
