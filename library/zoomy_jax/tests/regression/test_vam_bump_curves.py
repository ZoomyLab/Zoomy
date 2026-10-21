"""Escalante bump — the RUN-OUTPUT check: h and the bottom pressure curve.

Replaces ``test_vam1_chorin_split_matches_escalante_projection``, which compared
the elliptic rows to the paper's projection SYMBOLICALLY and failed on a nonzero
residual in the ``P_0`` row.  A symbolic identity is brittle: it breaks on any
algebraically-equivalent rearrangement of the derivation, and the residual it
prints is unreadable.  Checking what the scheme actually PRODUCES — the depth
curve and the bed pressure curve after a real march — is the robust form (user,
2026-07-22).

The bottom pressure is read from the model's OWN ``interpolate_to_3d()``
total-pressure slot at the bed (zeta = 0), never a hard-coded +-2*P_1: whatever
the basis, ``p_b = field_p(zeta=0)``, with the bed values of the basis functions
baked into the reconstruction, so the sign is right by construction.
"""
import time

import numpy as np
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL

FIRST_ORDER = dict(reconstruction=ReconstructionSpec(order=1), time_order=1)


def _bottom_pressure_head(model, h, P0, P1, g):
    """p_b/g from the model's own vertical reconstruction, read at zeta = 0."""
    zeta = sp.Symbol("zeta", real=True)
    p_bed = sp.sympify(model.interpolate_to_3d()[5]).subs(zeta, 0)
    Hs, P0s, P1s = sp.symbols("Hs P0s P1s")
    sub = {}
    for a in p_bed.atoms(AppliedUndef):
        nm = a.func.__name__
        if nm == "h":
            sub[a] = Hs
        elif nm == "P":
            sub[a] = P1s if int(a.args[0]) == 1 else P0s
    p_bed = p_bed.subs(sub)
    by_name = {str(s): s for s in p_bed.free_symbols}   # g/rho carry assumptions
    p_bed = p_bed.subs({by_name["g"]: g, by_name["rho"]: 1.0})
    return sp.lambdify((Hs, P0s, P1s), p_bed, "numpy")(h, P0, P1) / g


@pytest.mark.regression
@pytest.mark.jax
def test_vam_bump_curves(overwrite):
    """March the bump, then pin the h and p_b/g CURVES cell-by-cell."""
    model = models.vam(level=1, dimension=2, bc="bump")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    set_state_width(triple[0])

    mesh = LSQMesh.create_1d(domain=ESC_DOMAIN, n_inner_cells=ESC_NCELLS)
    t0 = time.perf_counter()
    Q, Qaux = chorin_march(triple, mesh, cfl=CFL, ic=bump_ic, n_steps=40,
                           h_scale=ESC_H_RES, **FIRST_ORDER)
    elapsed = time.perf_counter() - t0

    names = [str(s) for s in triple[0].state]
    h = Q[names.index("h"), :]
    P0 = Q[names.index("P_0"), :]
    P1 = Q[names.index("P_1"), :]
    pb = _bottom_pressure_head(model, h, P0, P1, g=9.81)

    assert np.all(np.isfinite(h)) and np.all(np.isfinite(pb))
    refs.check("vam_bump_curves", overwrite,
               h=h, pb=pb, Q=Q, Qaux=Qaux)
    refs.check_time("vam_bump_curves", elapsed, overwrite)
