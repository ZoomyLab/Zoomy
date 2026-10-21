"""ML-VAM `vertical_traction` (ANALYSIS 45j): with `VerticalTraction.shared` the w rows' interface
trace of tau_ez is one value of both layers' traces, by the stress `InterfaceFlux` (mean by
default), instead of each layer's own (`one_sided`, the pointwise projection); `shared` is the
default (ANALYSIS 63); `VerticalTraction.function` takes a rule `(below, above) -> value`.

The residuals differ by sgn phi_k(side) (f* - f_own) d_x z_1 / (rho mu_k) at the internal
interface z_1 = b + l_1 h, f = rho nu d_z u of the layer's modal profile, so the mode-0 layer sum
of the shared w rows has no interface term (it telescopes)."""
import pytest
import sympy as sp

from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.model.models.equations import Stress
from zoomy_core.model.models.ml_vam import MLVAM, VerticalTraction
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.model, pytest.mark.small]

CLOSURES = [Newtonian(), NavierSlip(), StressFree()]


def test_traction_choice_needs_the_full_stress():
    """With Stress.shallow the w rows carry no tau_ez: `shared` (the default) and `one_sided`
    give the same model; a trace function is refused."""
    a, b = (SystemModel.from_model(MLVAM(n_layers=2, level=1, dimension=2, closures=CLOSURES,
                                         vertical_traction=vt)).reconstruct_residuals()
            for vt in (VerticalTraction.shared, VerticalTraction.one_sided))
    assert all(sp.expand(x - y) == 0 for x, y in zip(a, b))
    with pytest.raises(ValueError, match="Stress.full"):
        MLVAM(n_layers=2, level=1, dimension=2, closures=CLOSURES,
              vertical_traction=VerticalTraction.function,
              vertical_traction_function=lambda lo, up: (lo + up) / 2).derive_model()
    with pytest.raises(ValueError, match="go together"):
        MLVAM(n_layers=2, level=1, dimension=2, closures=CLOSURES,
              vertical_traction=VerticalTraction.function).derive_model()


def test_function_trace_is_the_users_rule():
    """`VerticalTraction.function` with the mean is `shared` (the mean InterfaceFlux), and a
    weighted rule changes only the w rows."""
    mk = lambda **kw: SystemModel.from_model(MLVAM(n_layers=2, level=1, dimension=2,
                                                   closures=CLOSURES, stress=Stress.full, **kw))
    sms = {"mean": mk(vertical_traction=VerticalTraction.function,
                      vertical_traction_function=lambda lo, up: (lo + up) / 2),
           "shared": mk(),
           "lower": mk(vertical_traction=VerticalTraction.function,
                       vertical_traction_function=lambda lo, up: lo)}
    names = [str(q) for q in sms["shared"].state]
    res = {k: dict(zip(names, v.reconstruct_residuals())) for k, v in sms.items()}
    changed = set()
    for n in names:
        assert sp.simplify(sp.expand((res["mean"][n] - res["shared"][n]).doit())) == 0, n
        if sp.simplify(sp.expand((res["lower"][n] - res["shared"][n]).doit())) != 0:
            changed.add(n)
    assert changed and all(n.startswith("r_") for n in changed), changed


def test_shared_trace_replaces_the_own_trace():
    lvl = 1
    sms = {vt: SystemModel.from_model(MLVAM(n_layers=2, level=lvl, dimension=2, closures=CLOSURES,
                                            stress=Stress.full, vertical_traction=vt))
           for vt in (VerticalTraction.one_sided, VerticalTraction.shared)}
    sm = sms[VerticalTraction.shared]
    assert not sm.unclassified_terms
    names = [str(s) for s in sm.state]
    res = {vt: dict(zip(names, s.reconstruct_residuals())) for vt, s in sms.items()}
    t, x = sm.time, sm.space[0]
    F = {n: sp.Function(n, real=True)(t, x) for n in names}
    par = sm.parameters
    rho, nu, l1 = par.rho, par.nu, par.l_1
    L = Legendre_shifted(level=lvl + 2)
    zeta = sp.Symbol("zeta")
    phi = [L.eval(j, zeta) for j in range(lvl + 1)]
    h = F["h"]

    def shear(layer, frac, side):           # rho nu d_z u of the layer's profile at `side`
        return rho * nu / (frac * h) * sum(F[f"q_{layer}_{j}"] / (frac * h)
                                           * sp.diff(phi[j], zeta).subs(zeta, side)
                                           for j in range(lvl + 1))
    f_below, f_above = shear(1, l1, 1), shear(2, 1 - l1, 0)
    f_star = (f_below + f_above) / 2
    dz1 = sp.diff(F["b"] + l1 * h, x)
    for k in range(lvl + 1):
        for layer, side, sgn, own in ((1, 1, 1, f_below), (2, 0, -1, f_above)):
            r = f"r_{layer}_{k}"
            expected = sgn * phi[k].subs(zeta, side) * (f_star - own) * dz1 * (2 * k + 1) / rho
            got = res[VerticalTraction.shared][r] - res[VerticalTraction.one_sided][r]
            assert sp.simplify(sp.expand((got - expected).doit())) == 0, r
    for r in names:                         # the horizontal rows are unchanged
        if r.startswith("q_"):
            d = res[VerticalTraction.shared][r] - res[VerticalTraction.one_sided][r]
            assert sp.simplify(sp.expand(d.doit())) == 0, r
