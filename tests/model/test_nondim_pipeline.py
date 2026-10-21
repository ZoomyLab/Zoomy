"""The non-dimensional scaling pipeline (`derivation/nondim.py`, `models/scaling.py`): the
Buckingham-Pi check, the closure consistency check, the order-based truncation, and the moment
models' default truncations, which it now derives (ANALYSIS 49, 63):

* SME, ML-SME (`Stress.shallow`): thesis regime `Fr, De = O(1)`, orders (horizontal 0,
  vertical 0) -- byte-identical to the former hand rules (goldens m01-m06, m08, m10).
* ML-SME `Stress.full`: `De = O(1)`, orders (2, 2) (= Marche 2007 eq. (5.1) `De = O(eps)`,
  orders (1, 1)).  SME `Stress.full` keeps the hand rule, the consistent truncation at
  `De = O(eps)` (orders (1, 1)), not at `De = O(1)`.
* VAM: `Fr, De = O(1)`, orders (0, 2), `Stress.full` (2, 2): the vertical row keeps
  `-nu d_x d_z u`.  `VAMWeakViscosity`: `1/De = O(eps)`, `k_l = O(eps^2)`, orders (1, 2), the
  thesis VAM (goldens m11-m13, S02, N02, P01 unchanged).

Design: docs/dev/symbolic_solver/research/nondimensional_scaling.md; ANALYSIS 49, 63.
"""
import contextlib
import warnings

import pytest
import sympy as sp

import goldenlib

pytestmark = pytest.mark.model


def _sm(factory, unclosed=False):
    from zoomy_core.systemmodel.system_model import SystemModel, allow_unclosed
    with goldenlib.no_cache():
        with (allow_unclosed() if unclosed else contextlib.nullcontext()):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                return goldenlib.render_system_model(SystemModel.from_model(factory()))


def _orders(h, v, **regime):
    from zoomy_core.model.derivation.nondim import ShallowScaling, Regime
    return ShallowScaling(horizontal=h, vertical=v, regime=Regime(**regime) if regime else None)


def _cases():
    """`{name: (model, the same model with an explicit scaling)}`: the defaults are the stated
    regimes, and the alternative regimes that give the same truncation."""
    from zoomy_core.model.models import SME, VAM, VAMWeakViscosity
    from zoomy_core.model.models.ml_sme import MLSME
    from zoomy_core.model.models.equations import Stress
    from zoomy_core.model.models.closures import Newtonian, ManningFriction, StressFree
    std = goldenlib._std_closures
    swe = lambda: [Newtonian(), ManningFriction(), StressFree()]
    weak = dict(De=-1, k_l=2, C_n=1)
    small = {
        "sme1_shallow": (lambda: SME(level=1, closures=std()),
                         lambda: SME(level=1, closures=std(), scaling=_orders(0, 0))),
        "swe1d_manning": (lambda: SME(level=0, closures=swe()),
                          lambda: SME(level=0, closures=swe(), scaling=_orders(0, 0))),
        "sme1_full_marche": (lambda: SME(level=1, stress=Stress.full, closures=std()),
                             lambda: SME(level=1, closures=std(), scaling=_orders(1, 1, De=1))),
        "vam1": (lambda: VAM(level=1, closures=std()),
                 lambda: VAM(level=1, closures=std(), scaling=_orders(0, 2))),
        "vam1_weak": (lambda: VAMWeakViscosity(level=1, closures=std()),
                      lambda: VAM(level=1, closures=std(), scaling=_orders(1, 2, **weak))),
        "vam1_full": (lambda: VAM(level=1, stress=Stress.full, closures=std()),
                      lambda: VAM(level=1, closures=std(), scaling=_orders(1, 2, De=1))),
        "mlsme_m10": (lambda: MLSME(n_layers=2, level=1, closures=std()),
                      lambda: MLSME(n_layers=2, level=1, closures=std(), scaling=_orders(0, 0))),
    }
    large = {
        "sme2_full_marche": (lambda: SME(level=2, stress=Stress.full, closures=std()),
                             lambda: SME(level=2, closures=std(), scaling=_orders(1, 1, De=1))),
        "sme1_full_re1": (lambda: SME(level=1, stress=Stress.full, closures=std()),
                          lambda: SME(level=1, closures=std(), scaling=_orders(0, 0, De=2))),
        "vam2_weak": (lambda: VAMWeakViscosity(level=2, closures=std()),
                      lambda: VAM(level=2, closures=std(), scaling=_orders(1, 2, **weak))),
        "vam1_2d_weak": (lambda: VAMWeakViscosity(level=1, dimension=3, closures=std()),
                         lambda: VAM(level=1, dimension=3, closures=std(),
                                     scaling=_orders(1, 2, **weak))),
        "vam2_full": (lambda: VAM(level=2, stress=Stress.full, closures=std()),
                      lambda: VAM(level=2, closures=std(), scaling=_orders(1, 2, De=1))),
        "mlsme_l1_full": (lambda: MLSME(n_layers=2, level=1, stress=Stress.full,
                                        closures=std()),
                          lambda: MLSME(n_layers=2, level=1, closures=std(),
                                        scaling=_orders(1, 1, De=1))),
    }
    return small, large


_SMALL, _LARGE = (sorted(c) for c in _cases())


@pytest.mark.parametrize("name", _SMALL)
def test_default_is_the_stated_regime(name):
    """The model's default truncation renders byte-identically to the stated regime."""
    default, explicit = _cases()[0][name]
    assert _sm(default) == _sm(explicit)


@pytest.mark.large
@pytest.mark.parametrize("name", _LARGE)
def test_default_is_the_stated_regime_large(name):
    """As `test_default_is_the_stated_regime`, the larger / 2-D derivations."""
    default, explicit = _cases()[1][name]
    assert _sm(default) == _sm(explicit)


def test_vam_minus_weak_viscosity():
    """VAM(1) at De = O(1) differs from VAMWeakViscosity(1) by the vertical row's
    `-nu d_x d_z u` only: B[r_0, h] = 4 nu q_1 / h^2, B[r_0, q_1] = -2 nu / h."""
    from zoomy_core.model.models import VAM, VAMWeakViscosity
    std = goldenlib._std_closures
    a = _sm(lambda: VAM(level=1, closures=std())).splitlines()
    b = _sm(lambda: VAMWeakViscosity(level=1, closures=std())).splitlines()
    assert sorted(set(b) - set(a)) == []
    assert sorted(set(a) - set(b)) == [
        "nonconservative_matrix[4,1,0] = Mul(Integer(4), Pow(Symbol('h'), Integer(-2)), "
        "Symbol('nu'), Symbol('q_1'))",
        "nonconservative_matrix[4,3,0] = Mul(Integer(-1), Integer(2), Pow(Symbol('h'), "
        "Integer(-1)), Symbol('nu'))"]


def test_shear_only_closure_needs_the_weak_regime():
    """A bulk closure that gives only the vertical shear (Elder) cannot close the vertical
    row's stress that VAM keeps at De = O(1): refused, naming VAMWeakViscosity, which takes it
    (as SME does); a bed closure without scaling groups (RoughWall) is recorded unchecked."""
    from zoomy_core.model.derivation.nondim import ScalingError
    from zoomy_core.model.models import VAM, VAMWeakViscosity
    from zoomy_core.model.models.closures import ElderViscosity, RoughWall, StressFree
    cl = lambda: [ElderViscosity(friction=RoughWall()), RoughWall(), StressFree()]
    with pytest.raises(ScalingError, match="VAMWeakViscosity"):
        VAM(level=1, closures=cl()).derivation
    m = VAMWeakViscosity(level=1, closures=cl()).derivation
    assert "unchecked (no scaling groups): RoughWall" in str(m.history)


def test_weak_viscosity_refuses_full_stress():
    from zoomy_core.model.models import VAMWeakViscosity
    from zoomy_core.model.models.equations import Stress
    with pytest.raises(ValueError, match="Stress.full"):
        VAMWeakViscosity(level=1, stress=Stress.full,
                         closures=goldenlib._std_closures()).derivation


def test_pi_check_names_the_problem():
    """Incomplete, dependent and dimensional group sets are refused with a message naming the
    missing group / the offending one."""
    from zoomy_core.model.derivation.nondim import (
        PiBasis, DimensionError, LENGTH, VELOCITY, STRESS, ACCELERATION, DENSITY)
    L, H, U, S, g, rho = sp.symbols("L H U S g rho", positive=True)
    eps, Fr, De, X = sp.symbols("epsilon Fr De X", positive=True)
    q = {L: LENGTH, H: LENGTH, U: VELOCITY, S: STRESS, g: ACCELERATION, rho: DENSITY}
    ok = PiBasis(q, {eps: H / L, Fr: U / sp.sqrt(g * H), De: eps * rho * U ** 2 / S},
                 prefer=(L, H, U, S))
    assert ok.express[g] == U ** 2 / (Fr ** 2 * L * eps)
    with pytest.raises(DimensionError, match="missing e.g.") as err:
        PiBasis(q, {eps: H / L, Fr: U / sp.sqrt(g * H)})
    assert "S" in str(err.value) and "rho" in str(err.value)
    with pytest.raises(DimensionError, match="depends on the others"):
        PiBasis(q, {eps: H / L, Fr: U / sp.sqrt(g * H), X: U ** 2 / (g * H)})
    with pytest.raises(DimensionError, match="not dimensionless"):
        PiBasis(q, {eps: H / L, Fr: U / sp.sqrt(g * H), De: rho * U / S})


def _sme_blueprint():
    from zoomy_core import coords as C
    import zoomy_core.derivatives as d
    from zoomy_core.model.derivation import Model as DModel
    from zoomy_core.model.models.equations import Mass, Momentum
    t, x, z = C.t, C.x, C.z
    m = DModel(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0, "nu": 0.0,
                                             "lambda_s": 0.0, "e_x": 0.0})
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    m.declare_state(h)
    m.add_equation(Mass(m))
    mom = Momentum(m)
    m.add_equation(mom)
    return m, mom, h, b


def test_closure_larger_than_its_ansatz_is_an_error():
    """An in-plane ansatz `tau_xx ~ eps^2 S` makes the Newtonian `tau_hat_xx = 2 d_x u / eps`:
    InsertClosure refuses it, naming the component."""
    from zoomy_core.model.derivation.nondim import (
        ShallowScaling, InsertClosure, ScalingError)
    from zoomy_core.model.models.scaling import FreeSurfaceNonDim, EPS, S
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    m, mom, h, b = _sme_blueprint()

    class Tight(FreeSurfaceNonDim):
        def _spec(self, m_):
            out = super()._spec(m_)
            self.scales[mom.tau["xx"]] = EPS ** 2 * S
            return out
    m.apply(Tight(mom, heights=[h, b], free_surface=b + h, mass="mass"))
    m.apply(ShallowScaling(horizontal=2, vertical=0))
    with pytest.raises(ScalingError, match="tau_xx is O\\(eps\\^-1\\)"):
        m.apply(InsertClosure([Newtonian(), NavierSlip(), StressFree()]))


def test_vertical_nu_dxx_w_dropped_by_order():
    """With the full Newtonian stress the vertical row's `nu d_x d_x w` is eps^2 below
    `nu d_x d_z u` (Marche 2007 (3.2) -> (3.5)): the truncation after the closure drops exactly
    that term, by its order (no hand rule)."""
    from zoomy_core.model.derivation.nondim import (
        ShallowScaling, InsertClosure, Regime, Dimensionalize)
    from zoomy_core.model.models.scaling import FreeSurfaceNonDim
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    from zoomy_core import coords as C
    m, mom, h, b = _sme_blueprint()
    m.apply(FreeSurfaceNonDim(mom, heights=[h, b], free_surface=b + h, mass="mass"))
    m.apply(ShallowScaling(horizontal=1, vertical=1, regime=Regime(De=1)))
    m.apply(InsertClosure([Newtonian(), NavierSlip(), StressFree()]))
    st = m.nondim
    m.apply(ShallowScaling())
    (dropped,) = st.last_dropped["momentum_z"]
    hat_w = st.fields[mom.w.func].hat(*mom.w.args)
    assert dropped.has(sp.Derivative(hat_w, (C.x, 2)))
    assert st.order(dropped) == 3                          # eps^4 / De, De = O(eps)
    m.apply(Dimensionalize())
    nu = m.parameters.nu
    assert m.momentum_z.expr.has(nu * sp.Derivative(mom.uvel[0], C.x, C.z))
    assert not m.momentum_z.expr.has(sp.Derivative(mom.w, (C.x, 2)))


def test_inconsistent_regime_is_refused():
    """Weak viscosity (`1/De = O(eps)`) with Marche's O(eps) slip makes the bed traction
    O(eps^-1) against the shear scale (slip length << H, a bed boundary layer): refused, naming
    NavierSlip; with `k_l = O(eps^2)` it is consistent."""
    from zoomy_core.model.derivation.nondim import ScalingError
    from zoomy_core.model.models import VAM
    with pytest.raises(ScalingError, match="NavierSlip"):
        VAM(level=0, closures=goldenlib._std_closures(),
            scaling=_orders(1, 2, De=-1)).derivation


def test_hydrostatic_sme_refuses_vertical_inertia():
    """SME at De = O(1) and vertical order 2 keeps the eps^2 vertical inertia, which a
    hydrostatic model cannot take."""
    from zoomy_core.model.models import SME
    with pytest.raises(ValueError, match="keeps the vertical inertia"):
        SME(level=0, closures=goldenlib._std_closures(), scaling=_orders(2, 2)).derivation
