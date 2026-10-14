"""The non-dimensional scaling pipeline (`derivation/nondim.py`, `models/scaling.py`): the
Buckingham-Pi check, the closure consistency check, the order-based truncation, and the pin that
the moment models derived through it (`scaling=`) equal the hand-truncated ones byte for byte
(rendered SystemModels), each in the regime whose orders the hand rules correspond to:

* SME, ML-SME (`Stress.shallow`): thesis regime `Fr, De = O(1)`, orders (horizontal 0,
  vertical 0).
* SME, ML-SME `Stress.full`: Marche 2007 eq. (5.1) `1/Re = eps nu_0`, i.e. `De = O(eps)`,
  orders (1, 1) (also `De = O(eps^2)`, orders (0, 0): Marche (3.2) -> (3.5) with `Re = O(1)`).
* VAM (`Stress.shallow`): `1/De = O(eps)` with the slip `k_l = O(eps^2)`, orders (1, 2).
* VAM `Stress.full`: `De = O(1)`, orders (2, 2).

Design: docs/dev/symbolic_solver/research/nondimensional_scaling.md; ANALYSIS 49.
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
    from zoomy_core.model.models import SME, VAM
    from zoomy_core.model.models.ml_sme import MLSME
    from zoomy_core.model.models.equations import Stress
    from zoomy_core.model.models.closures import Newtonian, ManningFriction, StressFree
    std = goldenlib._std_closures
    swe = lambda: [Newtonian(), ManningFriction(), StressFree()]
    small = {
        "sme1_shallow": (lambda: SME(level=1, closures=std()),
                         lambda: SME(level=1, closures=std(), scaling=_orders(0, 0))),
        "swe1d_manning": (lambda: SME(level=0, closures=swe()),
                          lambda: SME(level=0, closures=swe(), scaling=_orders(0, 0))),
        "sme1_full_marche": (lambda: SME(level=1, stress=Stress.full, closures=std()),
                             lambda: SME(level=1, closures=std(), scaling=_orders(1, 1, De=1))),
        "vam1_shallow": (lambda: VAM(level=1, closures=std()),
                         lambda: VAM(level=1, closures=std(),
                                     scaling=_orders(1, 2, De=-1, k_l=2))),
        "vam1_full": (lambda: VAM(level=1, stress=Stress.full, closures=std()),
                      lambda: VAM(level=1, closures=std(), scaling=_orders(2, 2))),
        "mlsme_m10": (lambda: MLSME(n_layers=2, level=1, closures=std()),
                      lambda: MLSME(n_layers=2, level=1, closures=std(), scaling=_orders(0, 0))),
    }
    large = {
        "sme0_shallow": (lambda: SME(level=0, closures=std()),
                         lambda: SME(level=0, closures=std(), scaling=_orders(0, 0))),
        "sme2_shallow": (lambda: SME(level=2, closures=std()),
                         lambda: SME(level=2, closures=std(), scaling=_orders(0, 0))),
        "sme1_2d_shallow": (lambda: SME(level=1, dimension=3, closures=std()),
                            lambda: SME(level=1, dimension=3, closures=std(),
                                        scaling=_orders(0, 0))),
        "swe2d_manning": (lambda: SME(level=0, dimension=3, closures=swe()),
                          lambda: SME(level=0, dimension=3, closures=swe(),
                                      scaling=_orders(0, 0))),
        "sme2_full_marche": (lambda: SME(level=2, stress=Stress.full, closures=std()),
                             lambda: SME(level=2, closures=std(), scaling=_orders(1, 1, De=1))),
        "sme1_full_re1": (lambda: SME(level=1, stress=Stress.full, closures=std()),
                          lambda: SME(level=1, closures=std(), scaling=_orders(0, 0, De=2))),
        "vam2_shallow": (lambda: VAM(level=2, closures=std()),
                         lambda: VAM(level=2, closures=std(),
                                     scaling=_orders(1, 2, De=-1, k_l=2))),
        "vam1_2d_shallow": (lambda: VAM(level=1, dimension=3, closures=std()),
                            lambda: VAM(level=1, dimension=3, closures=std(),
                                        scaling=_orders(1, 2, De=-1, k_l=2))),
        "vam2_full": (lambda: VAM(level=2, stress=Stress.full, closures=std()),
                      lambda: VAM(level=2, closures=std(), scaling=_orders(2, 2))),
        "mlsme_l1_full_marche": (lambda: MLSME(n_layers=2, level=1, stress=Stress.full,
                                               closures=std()),
                                 lambda: MLSME(n_layers=2, level=1, closures=std(),
                                               scaling=_orders(1, 1, De=1))),
    }
    return small, large


_SMALL, _LARGE = (sorted(c) for c in _cases())


@pytest.mark.parametrize("name", _SMALL)
def test_pipeline_equals_hand_rules(name):
    """The model derived through the pipeline renders byte-identically to the hand rules."""
    hand, pipe = _cases()[0][name]
    assert _sm(pipe) == _sm(hand)


@pytest.mark.large
@pytest.mark.parametrize("name", _LARGE)
def test_pipeline_equals_hand_rules_large(name):
    """As `test_pipeline_equals_hand_rules`, the larger / 2-D derivations."""
    hand, pipe = _cases()[1][name]
    assert _sm(pipe) == _sm(hand)


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
