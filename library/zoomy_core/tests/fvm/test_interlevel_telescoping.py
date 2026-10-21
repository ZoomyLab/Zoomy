"""THE INTER-LEVEL TELESCOPING IDENTITY.

Pure algebra on ONE interface face pair: no mesh, no march, no solver, no
preCICE.  The identity is a property of the face operator alone, so proving it
needs nothing else — and a unit test that needs nothing else is the only form
of this proof that stays green when the coupling harness moves.

THE PAIRING.  Each arm applies to its OWN cell the contribution
``dm = F + D_minus``, evaluated with its own state as the minus slot, its
peer's as the plus slot, and its own OUTWARD normal.  The amount the pair fails
to conserve on a row is the sum of the two arms' contributions on that row —
the RESIDUAL measured below.

TWO BEHAVIOURS.

*CURRENT* — each arm runs ITS OWN operator in ITS OWN space: the low arm the
SME(n_L) operator on the peer state truncated to its rows, the high arm the
SME(n_H) operator on the zero-padded peer.  Two DIFFERENT Riemann problems, so
the halves do not pair.

*FIX* — BOTH arms are read off ONE evaluation of the HIGH operator at the
ZERO-PADDED pair, at a canonical orientation, with the two sides taken as sign
projections of that single answer::

    low arm    ( F + D_minus )[:n_L]
    high arm   ( -F + D_plus )

ONE evaluation, two projections — never two evaluations of "the same" pair with
each arm's own normal.  ``nHat`` is normalised as ``n/(|n| + SMALL)``, so an
x-normal face carries ``nHat.x() = 1 - 2.2e-16`` and the two arms'
algebraically-equal expressions differ in the last bit (measured to break
bit-equality on ~62% of states, ``swePreciceCoupling.C``).

WHAT THE FIX DOES AND DOES NOT REPAIR.
  * ``b``   — stationary; zero either way (the control).
  * ``h``   — conservative; NONZERO under CURRENT, machine zero under FIX.  The
              current pairing leaks MASS, not just momentum.
  * ``q_0`` — the row this construction exists to repair: machine zero under
              FIX, **on a flat bed only**.
  * ``q_1`` — NOT repaired, and MUST NOT BE.  Its nonconservative row is live,
              so ``(D_plus + D_minus)[q_1] = (A_int·ΔQ)[q_1] != 0`` is the
              correct path-conservative NCP floor.  Pinned nonzero so nobody
              later "fixes" it and silently deletes a term of the PDE.

SCOPE, PINNED BY TEST.  ``q_0`` telescoping is FLAT-BED CONDITIONAL: its
nonconservative row carries the live bed column ``B[q_0][b] = g·h``, leaving
``-B[q_0][b]·Δb``.  ``test_sloped_bed_breaks_q0_telescoping`` shows it does NOT
telescope across a bed step, so the limit cannot be over-claimed later.

NO NEW MACHINERY.  This calls only ``_compute_flux`` / ``_compute_fluctuations``
— the builders the registered kernels already use — at a padded pair.  The
zero-padding is a pure CALLER-side action: no new registered kernel, no printer
change, no new SystemModel API.
"""
import numpy as np
import pytest

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.models.sme import SME
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel

pytestmark = [pytest.mark.small, pytest.mark.unittest]

LOW_LEVEL, HIGH_LEVEL = 1, 2
#: Physical parameters, by symbol name (the emitted ``p`` slot order is the
#: model's own, so build the vector by name rather than by position).
PAR = {"g": 9.81, "rho": 1000.0, "nu": 1e-3, "lambda_s": 0.5, "e_x": 0.0}
#: Machine-zero means EXACTLY zero here — the whole claim is bit-level.
EXACT = 0.0
#: "Clearly nonzero", well above any rounding of these O(1e-2) quantities.
LIVE = 1e-9


def _numerics(level):
    """The house NSM path: model -> NumericalSystemModel -> built numerics."""
    nsm = NumericalSystemModel.from_model(
        SME(level=level, dimension=2,
            closures=[Newtonian(), NavierSlip(), StressFree()]),
        riemann=PositiveNonconservativeRusanov).derive()
    return nsm.build_numerics()


_CACHE = {}


def numerics(level):
    if level not in _CACHE:
        _CACHE[level] = (n := _numerics(level)), n.to_runtime_numpy()
    return _CACHE[level]


def _aux(num, q):
    """Aux vector for a state: every derivative slot zero, ``hinv`` = 1/h.

    ``hinv`` is the ONLY aux the face kernels read; the derivative slots feed
    the order-2 reconstruction, which a single-face algebraic check does not
    exercise.  Desingularised the same way the model does, so a near-dry state
    stays finite instead of turning the whole residual into NaN.
    """
    names = [str(s) for s in num.model.aux_state]
    a = np.zeros(len(names))
    h = float(q[1])
    if "hinv" in names:
        a[names.index("hinv")] = 1.0 / h if h > 1e-12 else 0.0
    return a


def _params(num):
    return np.array([PAR[str(s)] for s in num.model.parameters.keys()],
                    dtype=float)


def dm(level, q_own, q_peer, normal_sign):
    """``F + D_minus`` — the face contribution this arm applies to its OWN
    cell, with its own state in the minus slot and its own outward normal."""
    num, rt = numerics(level)
    n_dim = num.model.n_dim
    nrm = np.zeros(n_dim)
    nrm[0] = float(normal_sign)
    q_own = np.asarray(q_own, dtype=float)
    q_peer = np.asarray(q_peer, dtype=float)
    F = np.asarray(rt.numerical_flux(
        q_own, q_peer, _aux(num, q_own), _aux(num, q_peer),
        _params(num), nrm)).ravel()
    fl = np.asarray(rt.numerical_fluctuations(
        q_own, q_peer, _aux(num, q_own), _aux(num, q_peer),
        _params(num), nrm))
    return F + np.asarray(fl[1]).ravel()


def face(level, q_minus, q_plus, normal_sign):
    """``(F, D_plus, D_minus)`` from ONE evaluation of the ``level`` operator."""
    num, rt = numerics(level)
    nrm = np.zeros(num.model.n_dim)
    nrm[0] = float(normal_sign)
    q_minus = np.asarray(q_minus, dtype=float)
    q_plus = np.asarray(q_plus, dtype=float)
    args = (q_minus, q_plus, _aux(num, q_minus), _aux(num, q_plus),
            _params(num), nrm)
    F = np.asarray(rt.numerical_flux(*args)).ravel()
    fl = np.asarray(rt.numerical_fluctuations(*args))
    return F, np.asarray(fl[0]).ravel(), np.asarray(fl[1]).ravel()


def pad(q_low, n_high):
    q_low = np.asarray(q_low, dtype=float)
    return np.concatenate([q_low, np.zeros(n_high - q_low.size)])


# ── the state sweep ────────────────────────────────────────────────────────
#
# (label, h_low, h_high, q0_low, q0_high, b_low, b_high)
# Bed equal on both sides EXCEPT the explicitly sloped entries.
SWEEP = [
    ("equal depths",        0.50, 0.50,  0.030,  0.030, 0.0, 0.0),
    ("mild ratio 1:1.8",    0.10, 0.18,  0.030,  0.055, 0.0, 0.0),
    ("strong ratio 6:1",    1.80, 0.30,  0.240,  0.020, 0.0, 0.0),
    ("strong ratio 1:6",    0.30, 1.80,  0.020,  0.240, 0.0, 0.0),
    ("deep pair",           5.00, 4.20,  1.100,  0.900, 0.0, 0.0),
    ("near-dry left",       1e-6, 0.50,  1e-8,   0.040, 0.0, 0.0),
    ("near-dry right",      0.50, 1e-6,  0.040,  1e-8,  0.0, 0.0),
    ("near-dry both",       1e-6, 1e-6,  1e-9,   1e-9,  0.0, 0.0),
    ("flow reversal",       0.60, 0.55,  0.300, -0.400, 0.0, 0.0),
    ("reversal, deep/shal", 1.20, 0.25, -0.500,  0.350, 0.0, 0.0),
    ("stagnant",            0.80, 0.80,  0.000,  0.000, 0.0, 0.0),
]

ORDERINGS = [("low-left/high-right", +1.0), ("high-left/low-right", -1.0)]


def states(h_lo, h_hi, q0_lo, q0_hi, b_lo, b_hi):
    """``(q_low, q_high)`` — level-1 ``[b,h,q0,q1]`` and level-2
    ``[b,h,q0,q1,q2]``.  Higher moments are a fixed small fraction of q0, so
    the pair is a physically plausible profile rather than noise."""
    q_low = np.array([b_lo, h_lo, q0_lo, 0.13 * q0_lo])
    q_high = np.array([b_hi, h_hi, q0_hi, 0.13 * q0_hi, 0.045 * q0_hi])
    return q_low, q_high


def residual_current(q_low, q_high, n_low, n_high, sgn):
    """Each arm its OWN operator in its OWN space."""
    low = dm(LOW_LEVEL, q_low, q_high[:n_low], sgn)
    high = dm(HIGH_LEVEL, q_high, pad(q_low, n_high), -sgn)
    return low + high[:n_low]


def residual_fix(q_low, q_high, n_low, n_high, sgn):
    """ONE evaluation of the HIGH operator at the padded pair, both arms taken
    as sign projections of it."""
    F, Dp, Dm = face(HIGH_LEVEL, pad(q_low, n_high), q_high, sgn)
    low = (F + Dm)[:n_low]
    high = (-F + Dp)[:n_low]
    return low + high


@pytest.fixture(scope="module")
def dims():
    n_low = numerics(LOW_LEVEL)[0].n_variables
    n_high = numerics(HIGH_LEVEL)[0].n_variables
    assert n_high == n_low + 1, f"expected one extra row, got {n_low}->{n_high}"
    return n_low, n_high


ROWS = ["b", "h", "q_0", "q_1"]


@pytest.mark.parametrize("order_label,sgn", ORDERINGS)
@pytest.mark.parametrize("case", SWEEP, ids=[c[0] for c in SWEEP])
def test_fix_telescopes_h_and_q0_exactly(case, order_label, sgn, dims, capsys):
    """THE PROOF, over the sweep: ``h`` and ``q_0`` are EXACTLY zero under the
    fix, for every state and both orderings; ``q_1`` is not driven to zero."""
    n_low, n_high = dims
    label = case[0]
    q_low, q_high = states(*case[1:])
    cur = residual_current(q_low, q_high, n_low, n_high, sgn)
    fix = residual_fix(q_low, q_high, n_low, n_high, sgn)
    with capsys.disabled():
        print(f"\n  {label:22s} {order_label:20s} "
              + "  ".join(f"{r}: {cur[i]:+.3e}->{fix[i]:+.3e}"
                          for i, r in enumerate(ROWS)))

    assert np.isfinite(fix).all(), f"{label}: non-finite residual {fix}"
    for row in ("b", "h", "q_0"):
        i = ROWS.index(row)
        assert abs(fix[i]) == EXACT, (
            f"{label} [{order_label}]: {row} residual {fix[i]:.3e} is not "
            "machine zero under the padded high-operator pairing")
    i1 = ROWS.index("q_1")
    if abs(cur[i1]) > LIVE:          # a stagnant/dry pair has no NCP to carry
        assert abs(fix[i1]) > LIVE, (
            f"{label} [{order_label}]: q_1 was driven to {fix[i1]:.3e}. The "
            "construction must NOT make a nonconservative row conserve — that "
            "would delete a term of the PDE")


@pytest.mark.parametrize("order_label,sgn", ORDERINGS)
def test_current_pairing_misses_on_h_and_q0(order_label, sgn, dims):
    """THE DEFECT, reproduced: solving the face in two different spaces leaves
    a nonzero residual on rows that MUST conserve.  If this ever goes to zero
    the two spaces already agree and the fix is moot."""
    n_low, n_high = dims
    q_low, q_high = states(*SWEEP[1][1:])          # mild ratio, wet, flat bed
    cur = residual_current(q_low, q_high, n_low, n_high, sgn)
    print(f"\n[current {order_label}] h {cur[1]:+.6e}  q_0 {cur[2]:+.6e}")
    assert abs(cur[ROWS.index("h")]) > LIVE, (
        "the depth row was expected to MISS under the current pairing — the "
        "mass ledger does not close when each arm uses its own operator")
    assert abs(cur[ROWS.index("q_0")]) > LIVE, (
        "the discharge row was expected to MISS under the current pairing")


@pytest.mark.parametrize("order_label,sgn", ORDERINGS)
def test_stationary_bed_row_is_zero_in_both(order_label, sgn, dims):
    """``b`` is stationary — zero either way.  The control that shows the
    results above are not simply 'every row is zero'."""
    n_low, n_high = dims
    q_low, q_high = states(*SWEEP[1][1:])
    for label, res in (
            ("current", residual_current(q_low, q_high, n_low, n_high, sgn)),
            ("fix", residual_fix(q_low, q_high, n_low, n_high, sgn))):
        assert abs(res[ROWS.index("b")]) == EXACT, f"{label}: b != 0"


@pytest.mark.parametrize("db", [0.05, 0.40, -0.30])
def test_sloped_bed_breaks_q0_telescoping(db, dims):
    """THE DOCUMENTED LIMIT, PINNED.

    ``q_0`` telescoping is FLAT-BED CONDITIONAL.  Its nonconservative row
    carries the live bed column ``B[q_0][b] = g·h``, so across a bed step the
    pair leaves ``-B[q_0][b]·Δb != 0``.  ``h`` is unaffected — it is
    conservative regardless of the bed — which is exactly why the mass ledger
    survives a slope and the momentum ledger does not.

    Nobody may later restate this operator as 'conservative' without qualifying
    it: this test fails the moment that claim is made without the flat-bed
    hypothesis.
    """
    n_low, n_high = dims
    q_low, q_high = states(0.60, 0.55, 0.30, 0.26, 0.0, db)
    fix = residual_fix(q_low, q_high, n_low, n_high, +1.0)
    print(f"\n[sloped db={db:+.2f}] h {fix[1]:+.6e}  q_0 {fix[2]:+.6e}")
    assert abs(fix[ROWS.index("h")]) == EXACT, (
        f"depth must telescope across a bed step too, got {fix[1]:.3e}")
    assert abs(fix[ROWS.index("q_0")]) > LIVE, (
        f"q_0 residual {fix[2]:.3e} vanished across a bed step of {db}. Either "
        "the bed column B[q_0][b] has been dropped, or the flat-bed scope "
        "condition is being over-claimed — both are defects")


def test_one_evaluation_matches_the_two_evaluation_form(dims):
    """The single-evaluation projection agrees with the naive two-evaluation
    pairing to within rounding — and is the form to USE, because only it is
    bit-exact on both arms (the last-bit normal trap)."""
    n_low, n_high = dims
    q_low, q_high = states(*SWEEP[1][1:])
    one = residual_fix(q_low, q_high, n_low, n_high, +1.0)
    two = (dm(HIGH_LEVEL, pad(q_low, n_high), q_high, +1.0)[:n_low]
           + dm(HIGH_LEVEL, q_high, pad(q_low, n_high), -1.0)[:n_low])
    assert np.allclose(one, two, rtol=1e-9, atol=1e-12), (
        f"single-evaluation {one} vs two-evaluation {two}")
