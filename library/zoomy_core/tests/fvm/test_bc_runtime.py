"""E5 — runtime boundary-condition semantics on the numpy solvers
(spec §1b/§2 rank 6).

Ghost values, characteristic/eigensystem ghosts, periodic wrap, elliptic pin
consumption: ALL runtime.  The solver golden (X01) is periodic-at-rest, where
every one of these failures is invisible.

Merged from: test_boundary_conditions_per_field.py + test_characteristic_bc.py
+ test_periodic_bc.py + the wall-mass twin of test_sme_dambreak.py + the
REQ-174/174c essentials of test_elliptic_bc_req174.py /
test_chorin_wetdry_positivity_req174c.py.
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import SME, VAM
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree)
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, resolve_per_field, PerFieldBoundary, Wall,
    Extrapolation, Periodic, Coupled, Dirichlet, ZeroNeumann, Flux, Lambda,
    FromModel, CharacteristicFarField, CharacteristicWall)
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.initial_conditions import RP, Constant
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm.solver_chorin_vam_numpy import (
    ChorinSplitVAMSolver, _pad_to_square)
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver]

STATE = ["b", "h", "q_0", "q_1", "q_2"]          # SME(level=2) state


# ── per-field ghost values ──────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.gate
def test_perfield_ghosts():
    """One tag, different fields: momentum REFLECTS (Wall) while h
    EXTRAPOLATES; Wall reflects only the NORMAL component in 2-D (transverse
    kept); the named BC family resolves per-slot (ZeroNeumann copy, Dirichlet
    value, Flux gradient, Lambda delegate)."""
    bcs = resolve_per_field(
        [Wall("left", on="momentum"),
         Extrapolation("left", on="h"),
         Extrapolation("right")],
        STATE, aliases={"momentum": "q"})
    left = bcs.boundary_conditions_list_dict["left"]
    assert isinstance(left, PerFieldBoundary)

    Qin = np.array([0.1, 1.5, 0.7, 0.2, -0.1])    # [b, h, q_0, q_1, q_2]
    ghost = left.face_value(Qin, np.zeros(0), np.array([-1.0]), 0.5, 0.0,
                            np.zeros(0))
    assert np.isclose(ghost[0], Qin[0]),  "b should be copied"
    assert np.isclose(ghost[1], Qin[1]),  "h should EXTRAPOLATE (copied)"
    for k in (2, 3, 4):
        assert np.isclose(ghost[k], -Qin[k]), f"q_{k-2} should REFLECT"
    right = bcs.boundary_conditions_list_dict["right"]
    g_r = right.face_value(Qin, np.zeros(0), np.array([1.0]), 0.5, 0.0,
                           np.zeros(0))
    assert np.allclose(g_r, Qin), "unspecified tag -> all extrapolate"

    # 2-D: NORMAL-only reflection, transverse kept (dimension-agnostic Wall)
    state2d = ["b", "h", "q_x_0", "q_y_0", "q_x_1", "q_y_1"]
    bcs2 = resolve_per_field([Wall("left", on="momentum")], state2d,
                             aliases={"momentum": "q"})
    left2 = bcs2.boundary_conditions_list_dict["left"]
    Qin2 = np.array([0.0, 1.5, 0.7, 0.3, 0.2, -0.1])
    g2 = left2.face_value(Qin2, np.zeros(0), np.array([1.0, 0.0]), 0.5, 0.0,
                          np.zeros(0))
    assert np.isclose(g2[2], -0.7) and np.isclose(g2[4], -0.2), \
        "q_x (normal) must reflect"
    assert np.isclose(g2[3], 0.3) and np.isclose(g2[5], -0.1), \
        "q_y (transverse) must be kept"

    # named family per-slot
    bcs3 = resolve_per_field(
        [Wall("left", on="momentum"),
         Dirichlet("left", on="h", value=1.5),
         ZeroNeumann("left", on="b")],
        STATE, aliases={"momentum": "q"})
    left3 = bcs3.boundary_conditions_list_dict["left"]
    Qin3 = np.array([0.3, 1.0, 0.7, 0.2, -0.1])
    g3 = left3.face_value(Qin3, np.zeros(0), np.array([-1.0]), 0.5, 0.0,
                          np.zeros(0))
    assert np.isclose(g3[0], 0.3)                      # b: ZeroNeumann -> copy
    assert np.isclose(g3[1], 1.5)                      # h: Dirichlet -> value
    assert np.isclose(g3[2], -0.7) and np.isclose(g3[3], -0.2)
    lb = resolve_per_field([Lambda("right", on="q_0",
                                   prescribe_fields={2: lambda *a: 9.0})],
                           STATE, aliases={})
    right_l = lb.boundary_conditions_list_dict["right"]
    assert isinstance(right_l, PerFieldBoundary)
    assert isinstance(right_l._slot_bc[2], Lambda)
    fbcs = resolve_per_field([Flux("right", on="h", gradient=2.0)], STATE,
                             aliases={})
    fr = fbcs.boundary_conditions_list_dict["right"]
    grad = fr.face_gradient(Qin3, Qin3, np.zeros(0), np.array([1.0]), 0.5, 0.0,
                            np.zeros(0))
    assert np.isclose(grad[1], 2.0)                   # h gets the prescribed flux


@pytest.mark.small
@pytest.mark.gate
def test_patch_passthrough_and_errors():
    """Periodic / Coupled are whole-patch BCs detected by TYPE at the tag level:
    never wrapped in a PerFieldBoundary; mixing with per-field at one tag,
    conflicting per-field BCs and unknown fields all raise."""
    bcs = resolve_per_field(
        [Periodic("left", periodic_to_physical_tag="right"), Coupled("top"),
         Wall("front", on="momentum"), Extrapolation("front", on="h")],
        STATE, aliases={"momentum": "q"})
    d = bcs.boundary_conditions_list_dict
    assert type(d["left"]) is Periodic and type(d["top"]) is Coupled
    assert isinstance(d["front"], PerFieldBoundary)
    with pytest.raises(ValueError, match="whole-patch"):
        resolve_per_field([Periodic("left", periodic_to_physical_tag="right"),
                           Wall("left", on="momentum")], STATE,
                          aliases={"momentum": "q"})
    with pytest.raises(ValueError, match="conflicting"):
        resolve_per_field([Wall("left", on="h"), Extrapolation("left", on="h")],
                          STATE, aliases={})
    with pytest.raises(ValueError, match="not a state field"):
        resolve_per_field([Wall("left", on="nope")], STATE, aliases={})


# ── characteristic ghosts ───────────────────────────────────────────────────

def _run_characteristic(bcs, ic, t_end, nc=100):
    sm = SystemModel.from_model(SME(
        closures=[Newtonian(), NavierSlip(), StressFree()], level=0,
        boundary_conditions=bcs))
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return np.asarray(Q[:, :nc], float)


@pytest.mark.small
@pytest.mark.gate
def test_characteristic_lake_at_rest():
    """Far-field characteristic ghost (P- = R 1_{lambda<0} L over the opaque
    eigensystem kernel) holds a lake at rest EXACTLY (1e-12)."""
    ff = [CharacteristicFarField(tag=tg, far_field=[0.0, 1.0, 0.0])
          for tg in ("left", "right")]
    Q = _run_characteristic(BoundaryConditions(ff),
                            lambda xv: np.array([0.0, 1.0, 0.0]), t_end=0.5)
    assert np.abs(Q[1] - 1.0).max() < 1e-12
    assert np.abs(Q[2]).max() < 1e-12


@pytest.mark.large
def test_characteristic_radiation_and_mass():
    """[large] A surface hump radiates OUT through far-field boundaries (domain
    relaxes back, residual < 0.08), and a CharacteristicWall closed box
    conserves mass to 5e-8 while reflecting."""
    ff = [CharacteristicFarField(tag=tg, far_field=[0.0, 1.0, 0.0])
          for tg in ("left", "right")]

    def hump(xv):
        xx = float(xv[0])
        return np.array([0.0, 1.0 + 0.2 * np.exp(-((xx - 5.0) ** 2) / 0.5), 0.0])

    Q = _run_characteristic(BoundaryConditions(ff), hump, t_end=1.5)
    assert np.all(np.isfinite(Q))
    assert np.abs(Q[1] - 1.0).max() < 0.08      # hump (0.2) mostly gone

    wl = CharacteristicWall(tag="left", momentum_field_indices=[[2]])
    wr = CharacteristicWall(tag="right", momentum_field_indices=[[2]])

    def dam(xv):
        return np.array([0.0, 1.2 if float(xv[0]) < 5.0 else 1.0, 0.0])

    Q = _run_characteristic(BoundaryConditions([wl, wr]), dam, t_end=1.0)
    assert np.all(np.isfinite(Q))
    mass = Q[1].sum() * 0.1
    assert abs(mass - 11.0) < 5e-8


# ── periodic wrap ───────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.gate
def test_periodic_idempotent_and_wrap(one_hyperbolic_step):
    """(1) resolve_periodic_bcs is IDEMPOTENT on the same mesh (a
    non-idempotent resolve alternates periodic/open across restarts);
    (2) one real step on a periodic advecting bump conserves mass to 1e-12 and
    remaps the internal mesh to the opposite side — the two historical breaks
    that silently degraded Periodic to extrapolation."""
    NC = 50
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=NC)
    for _ in range(3):
        mesh.resolve_periodic_bcs(bcs)
        assert (mesh.boundary_face_cells[0] == NC - 1
                and mesh.boundary_face_cells[1] == 0)

    # 1-step wrap twin
    NC, XMAX = 100, 10.0
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    sm = SystemModel.from_model(SME(level=0,
                                    closures=[Newtonian(), NavierSlip(),
                                              StressFree()],
                                    parameters={"nu": 1e-6, "lambda_s": 0.0},
                                    boundary_conditions=bcs))
    n_state = len(sm.state)

    def ic(xv):
        x = float(xv[0])
        out = np.zeros(n_state)
        out[1] = 1.0 + 0.3 * np.exp(-((x - 9.0) ** 2) / 0.1)
        out[2] = out[1] * 1.0          # u = 1: bump moves right, must wrap
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    xc = np.linspace(XMAX / NC / 2, XMAX - XMAX / NC / 2, NC)
    dx = XMAX / NC
    m_ic = sum(ic([x])[1] for x in xc) * dx
    solver = HyperbolicSolver(
        time_end=0.5, compute_dt=timestepping.adaptive(CFL=0.9, dimension=1))
    Q = one_hyperbolic_step(solver, mesh, nsm)
    assert solver._bf_cells[0] == NC - 1 and solver._bf_cells[1] == 0
    assert np.all(np.isfinite(Q[:, :NC]))
    np.testing.assert_allclose(Q[1, :NC].sum() * dx, m_ic, rtol=1e-12)


# ── model-derived wall: mirror ghost + closed-box mass ──────────────────────

@pytest.mark.small
@pytest.mark.gate
def test_wall_mass_twin(one_hyperbolic_step):
    """The wall is DEFINED in the derivation (register_group('boundary:wall'))
    and accessed via FromModel: the ghost REFLECTS the momentum vector about
    the face, ``q -> q - 2 n (n.q)`` — in 1-D (n0 = +/-1) that is the mirror
    state ``-q`` for every moment, with b/h extrapolating; so a closed-box dam
    break conserves mass to 1e-10 after a real step.  Asserted on the ghost
    with the face normal SUBSTITUTED (the physics), not on the un-evaluated
    n0-carrying expression tree."""
    sm = SystemModel.from_model(SME(
        closures=[Newtonian(), NavierSlip(), StressFree()], level=2))
    bc = FromModel(tag="left", definition="wall").resolve(sm)
    b, h, q0, q1, q2 = sm.state
    for n0 in (1, -1):                       # both outward orientations
        ghost = bc.compute_boundary_condition(
            sm.time, sm.position, None, sm.variables,
            sm.aux_variables, sm.parameters, [n0])
        assert [sp.expand(g) for g in ghost] == [b, h, -q0, -q1, -q2], (
            f"1-D wall ghost wrong for n0={n0}: {list(ghost)}")

    # closed-box 1-step mass twin
    nc, hL, hR = 50, 2.0, 1.0
    sm = SystemModel.from_model(SME(
        closures=[Newtonian(), NavierSlip(), StressFree()], level=2,
        boundary_conditions=BoundaryConditions(
            [FromModel(tag="left", definition="wall"),
             FromModel(tag="right", definition="wall")])))
    n_state = len(sm.state)
    high = np.zeros(n_state); high[1] = hL
    low = np.zeros(n_state);  low[1] = hR
    sm.initial_conditions = RP(high=lambda n, hi=high: hi,
                               low=lambda n, lo=low: lo, jump_position_x=1.0)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 2.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=1.0,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    h_row = one_hyperbolic_step(solver, mesh, nsm)[1, :nc]
    assert np.all(np.isfinite(h_row)) and h_row.min() > 0.0
    mass0 = (hL + hR) / 2 * 2.0
    assert abs(h_row.sum() * (2.0 / nc) - mass0) < 1e-10 * mass0, "wall leaks mass"


@pytest.mark.small
@pytest.mark.gate
def test_wall_is_normal_aware_2d():
    """The model-registered wall is a FREE-SLIP wall in 2-D: it reflects only
    the NORMAL component of every momentum moment and leaves the TANGENTIAL
    one untouched.

    Before the FromModel normal substitution the registered wall was a
    normal-BLIND full reversal (q_d -> -q_d in EVERY direction), which
    destroyed the tangential momentum of any flow running along the wall
    (measured: 94% of q_t lost over 0.5 s).  Asserted EXACTLY (= 0), on the
    ghost, for an axis-aligned AND an oblique face — the oblique face is the
    one a per-direction ``wall_x``/``wall_y`` spec could never represent.
    """
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.ml_sme import MLSME
    from zoomy_core.model.models.ml_vam import MLVAM

    # VAM / ML-VAM registered NO wall at all until the lateral-wall frame was
    # wired into their derivations: FromModel(definition="wall") raised
    # KeyError on every VAM.  Their vertical (r) and pressure (P) moments are
    # scalars under a HORIZONTAL wall normal and must pass through untouched —
    # only the horizontal momentum vector of each moment reflects.
    # SWE carries no stress moments; every moment family needs its σ rows
    # closed, or ``from_model`` refuses the open system.
    _c = dict(closures=[Newtonian(), NavierSlip(), StressFree()])
    for build in (lambda **k: SWE(dimension=2, **k),
                  lambda **k: SME(level=0, dimension=3, **_c, **k),
                  lambda **k: SME(level=1, dimension=3, **_c, **k),
                  lambda **k: MLSME(n_layers=2, level=0, dimension=3, **_c, **k),
                  lambda **k: VAM(level=1, dimension=3, **_c, **k),
                  lambda **k: MLVAM(n_layers=2, level=1, dimension=3, **_c, **k)):
        sm = SystemModel.from_model(build())
        bc = FromModel(tag="left", definition="wall").resolve(sm)
        names = [str(s) for s in sm.state]
        xi = [i for i, s in enumerate(names) if s == "hu" or "_x" in s]
        yi = [i for i, s in enumerate(names) if s == "hv" or "_y" in s]
        assert xi and len(xi) == len(yi), names
        p = np.zeros(len(list(sm.parameters)))
        X = np.zeros(3)

        def ghost(Q, n):
            return np.array([float(v) for v in bc.compute_boundary_condition(
                0.0, X, X, np.asarray(Q, float), np.zeros(1), p, np.asarray(n))])

        # (a) axis-aligned wall n=(1,0), flow purely tangential (+y)
        Q = np.zeros(len(names)); Q[1] = 2.0
        for i in yi:
            Q[i] = 1.0
        g = ghost(Q, [1.0, 0.0])
        assert max(abs(g[i] - Q[i]) for i in yi) == 0.0, \
            f"{names}: wall destroyed the TANGENTIAL momentum -> {g}"
        assert max(abs(g[i] + Q[i]) for i in xi) == 0.0, \
            f"{names}: wall did not reflect the NORMAL momentum -> {g}"
        assert g[1] == Q[1], f"{names}: h must extrapolate at a wall -> {g}"

        # (b) oblique wall n=(1,1)/sqrt2, flow tangential to it: ghost == inner
        r = np.sqrt(0.5)
        Qo = np.zeros(len(names)); Qo[1] = 2.0
        for i in xi:
            Qo[i] = r
        for i in yi:
            Qo[i] = -r
        assert np.abs(ghost(Qo, [r, r]) - Qo).max() == 0.0, \
            f"{names}: oblique free-slip wall is not transparent to tangential flow"

        # (c) lake at rest against the wall: h through, q stays zero
        Qr = np.zeros(len(names)); Qr[1] = 2.0
        gr = ghost(Qr, [0.0, 1.0])
        assert gr[1] == Qr[1] and np.abs(gr[2:]).max() == 0.0, \
            f"{names}: lake at rest is not preserved at a wall -> {gr}"


# ── Chorin elliptic/predictor BC consumption (REQ-174 / 174c) ───────────────

G, H_RES, H_DRY, Q_IN = 9.81, 0.34, 0.015, 0.11197
DOMAIN = (-1.5, 1.5)
PIN = 0.5                        # non-zero => crisp exactness + no early-exit
BUMP = lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


@pytest.fixture(scope="module")
def vam3d_split():
    """SHARED VAM(1, dim=3) extruded-escalante split (warm cache): inflow
    discharge left, Dirichlet-P pin (PIN) at the x-hi outflow, lateral
    extrapolation — one expensive derivation for the whole file."""
    from zoomy_core.model.models.closures import Newtonian as NT, StressFree as SF
    bcs = [Dirichlet("left", on="q_x_0", value=Q_IN),
           Dirichlet("left", on="q_x_1", value=0.0),
           Dirichlet("left", on="q_y_0", value=0.0),
           Dirichlet("left", on="q_y_1", value=0.0),
           Dirichlet("left", on="r_0", value=0.0),
           Dirichlet("left", on="r_1", value=0.0),
           Dirichlet("right", on="P_0", value=PIN),
           Dirichlet("right", on="P_1", value=PIN),
           Extrapolation(tag="bottom"), Extrapolation(tag="top")]
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[NT(), SF()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    split = model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    return split, sm


def _setup_chorin(vam3d_split, nx=30, ny=4):
    split, sm = vam3d_split
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        pressure_tol=1e-9, pressure_maxit=200, pressure_solver="lu")
    mesh = BaseMesh.create_2d((DOMAIN[0], DOMAIN[1], 0.0, 0.4), nx, ny)
    Q = np.asarray(solver.setup_simulation(mesh))
    idx = {str(s): k for k, s in enumerate(sm.state)}
    cc = np.asarray(solver._sim_mesh.cell_centers)
    xc = cc[0, :solver.nc]
    b = BUMP(xc)
    Q[idx["b"], :solver.nc] = b
    Q[idx["h"], :solver.nc] = np.maximum(
        np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    solver.update_aux_variables()
    return solver, idx, xc, cc


@pytest.mark.small
@pytest.mark.gate
def test_chorin_bc_consumption(vam3d_split):
    """REQ-174 + REQ-174c on ONE shared VAM d3 fixture:

    * ``_pad_to_square`` propagates ``_bc_source`` (tag-remap root cause);
    * per-face ghosts route correctly (outflow extrapolates q, inflow keeps
      its declared Dirichlet discharge);
    * the elliptic stage CONSUMES the declared Dirichlet-P pin to machine
      precision and ``last_elliptic_rel_resid`` is the residual of the ACTUAL
      solved system;
    * one full step keeps ``h >= 0`` with exact top/bottom symmetry
      (positivity is a per-step property; the pre-fix mis-route broke it on
      step one)."""
    split, sm = vam3d_split

    # (a) padded predictor carries the BC container
    padded = _pad_to_square(split.SM_pred)
    src = getattr(padded, "_bc_source", None)
    assert src is not None, (
        "_pad_to_square dropped the BC container — the flux operator's "
        "boundary-tag remap will be skipped and 2-D tags mis-route.")
    assert list(src.list_sorted_function_names) == ["bottom", "left", "right", "top"]

    # (b) per-face ghosts: outflow extrapolates q_x, inflow keeps Q_IN
    solver, idx, xc, cc = _setup_chorin(vam3d_split)
    mesh = solver._sim_mesh
    fc = mesh.face_centers
    normals = np.asarray(mesh.face_normals)
    qx0 = idx["q_x_0"]
    Q = solver._sim_Q
    Qaux = solver._sim_Qaux
    x_hi = float(fc[:, 0].max())
    x_lo = float(fc[:, 0].min())
    seen_out = seen_in = False
    for i in range(solver._n_bf):
        fidx = solver._bf_fidx[i]
        fx = fc[fidx, 0]
        gh = np.asarray(solver._bc_fn(
            solver._bc_indices[i], 0.0, fc[fidx, :], solver._d_face[i],
            Q[:, solver._bf_cells[i]], Qaux[:, solver._bf_cells[i]],
            solver._sim_parameters, normals[:, fidx]), float).reshape(-1)
        if abs(fx - x_hi) < 1e-9:
            assert abs(gh[qx0]) < 1e-9, (
                f"outflow ghost q_x_0 = {gh[qx0]} (expected extrapolation); "
                "the outflow inherited the inflow Dirichlet (mis-route).")
            seen_out = True
        elif abs(fx - x_lo) < 1e-9:
            assert abs(gh[qx0] - Q_IN) < 1e-9, (
                f"inflow ghost q_x_0 = {gh[qx0]} (expected Q_IN={Q_IN})")
            seen_in = True
    assert seen_out and seen_in

    # (c) Dirichlet-P pin exact + rel_resid of the actual system + positivity
    pd = solver._press_dir
    assert pd is not None, "a declared Dirichlet P must populate _press_dir"
    pinned = np.nonzero(pd["cell_mask"][0])[0]
    assert pinned.size > 0
    assert np.allclose(xc[pinned], xc.max()), "pin must land on the outflow column"

    nx = 30
    dx = (DOMAIN[1] - DOMAIN[0]) / nx
    dt = 0.08 * dx / np.sqrt(G * H_RES)
    yc = cc[1, :solver.nc]
    x_cols = np.unique(np.round(xc, 6))
    solver.step(dt)

    Qn = np.asarray(solver._sim_Q, dtype=float)
    for k, s in enumerate(solver._press_state_idx):
        cells = np.nonzero(pd["cell_mask"][k])[0]
        assert np.allclose(Qn[s, cells], PIN, atol=1e-9), (
            f"mode {k}: pinned cells = {Qn[s, cells]}, expected {PIN}")
    r = solver.last_elliptic_rel_resid
    assert r is not None and np.isfinite(r)
    assert 0.0 <= r < 1e-6

    h = Qn[idx["h"], :solver.nc]
    assert np.isfinite(h).all(), "non-finite h after one step"
    assert h.min() >= 0.0, f"h went negative after one step: {h.min():.3e}"
    tb = 0.0
    for xv in x_cols:
        col = h[np.abs(xc - xv) < 1e-9]
        ys = yc[np.abs(xc - xv) < 1e-9]
        colo = col[np.argsort(ys)]
        tb = max(tb, float(np.max(np.abs(colo - colo[::-1]))))
    assert tb < 1e-11, f"top/bottom asymmetry {tb:.3e} (tag mis-route)"
