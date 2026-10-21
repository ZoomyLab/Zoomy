"""REQ-194 — the depth-law parameter-study MACHINERY.

These tests pin that every axis is selectable at CONSTRUCTION time and that the
whole grid constructs and emits.  They assert NOTHING about which setting is
better — that is what the study is for, and baking a preference in here would
prejudge it.

What IS pinned:

* the 16-point grid (2 paths × 4 guards × 2 spectrum treatments) constructs and
  emits for SWE(dim=1,2) and SME(level=1,2);
* the two regularization paths are MUTUALLY EXCLUSIVE on a system;
* both paths share ONE Kurganov–Petrova definition — no ``1/(h+eps)``, no
  ``1/max(eps,h)`` reciprocal anywhere;
* the regularization epsilon is its OWN knob and does not read ``wet_dry_eps``;
* ``map_operator_slots`` reaches every slot by default and RAISES on a typo'd
  exclusion (the failure mode the old closed 14-name list had);
* the NSM's shipped defaults are UNCHANGED — a caller that passes none of the
  new axes still gets exactly ``[desingularize_hinv()]``.
"""
import sympy as sp
import pytest

from zoomy_core.model.boundary_conditions import Extrapolation
from zoomy_core.model.models import SME, SWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.depth_law_study import (
    AXES,
    DEFAULT_EPS,
    axis_id,
    build_nsm,
    iter_axes,
)
from zoomy_core.systemmodel.operations import (
    OPERATOR_SLOTS,
    map_operator_slots,
    regularize_depth_aux,
    regularize_depth_direct,
)
from zoomy_core.systemmodel.system_model import SystemModel

# PARKED under the REQ-194 study tag (approved spec §3): shipped bits are
# pinned by N01; this study scaffolding is excluded from EVERY tier and runs
# only via an explicit `-m study`.  Dies when the study closes.
pytestmark = [pytest.mark.systemmodel, pytest.mark.study]
from zoomy_core.transformation.generic_c import GenericCppModel


# ── the systems under study ───────────────────────────────────────────────

def _swe(dim):
    return lambda: SWE(dimension=dim)


def _sme(level):
    return lambda: SME(
        level=level, parameters={"nu": 1e-3, "lambda_s": 0.0},
        closures=[Newtonian(), NavierSlip(), StressFree()],
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")])


SYSTEMS = {
    "SWE_d1": _swe(1),
    "SWE_d2": _swe(2),
    "SME_l1": _sme(1),
    "SME_l2": _sme(2),
}

GRID = list(iter_axes())


# ── (1) the whole grid constructs AND emits ───────────────────────────────

@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("combo", GRID, ids=[axis_id(c) for c in GRID])
def test_grid_point_constructs_and_emits(system, combo):
    """Every one of the 16 axis points builds a derived NSM and lowers to code
    for every system.  ``create_code`` is the strictest cheap check available:
    it walks every operator slot, scalarizes it and prints it, so an operation
    that left a malformed tree (unbound aux symbol, ragged array, stale kernel
    signature) fails here rather than at run time."""
    nsm = build_nsm(SYSTEMS[system](), **combo)
    assert isinstance(nsm, NumericalSystemModel)
    code = GenericCppModel(nsm).create_code()
    assert "res[0]" in code, f"{system}/{axis_id(combo)}: nothing emitted"


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_grid_is_16_points(system):
    """The advertised grid is exactly 2 paths × 4 guards × 2 treatments."""
    assert len(GRID) == 16
    assert len(AXES["path"]) == 2
    assert len(AXES["guard"]) == 4
    assert len(AXES["eigenvalues"]) == 2


# ── (2) the two paths are mutually exclusive ──────────────────────────────

def test_paths_are_mutually_exclusive():
    """Applying both regularization paths to one system RAISES.  They
    regularize the same reciprocal, so the second would rewrite what the first
    already replaced and leave the operators disagreeing about what ``1/h``
    means — an error, not a double-regularization."""
    sm = SystemModel.from_model(SWE(dimension=1))
    sm.apply(regularize_depth_direct(1e-2))
    with pytest.raises(ValueError, match="mutually exclusive"):
        sm.apply(regularize_depth_aux(1e-2))

    sm2 = SystemModel.from_model(SWE(dimension=1))
    sm2.apply(regularize_depth_aux(1e-2))
    with pytest.raises(ValueError, match="mutually exclusive"):
        sm2.apply(regularize_depth_direct(1e-2))


def test_same_path_twice_also_raises():
    """Re-applying the SAME path is equally an error — it would nest the KP
    form inside itself."""
    sm = SystemModel.from_model(SWE(dimension=1))
    sm.apply(regularize_depth_aux(1e-2))
    with pytest.raises(ValueError, match="already applied"):
        sm.apply(regularize_depth_aux(1e-2))


# ── (3) ONE Kurganov–Petrova definition, no naive reciprocal ──────────────

def _kp_radicand(h, eps):
    """The radicand of the KP denominator, ``h⁴ + max(h, eps)⁴``.

    Matched on the RADICAND rather than on ``sqrt(...)``: the reciprocal enters
    the operators as ``(h⁴ + max(h,eps)⁴)**(-1/2)``, and ``expr.has(sqrt(X))``
    does NOT match ``X**(-1/2)`` — they are different ``Pow`` nodes.  The
    radicand is present in both."""
    return h ** 4 + sp.Max(sp.Float(eps), h) ** 4


def _depth_floors(expr, h):
    """Every ``Max(<number>, h)`` node in ``expr`` — a direct floor of the depth
    at a constant.

    Deliberately narrow: it does NOT match the model's own wet/dry momentum cap
    ``Max(0, h - wet_dry_eps)`` (whose depth argument is an Add, not ``h``),
    which is pre-existing model behaviour and none of this REQ's business."""
    return [m for m in expr.atoms(sp.Max)
            if h in m.args and any(a.is_number for a in m.args)]


@pytest.mark.parametrize("path", ["direct", "aux"])
def test_both_paths_use_the_same_kp_denominator(path):
    """Both study paths carry the SAME KP denominator ``√(h⁴ + max(h,eps)⁴)``
    — the direct path inline in the operators, the aux path in the ``hinv``
    update row.  There is exactly one regularized-reciprocal definition."""
    nsm = build_nsm(_swe(1)(), path=path, eps=1e-2)
    h = next(s for s in nsm.state if str(s) == "h")
    if path == "aux":
        # AUX: the reciprocal lives in exactly ONE place, the hinv update row.
        carriers = [sp.sympify(next(
            nsm.update_aux_variables[i, 0]
            for i, s in enumerate(nsm.aux_state) if str(s) == "hinv"))]
    else:
        # DIRECT: it is inlined, so it shows up in the momentum flux entries.
        carriers = [sp.sympify(e) for e in sp.flatten(nsm.flux)]
    assert any(c.has(_kp_radicand(h, 1e-2)) for c in carriers), (
        f"{path}: KP denominator not found in {carriers}")


@pytest.mark.parametrize("path", ["direct", "aux"])
def test_no_naive_reciprocal_anywhere(path):
    """Neither ``1/(h + eps)`` nor ``1/max(eps, h)`` appears in ANY slot.  KP is
    the only regularization form; a bare ``Max(eps, h)`` outside the KP
    denominator would be a floor on the depth, which is forbidden."""
    eps = 1e-2
    nsm = build_nsm(_swe(2)(), path=path, eps=eps)
    h = next(s for s in nsm.state if str(s) == "h")
    kp_scale = sp.Max(sp.Float(eps), h)
    kp_rad = _kp_radicand(h, eps)
    for slot in OPERATOR_SLOTS:
        obj = getattr(nsm, slot, None)
        if obj is None:
            continue
        obj = getattr(obj, "definition", obj)
        entries = [obj] if isinstance(obj, sp.Basic) and not isinstance(
            obj, (sp.NDimArray, sp.MatrixBase)) else sp.flatten(obj)
        for e in entries:
            e = sp.sympify(e)
            # (i) the ONLY constant floor of the depth is KP's denominator
            #     SCALE — no 1/max(1e-14, h), no other clamp.
            for m in _depth_floors(e, h):
                assert m == kp_scale, (
                    f"{slot}: depth floor {m} is not the KP denominator scale")
            # (ii) no naive 1/(h + eps): a negative Pow over an Add containing
            #      h, where that Add is not the KP radicand.
            for p in e.atoms(sp.Pow):
                if (p.exp.is_number and p.exp.is_negative
                        and isinstance(p.base, sp.Add) and p.base.has(h)):
                    assert p.base == kp_rad, (
                        f"{slot}: naive reciprocal 1/({p.base}) present")


# ── (4) eps is its own knob, NOT wet_dry_eps ──────────────────────────────

def test_regularization_eps_is_not_wet_dry_eps():
    """The regularization scale is a literal from the ``eps`` knob, and the
    model's ``wet_dry_eps`` parameter symbol never enters the reciprocal.
    Conflating the two is what put ``h/(h + 1e-2)`` on the celerity."""
    nsm = build_nsm(_swe(1)(), path="aux", eps=1e-7)
    row = sp.sympify(next(
        nsm.update_aux_variables[i, 0]
        for i, s in enumerate(nsm.aux_state) if str(s) == "hinv"))
    assert row.has(sp.Float(1e-7)), f"eps knob not honoured: {row}"
    wde = nsm.parameters.wet_dry_eps
    assert not row.has(wde), f"wet_dry_eps leaked into the reciprocal: {row}"


def test_eps_default_is_1em2():
    """The study default is the user's 'start near the old working state'."""
    assert DEFAULT_EPS == 1e-2
    nsm = build_nsm(_swe(1)(), path="aux")
    row = sp.sympify(next(
        nsm.update_aux_variables[i, 0]
        for i, s in enumerate(nsm.aux_state) if str(s) == "hinv"))
    assert row.has(sp.Float(1e-2))


# ── (5) the eigenvalue axes are independent of the path ───────────────────

@pytest.mark.parametrize("path", ["direct", "aux"])
def test_eigenvalue_exclusion_keeps_exact_reciprocal(path):
    """``eigenvalues="exclude"`` leaves the spectrum on the EXACT ``1/h`` while
    the flux still carries the regularized one — the carve-out expressed as a
    slot exclusion rather than as a depth floor."""
    excl = build_nsm(_swe(2)(), path=path, eigenvalues="exclude")
    h = next(s for s in excl.state if str(s) == "h")
    ev = [sp.sympify(e) for e in sp.flatten(excl.eigenvalues)]
    assert any(e.has(sp.Pow(h, -1)) for e in ev), (
        f"{path}: excluded spectrum does not carry the exact 1/h: {ev}")
    # ... and the exclusion is genuinely selective: the flux was still swept.
    flux = [sp.sympify(e) for e in sp.flatten(excl.flux)]
    assert not any(e.has(sp.Pow(h, -1)) for e in flux), (
        f"{path}: flux kept a bare 1/h — the sweep did not run")


@pytest.mark.parametrize("path", ["direct", "aux"])
def test_eigenvalue_regularization_reaches_the_spectrum(path):
    """``eigenvalues="regularize"`` (the other axis point) leaves NO bare
    ``1/h`` in the spectrum."""
    nsm = build_nsm(_swe(2)(), path=path, eigenvalues="regularize")
    h = next(s for s in nsm.state if str(s) == "h")
    ev = [sp.sympify(e) for e in sp.flatten(nsm.eigenvalues)]
    assert not any(e.has(sp.Pow(h, -1)) for e in ev), ev


def test_guard_axis_is_independent_of_path_and_eps():
    """The guard axis composes with every path: the dry conditional appears iff
    the guard asked for it, whatever the reciprocal underneath."""
    cond = sp.Function("conditional")
    for path in ("direct", "aux"):
        for guard, want in ((None, False), ("power", False),
                            ("gate", True), ("both", True)):
            nsm = build_nsm(_swe(1)(), path=path, guard=guard, eps=1e-3)
            got = any(sp.sympify(e).has(cond)
                      for e in sp.flatten(nsm.eigenvalues))
            assert got is want, f"path={path} guard={guard}: gate={got}"


# ── (6) map_operator_slots coverage + typo'd exclusion raises ─────────────

def test_exclusion_typo_raises():
    """A misspelled exclusion RAISES instead of silently widening coverage —
    the exact failure mode of the predecessor's closed 14-name list, which
    skipped ``interpolate_to_3d`` and every boundary kernel without a word."""
    sm = SystemModel.from_model(SWE(dimension=1))
    with pytest.raises(KeyError, match="unknown slot"):
        map_operator_slots(sm, lambda e: e, exclude=("eigenvaluess",))


def test_sweep_reaches_the_slots_the_old_list_missed():
    """``interpolate_to_3d`` and the boundary kernels are in the default scope.
    The old closed list reached neither, so an ``interpolate_to_3d`` carrying a
    bare ``hu/h`` survived every desingularization."""
    assert "interpolate_to_3d" in OPERATOR_SLOTS
    assert "boundary_conditions" in OPERATOR_SLOTS
    sm = SystemModel.from_model(SWE(dimension=1))
    report = map_operator_slots(sm, lambda e: sp.sympify(e))
    assert set(report) == set(OPERATOR_SLOTS)
    assert all(v in ("rewritten", "unchanged", "absent", "excluded")
               for v in report.values())


# ── (7) shipped defaults ──────────────────────────────────────────────────

def test_nsm_regularization_defaults_are_unchanged_by_this_req():
    """A caller that passes none of the REGULARIZATION axes still gets the
    legacy ``desingularize_hinv()``: ``depth_regularization``,
    ``eigenvalue_guard`` and ``eigenvalue_treatment`` all keep pre-REQ-194
    behaviour, so no existing backend's regularization moves.

    ``normalize_face_normal`` now leads the list: REQ-208 item (2) flipped
    ``normalize_normal`` to default True.  That is the one REQ-194 axis which
    deliberately DOES move emitted code — it is a fact about the mesh (every
    ``_face_normals_*`` builder divides by ``np.linalg.norm``), and leaving it
    opt-in made it unreachable, since the flag is set at NSM construction
    inside each backend's case code and no case ever passed it.
    """
    for build in (_swe(1), _swe(2), _sme(1)):
        nsm = NumericalSystemModel.from_system_model(
            SystemModel.from_model(build()))
        names = [getattr(op, "name", None) for op in nsm.default_operations()]
        assert names == ["normalize_face_normal", "desingularize_hinv"], names


def test_normal_normalization_is_opt_out_able():
    """The REQ-208 default is a default, not a hard-wiring."""
    for build in (_swe(1), _swe(2), _sme(1)):
        nsm = NumericalSystemModel.from_system_model(
            SystemModel.from_model(build()), normalize_normal=False)
        names = [getattr(op, "name", None) for op in nsm.default_operations()]
        assert names == ["desingularize_hinv"], names
