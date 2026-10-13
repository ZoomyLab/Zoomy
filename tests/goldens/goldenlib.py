"""Golden-snapshot infrastructure for the zoomy_core test suite.

APPROVED spec: steward proposals 2026-07-19-test-refactor-core(-v2).md.

A *golden* is a checked-in text file under ``tests/goldens/`` — one file per
golden — holding a NORMALIZED snapshot of a derived object:

* model goldens      — the ``SystemModel`` built from a model, derived
                       **NO-CACHE** (``ZOOMY_DERIVATION_CACHE=0`` for the
                       build), every operator rendered as canonical srepr;
* systemmodel goldens — ``from_model`` freeze / Chorin splits;
* NSM goldens        — ``NumericalSystemModel`` incl. numerics knobs and the
                       built face-kernel definitions;
* printer goldens    — the full emitted source (verbatim);
* procedure goldens  — the solver Procedure/Statement IR walked to C-family
                       text (``Q01``); the shared example procedures are built
                       by ``solver_procedure_examples`` so the golden and the
                       python-``ProcedureBuilder`` execution tests check the
                       SAME IR.

Normalization: canonical ``srepr`` per entry (symbols print NAME-ONLY —
assumptions are pinned once per symbol from the deduced ``assumptions0`` in a
``symbol_assumptions`` block, killing the constructor-kwargs
process-history churn), Dummy symbols alpha-renamed in deterministic
(pre-order) traversal order, fixed row-major entry ordering, zero entries
elided.  Each file carries a header recording the sympy version; the header
is NOT part of the comparison (the body is).

Regeneration: ``python scripts/regen_goldens.py`` rebuilds every golden (and
the N02 time-canary baseline); review = ``git diff``.  A regen touching more
than one family requires the rederive tier green first (re-bless protocol).
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform

import sympy as sp

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests" / "goldens"

# Header lines carry a DISTINCTIVE prefix: a bare '#' would collide with the
# C-preprocessor lines (#pragma/#include) in the printer goldens' bodies —
# read_golden_body would silently strip them from the expectation.
HEADER_PREFIX = "#|"


# ── env: NO-CACHE derivation for model goldens ─────────────────────────────

@contextlib.contextmanager
def no_cache():
    """Force a fresh symbolic derivation: disables BOTH the in-process
    spec-keyed derivation memo (basemodel) and the REQ-163 sm_cache tiers
    (memory / user-dir / _prebuilt).  Model goldens derive under this by spec."""
    old = os.environ.get("ZOOMY_DERIVATION_CACHE")
    os.environ["ZOOMY_DERIVATION_CACHE"] = "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ZOOMY_DERIVATION_CACHE", None)
        else:
            os.environ["ZOOMY_DERIVATION_CACHE"] = old


# ── normalization ──────────────────────────────────────────────────────────

from sympy.printing.repr import ReprPrinter as _ReprPrinter


class _CanonRepr(_ReprPrinter):
    """srepr with construction-order-INDEPENDENT symbols.

    Plain ``sp.srepr`` prints a Symbol's *original* constructor kwargs
    (``_assumptions_orig``) — and sympy's symbol cache hands the same object
    to every later construction with equivalent deduced assumptions, so the
    printed kwargs depend on WHICH construction ran first in the process
    (e.g. ``Symbol('g', positive=True)`` vs ``Symbol('g', real=True,
    positive=True)`` after a prebuilt-cache pickle load).  Symbols therefore
    print NAME-ONLY here; the golden pins the assumptions separately from
    the deterministic deduced ``assumptions0`` (see ``symbol_assumptions``
    in ``render_system_model``)."""

    def _print_Symbol(self, expr):
        return f"{expr.__class__.__name__}({self._print(expr.name)})"

    _print_Dummy = _print_Symbol


_CANON = _CanonRepr()


def canonical_true_facts(sym) -> str:
    """Deterministic assumption fingerprint of a Symbol: the sorted TRUE
    facts of the DEDUCED ``assumptions0`` (construction-order independent),
    with the always-on ``commutative`` elided."""
    facts = sorted(k for k, v in sym.assumptions0.items()
                   if v is True and k != "commutative")
    return "{" + ", ".join(facts) + "}"


def norm_expr(e) -> str:
    """Canonical srepr of ``e``: Dummy symbols alpha-renamed in deterministic
    pre-order traversal order (``_dummy0, _dummy1, …``) and every Symbol
    printed name-only (assumptions pinned separately — see ``_CanonRepr``),
    so a golden never diffs on sympy's dummy counter or on the process's
    symbol-construction history."""
    e = sp.sympify(e)
    dummies = []
    for node in sp.preorder_traversal(e):
        if isinstance(node, sp.Dummy) and node not in dummies:
            dummies.append(node)
    if dummies:
        e = e.xreplace({d: sp.Symbol(f"_dummy{i}", **d.assumptions0)
                        for i, d in enumerate(dummies)})
    return _CANON.doprint(e)


def _flatiter(arr):
    """Yield (index-tuple, entry) for a ZArray / Matrix / list, row-major."""
    if hasattr(arr, "shape") and hasattr(arr, "__getitem__"):
        shape = arr.shape
        if len(shape) == 0:
            yield (), arr
            return
        import itertools
        for idx in itertools.product(*(range(s) for s in shape)):
            yield idx, arr[idx if len(idx) > 1 else idx[0]]
        return
    for i, e in enumerate(arr):
        yield (i,), e


def render_array(label: str, arr, out: list) -> None:
    """Render a (possibly None / all-zero) operator array: a shape line plus
    one ``label[i,j,...] = <srepr>`` line per NON-zero entry, row-major."""
    if arr is None:
        out.append(f"{label}: None")
        return
    try:
        shape = tuple(arr.shape) if hasattr(arr, "shape") else (len(list(arr)),)
    except TypeError:
        out.append(f"{label}: <unrenderable {type(arr).__name__}>")
        return
    out.append(f"{label}: shape={shape}")
    for idx, e in _flatiter(arr):
        try:
            expr = sp.sympify(e)
        except (sp.SympifyError, AttributeError, TypeError):
            out.append(f"{label}[{','.join(map(str, idx))}] = <raw> {e!r}")
            continue
        if expr == 0:
            continue
        out.append(f"{label}[{','.join(map(str, idx))}] = {norm_expr(expr)}")


def render_mapping(label: str, mapping, out: list) -> None:
    if mapping is None:
        out.append(f"{label}: None")
        return
    out.append(f"{label}:")
    items = mapping.items() if hasattr(mapping, "items") else enumerate(mapping)
    for k, v in items:
        try:
            key = norm_expr(k) if isinstance(k, sp.Basic) else str(k)
        except Exception:
            key = str(k)
        out.append(f"  {key} -> {norm_expr(v)}")


# ── SystemModel snapshot ───────────────────────────────────────────────────

def render_system_model(sm, *, title: str = "SystemModel") -> str:
    """Deterministic text snapshot of a SystemModel's full frozen surface.

    Includes state/aux/parameters, every canonical operator (M, F, P, A, B, S,
    S_exp), eigenvalues, the state-hygiene maps (update_variables /
    update_aux_variables), the WB reconstruction pair, interpolate/project,
    boundary specs + attached BC tags + BC kernel, the aux registry and the
    stationary rows — the pins the absorbed structure tests relied on."""
    out = [f"== {title} =="]
    out.append(f"class: {type(sm).__name__}")
    out.append("state: [" + ", ".join(norm_expr(s) for s in sm.state) + "]")
    out.append("aux_state: [" + ", ".join(norm_expr(s) for s in sm.aux_state) + "]")
    pnames = list(sm.parameters.keys()) if hasattr(sm.parameters, "keys") else []
    out.append("parameters: [" + ", ".join(str(p) for p in pnames) + "]")
    # assumptions pinned ONCE per symbol, from the DEDUCED assumptions0
    # (deterministic) — expressions above/below print symbols name-only.
    out.append("symbol_assumptions:")
    seen = set()
    psyms = []
    if hasattr(sm.parameters, "values"):
        psyms = [p for p in sm.parameters.values() if isinstance(p, sp.Symbol)]
    for s in list(sm.state) + list(sm.aux_state) + psyms:
        if isinstance(s, sp.Symbol) and str(s) not in seen:
            seen.add(str(s))
            out.append(f"  {s} : {canonical_true_facts(s)}")
    pv = getattr(sm, "parameter_values", None)
    if pv is not None and hasattr(pv, "keys"):
        out.append("parameter_values: {"
                   + ", ".join(f"{k}: {pv[k]!r}" for k in pv.keys()) + "}")
    out.append(f"n_equations: {sm.n_equations}  n_state: {sm.n_state}  "
               f"n_dim: {sm.n_dim}")
    eqn = getattr(sm, "equation_names", None)
    if eqn:
        out.append("equation_names: [" + ", ".join(map(str, eqn)) + "]")
    e2s = getattr(sm, "equation_to_state_index", None)
    if e2s is not None:
        out.append(f"equation_to_state_index: {list(e2s)}")
    out.append(f"stationary_indices: {sorted(getattr(sm, 'stationary_indices', []) or [])}")

    render_array("mass_matrix", sm.mass_matrix, out)
    render_array("flux", sm.flux, out)
    render_array("hydrostatic_pressure", sm.hydrostatic_pressure, out)
    render_array("diffusion_matrix", getattr(sm, "diffusion_matrix", None), out)
    render_array("nonconservative_matrix", sm.nonconservative_matrix, out)
    render_array("source", sm.source, out)
    render_array("source_explicit", getattr(sm, "source_explicit", None), out)
    render_array("eigenvalues", getattr(sm, "eigenvalues", None), out)
    render_array("eigenvalues_cfl", getattr(sm, "eigenvalues_cfl", None), out)
    render_array("update_variables", getattr(sm, "update_variables", None), out)
    render_array("update_aux_variables",
                 getattr(sm, "update_aux_variables", None), out)
    render_array("reconstruction_variables",
                 getattr(sm, "reconstruction_variables", None), out)
    render_array("state_from_reconstruction",
                 getattr(sm, "state_from_reconstruction", None), out)
    interp = getattr(sm, "interpolate_to_3d", None)
    if interp is not None and hasattr(interp, "keys"):
        render_mapping("interpolate_to_3d", interp, out)
    else:
        render_array("interpolate_to_3d", interp, out)
    proj = getattr(sm, "project_from_3d", None)
    if proj is not None and hasattr(proj, "keys"):
        render_mapping("project_from_3d", proj, out)
    else:
        render_array("project_from_3d", proj, out)

    specs = getattr(sm, "boundary_specs", None)
    if specs:
        out.append("boundary_specs:")
        for name in sorted(specs):
            spec = specs[name]
            if hasattr(spec, "items"):
                for k in sorted(spec.keys(), key=str):
                    out.append(f"  {name}: {k} -> {norm_expr(spec[k])}")
            else:
                out.append(f"  {name}: {spec!r}")
    else:
        out.append("boundary_specs: {}")
    bcsrc = getattr(sm, "_bc_source", None)
    tags = (sorted(b.tag for b in bcsrc.boundary_conditions_list)
            if bcsrc is not None and getattr(bcsrc, "boundary_conditions_list", None)
            else [])
    out.append(f"bc_tags: {tags}")
    kern = getattr(sm, "boundary_conditions", None)
    kdef = getattr(kern, "definition", None) if kern is not None else None
    if kdef is not None:
        try:
            out.append(f"bc_kernel: {norm_expr(kdef)}")
        except Exception:
            out.append(f"bc_kernel: <unrenderable> {type(kdef).__name__}")
    else:
        out.append("bc_kernel: None")

    reg = getattr(sm, "aux_registry", None)
    out.append("aux_registry:")
    for entry in (reg or []):
        if hasattr(entry, "items"):
            keys = sorted(entry.keys(), key=str)
            parts = []
            for k in keys:
                v = entry[k]
                parts.append(f"{k}={norm_expr(v)}" if isinstance(v, sp.Basic)
                             else f"{k}={v!r}")
            out.append("  {" + ", ".join(parts) + "}")
        else:
            out.append(f"  {entry!r}")
    pos = getattr(sm, "positive_state", None)
    if pos:
        out.append("positive_state: [" + ", ".join(norm_expr(s) for s in pos) + "]")
    return "\n".join(out) + "\n"


def render_nsm(nsm, *, title="NumericalSystemModel", with_numerics=True) -> str:
    """SystemModel snapshot + the numerical knobs + (optionally) the built
    face-kernel definitions (numerical_flux / numerical_fluctuations /
    local_max_abs_eigenvalue) — the REQ-189 lazy-Max pin lives in those."""
    out = [render_system_model(nsm, title=title)]
    k = ["== numerics =="]
    k.append(f"riemann: {getattr(nsm.riemann, '__name__', nsm.riemann)!s}")
    k.append(f"riemann_explicit: {nsm.riemann_explicit}")
    k.append(f"reconstruction: {nsm.reconstruction}")
    k.append(f"diffusion: {nsm.diffusion}")
    k.append(f"eigenvalue_eps: {nsm.eigenvalue_eps!r}")
    k.append(f"dt_max: {nsm.dt_max!r}")
    k.append(f"depth_regularization: {nsm.depth_regularization!r}")
    k.append(f"regularization_eps: {nsm.regularization_eps!r}")
    k.append(f"eigenvalue_treatment: {nsm.eigenvalue_treatment!r}")
    k.append(f"eigenvalue_guard: {nsm.eigenvalue_guard!r}")
    k.append(f"normalize_normal: {nsm.normalize_normal!r}")
    k.append(f"source_treatment: {nsm.source_treatment!r}")
    k.append("default_operations: "
             + str([type(op).__name__ if not hasattr(op, "__name__")
                    else op.__name__ for op in nsm.default_operations()]))
    k.append("extra_operations: "
             + str([type(op).__name__ if not hasattr(op, "__name__")
                    else op.__name__ for op in nsm.extra_operations]))
    out.append("\n".join(k) + "\n")
    if with_numerics:
        num = nsm.build_numerics()
        n = ["== built numerics =="]
        n.append(f"class: {type(num).__name__}")
        fns = getattr(num, "functions", {}) or {}
        for fname in sorted(fns, key=str):
            fn = fns[fname]
            definition = getattr(fn, "definition", None)
            if definition is None:
                n.append(f"{fname}: None")
                continue
            try:
                n.append(_render_kernel_definition(str(fname), definition))
            except Exception:
                n.append(f"{fname}: <unrenderable> {type(definition).__name__}")
        out.append("\n".join(n) + "\n")
    return "\n".join(out)


_KERNEL_SREPR_LIMIT = 20000


def _render_kernel_definition(fname: str, definition) -> str:
    """Face-kernel definition line: full normalized srepr when small; for the
    huge packed kernels (numerical_flux carries [flux | D+ | D- | lambda_max])
    a byte-pinning sha256 of the normalized srepr PLUS the structural atoms the
    REQ-189/167 pins need (lazy Max node, opaque eigenvalues kernel)."""
    import hashlib
    s = norm_expr(definition)
    e = sp.sympify(definition)
    has_max = bool(e.atoms(sp.Max))
    try:
        from zoomy_core.model.kernel_functions import eigenvalues as _eig
        has_eig = bool(e.atoms(_eig))
    except Exception:
        has_eig = any(getattr(a, "name", getattr(getattr(a, "func", None),
                                                 "__name__", ""))
                      == "eigenvalues" for a in e.atoms(sp.Function))
    tags = f" lazy_Max={has_max} opaque_eigenvalues={has_eig}"
    if len(s) <= _KERNEL_SREPR_LIMIT:
        return f"{fname}:{tags}\n{fname} = {s}"
    digest = hashlib.sha256(s.encode()).hexdigest()
    return (f"{fname}:{tags} srepr_len={len(s)} "
            f"srepr_sha256={digest}")


# ── golden file I/O ────────────────────────────────────────────────────────

def golden_path(name: str) -> pathlib.Path:
    return GOLDEN_DIR / f"{name}.txt"


def _header(name: str) -> str:
    return (f"{HEADER_PREFIX} golden: {name}\n"
            f"{HEADER_PREFIX} sympy: {sp.__version__}\n"
            f"{HEADER_PREFIX} regenerate: python scripts/regen_goldens.py\n"
            f"{HEADER_PREFIX} (header lines are informational; comparison is "
            f"on the body below)\n")


def _split_header(lines):
    """Split leading header lines (HEADER_PREFIX) from the body.  Only the
    LEADING block is header — a '#|' later in a body is body."""
    i = 0
    while i < len(lines) and lines[i].startswith(HEADER_PREFIX):
        i += 1
    return lines[:i], lines[i:]


def write_golden(name: str, body: str) -> pathlib.Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    p = golden_path(name)
    p.write_text(_header(name) + body)
    return p


def read_golden_body(name: str) -> str:
    p = golden_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"golden {name!r} missing at {p} — run scripts/regen_goldens.py "
            "and review/commit the diff")
    lines = p.read_text().splitlines(keepends=True)
    _header_lines, body = _split_header(lines)
    return "".join(body)


def assert_matches_golden(name: str, body: str) -> None:
    """Compare ``body`` against the checked-in golden; on mismatch show a
    unified diff head and the regen instruction."""
    expected = read_golden_body(name)
    if body == expected:
        return
    import difflib
    diff = list(difflib.unified_diff(
        expected.splitlines(), body.splitlines(),
        fromfile=f"goldens/{name}.txt", tofile="fresh", lineterm=""))
    head = "\n".join(diff[:80])
    more = max(0, len(diff) - 80)
    raise AssertionError(
        f"golden {name!r} drifted ({len(diff)} diff lines"
        + (f", first 80 shown; +{more} more" if more else "") + ").\n"
        + head
        + "\n\nIf the change is INTENDED: run scripts/regen_goldens.py, get the "
          "rederive tier green (re-bless protocol) and commit the diff.")


# ── model-golden builders (derive NO-CACHE by spec) ────────────────────────

def _std_closures():
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    return [Newtonian(), NavierSlip(), StressFree()]


def _swe_bcs(dim2: bool):
    """Model-derived wall (FromModel -> the registered ``boundary:wall``
    group — the G01 'BC kernels' pin) + extrapolation elsewhere."""
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, FromModel, Extrapolation)
    tags = (["left", "right", "top", "bottom"] if dim2 else ["left", "right"])
    bcs = ([FromModel(tag=tags[0], definition="wall")]
           + [Extrapolation(tag=t) for t in tags[1:]])
    return BoundaryConditions(bcs)


def _swe_model(dimension: int):
    """The DERIVED shallow-water model: SME(level=0) composition + Manning bed
    friction + BCs + IC declared on the model.  NEVER the hand-built SWE class
    (user mandate).  NOTE (reported, not hand-rolled): the wet/dry momentum cap
    (``update_variables``) and the Malpasset ``U_MAX`` cap exist only on the
    hand-built ``SWE``/``MalpassetSWE`` classes — no closure provides them for
    the derived SME(level=0), so these goldens pin the derived system WITHOUT
    the cap (spec gap flagged in the refactor report)."""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import (
        Newtonian, ManningFriction, StressFree)
    import zoomy_core.model.initial_conditions as IC
    model = SME(level=0, dimension=dimension,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=_swe_bcs(dimension == 3))
    model.initial_conditions = IC.Constant()
    return model


def build_m01_swe_1d():
    from zoomy_core.systemmodel.system_model import SystemModel
    with no_cache():
        model = _swe_model(2)
        sm = SystemModel.from_model(model)
    body = render_system_model(sm, title="SWE 1D == SME(level=0, dimension=2) "
                               "+ Manning + BCs + IC")
    assert sm.initial_conditions is model.initial_conditions
    return body


def build_m02_swe_2d():
    from zoomy_core.systemmodel.system_model import SystemModel
    with no_cache():
        model = _swe_model(3)
        sm = SystemModel.from_model(model)
    body = render_system_model(sm, title="SWE 2D == SME(level=0, dimension=3) "
                               "+ Manning + BCs + IC")
    assert sm.initial_conditions is model.initial_conditions
    return body


def _simple_model_golden(factory, title, unclosed=False):
    """Render one model's SystemModel snapshot.

    ``unclosed=True`` pins a model built WITHOUT a stress closure — the honest
    open system (m11, m16).  That is exactly what ``UnclosedClosureError``
    refuses to hand back by default, so the build is wrapped in the explicit
    ``allow_unclosed()`` opt-in; the golden keeps pinning the open operator
    shape, and the guard keeps firing everywhere it is not asked for.
    """
    from zoomy_core.systemmodel.system_model import SystemModel, allow_unclosed
    import contextlib
    import warnings
    with no_cache():
        with (allow_unclosed() if unclosed else contextlib.nullcontext()):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*unclosed term", category=UserWarning)
                sm = SystemModel.from_model(factory())
    return render_system_model(sm, title=title)


def build_m03_sme_l1():
    from zoomy_core.model.models import SME
    return _simple_model_golden(
        lambda: SME(level=1, dimension=2, closures=_std_closures()),
        "SME(level=1, dimension=2) + Newtonian/NavierSlip/StressFree")


def build_m04_sme_l2():
    from zoomy_core.model.models import SME
    return _simple_model_golden(
        lambda: SME(level=2, dimension=2, closures=_std_closures()),
        "SME(level=2, dimension=2) + Newtonian/NavierSlip/StressFree")


def build_m05_sme_2d():
    from zoomy_core.model.models import SME
    return _simple_model_golden(
        lambda: SME(level=1, dimension=3, closures=_std_closures()),
        "SME(level=1, dimension=3) — two horizontal directions")


def build_m06_elder_sme():
    from zoomy_core.model.models.turbulent_sme import ElderSME
    return _simple_model_golden(
        lambda: ElderSME(level=1, dimension=2),
        "ElderSME(level=1, dimension=2) — algebraic Elder closure")


def build_m07_sme_inplane():
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.equations import Stress
    return _simple_model_golden(
        lambda: SME(level=1, dimension=2, stress=Stress.full, closures=_std_closures()),
        "SME(level=1, dimension=2, stress=Stress.full) (the full Newtonian stress "
        "on the untransformed 3-D PDE)")


def build_m08_sme_bingham():
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Bingham, NavierSlip, StressFree
    return _simple_model_golden(
        lambda: SME(level=1, dimension=2, quadrature_order=8,
                    closures=[Bingham(), NavierSlip(), StressFree()]),
        "SME(level=1, dimension=2, quadrature_order=8) + Bingham "
        "(viscoplastic bulk; Gauss quadrature)")


def build_m09_mlswe():
    from zoomy_core.model.models.ml_swe import MLSWE
    return _simple_model_golden(
        lambda: MLSWE(n_layers=2, dimension=2, closures=_std_closures()),
        "MLSWE(n_layers=2, dimension=2) — upwind interface (both Piecewise "
        "branches)")


def build_m10_mlsme():
    from zoomy_core.model.models.ml_sme import MLSME
    return _simple_model_golden(
        lambda: MLSME(n_layers=2, level=1, dimension=2,
                      closures=_std_closures()),
        "MLSME(n_layers=2, level=1, dimension=2)")


def build_m11_vam_1d():
    from zoomy_core.model.models import VAM
    return _simple_model_golden(
        lambda: VAM(level=1, dimension=2),
        "VAM(level=1, dimension=2) — 8x8 square DAE (unclosed tau)",
        unclosed=True)


def vam_escalante_model():
    """The Escalante-bump configuration — base model for S02/N02/P01."""
    from zoomy_core.model.models import VAM
    return VAM(level=1, dimension=2, closures=_std_closures())


def build_m12_vam_escalante():
    return _simple_model_golden(
        vam_escalante_model,
        "VAM(level=1, dimension=2) Escalante-bump exact system "
        "(Newtonian/NavierSlip/StressFree; REQ-80 bed-slope flux routing)")


def build_m13_vam_2d():
    from zoomy_core.model.models import VAM
    return _simple_model_golden(
        lambda: VAM(level=1, dimension=3, closures=_std_closures()),
        "VAM(level=1, dimension=3) — P-modes couple all q_x/q_y/r rows")


def build_m14_vam_inplane():
    from zoomy_core.model.models import VAM
    from zoomy_core.model.models.equations import Stress
    return _simple_model_golden(
        lambda: VAM(level=1, dimension=2, stress=Stress.full, closures=_std_closures()),
        "VAM(level=1, dimension=2, stress=Stress.full) (REQ-176(4): retain "
        "form + live nu*q*db/dx topography couplings)")


def build_m15_mlvam():
    from zoomy_core.model.models.ml_vam import MLVAM
    return _simple_model_golden(
        lambda: MLVAM(n_layers=2, level=1, dimension=2,
                      closures=_std_closures()),
        "MLVAM(n_layers=2, level=1, dimension=2) — square DAE, zero mass "
        "rows for the pressure constraints")


def build_m16_sigma3d():
    from zoomy_core.model.models.sigma3d import Sigma3D
    return _simple_model_golden(
        lambda: Sigma3D(),
        "Sigma3D() — zeta as flux direction, zeta-zeta diffusion",
        unclosed=True)


# ── systemmodel goldens (warm cache allowed — they pin the SM layer) ───────

def _sme1_with_bcs_ic():
    from zoomy_core.model.models import SME
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, FromModel, Extrapolation)
    model = SME(level=1, dimension=2, closures=_std_closures(),
                boundary_conditions=BoundaryConditions(
                    [FromModel(tag="left", definition="wall"),
                     Extrapolation(tag="right")]))
    model.initial_conditions = IC.Constant()
    return model


def build_s01_sme_freeze():
    from zoomy_core.systemmodel.system_model import SystemModel
    model = _sme1_with_bcs_ic()
    sm = SystemModel.from_model(model)
    # REQ-87/103/154 identity pins (runtime asserts, not snapshot text):
    assert sm.initial_conditions is model.initial_conditions
    names = [str(a) for a in sm.aux_state]
    assert len(names) == len(set(names)), "doubled aux after BC attach"
    return render_system_model(
        sm, title="S01 from_model(SME(1)+BCs+IC) freeze")


def _chorin_split_bump():
    """S02: desingularize_hinv applied BEFORE the split (spec order)."""
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.systemmodel.operations import desingularize_hinv
    model = vam_escalante_model()
    sm = SystemModel.from_model(model)
    sm.apply(desingularize_hinv())
    split = model.chorin_split(system_model=sm)
    return sm, split


def render_split(split, title: str) -> str:
    out = [f"== {title} =="]
    stages = split.stages
    out.append("stages: " + str([(s.label, s.kind) for s in stages]))
    body = "\n".join(out) + "\n\n"
    for stage in stages:
        body += render_system_model(stage.sm, title=f"stage {stage.label}") + "\n"
    return body


def build_s02_chorin_split_bump():
    import sympy as _sp
    sm, split = _chorin_split_bump()
    # REQ-169: explicit elliptic Jacobian column/aux-width agreement.
    press = split.SM_press
    J = getattr(press, "source_jacobian_wrt_aux_variables", None)
    if J is not None:
        assert J.shape[1] == len(list(press.aux_state)), (
            f"REQ-169: J.shape[1]={J.shape[1]} != len(aux_state)="
            f"{len(list(press.aux_state))}")
    # dt reaches the corrector update.
    corr_syms = set().union(*[_sp.sympify(e).free_symbols
                              for e in _sp.flatten(split.SM_corr.update_variables)])
    assert any(str(s) == "dt" for s in corr_syms)
    return render_split(split, "S02 chorin_split(VAM(1) Escalante-bump), "
                        "desingularize_hinv BEFORE split")


def build_s03_chorin_split_mlvam():
    from zoomy_core.model.models.ml_vam import MLVAM
    from zoomy_core.systemmodel.system_model import SystemModel
    model = MLVAM(n_layers=2, level=1, dimension=2, closures=_std_closures())
    sm = SystemModel.from_model(model)
    split = model.chorin_split(system_model=sm)
    return render_split(split, "S03 chorin_split(MLVAM(2,1))")


# ── NSM goldens ────────────────────────────────────────────────────────────

def build_n01_swe2d_order2():
    """NSM defaults on the DERIVED 2-D shallow-water system with order-2
    reconstruction: defaults [normalize_face_normal, desingularize_hinv],
    KP hinv row, order-2 recon map routed through hinv, no wet_dry_eps leak."""
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    sm = SystemModel.from_model(_swe_model(3))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2)).derive()
    return render_nsm(nsm, title="N01 NSM(SWE 2D == SME(0,d3)) order-2 "
                      "defaults", with_numerics=False)


def build_n02_vam_nsm():
    """N02: the VAM NSM — lazy Max(|eigenvalues-kernel|) wavespeed (REQ-189
    structure, REQ-167 no-Gershgorin).  Carries the time canary (test-side)."""
    from zoomy_core.numerics import NumericalSystemModel
    from zoomy_core.systemmodel.system_model import SystemModel
    sm = SystemModel.from_model(vam_escalante_model())
    nsm = NumericalSystemModel.from_system_model(sm).derive()
    return render_nsm(nsm, title="N02 NSM(VAM(1) Escalante)",
                      with_numerics=True)


def build_n03_sme_gate_guard():
    """N03: opt-in gate/guard exactness — srepr-pinned conditional(h>eps,.,0)
    spectrum, Max floor idempotent (REQ-181); asserts both ABSENT from N01."""
    from zoomy_core.numerics import NumericalSystemModel
    from zoomy_core.systemmodel.operations import (
        gate_eigenvalues_dry, guard_eigenvalue_powers)
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.model.models import SME
    sm = SystemModel.from_model(
        SME(level=1, dimension=2, closures=_std_closures()))
    nsm = NumericalSystemModel.from_system_model(
        sm, extra_operations=[guard_eigenvalue_powers(),
                              gate_eigenvalues_dry()]).derive()
    return render_nsm(nsm, title="N03 NSM(SME(1)) + guard_eigenvalue_powers "
                      "+ gate_eigenvalues_dry (opt-in)", with_numerics=False)


# ── solver Procedure/Statement IR (Q01) ────────────────────────────────────
#
# The example procedures are shared by the golden (C text) and the execution
# tests (python builder), so both walkers are checked against ONE IR — a
# divergence between the two families shows up as a test failure, not as a
# backend surprise.

def solver_procedure_examples():
    """Three design-faithful solver blocks exercising all five statement
    kinds: ``solver_dt_pass`` (ForEachFace + an opaque ``eigenvalues`` call in
    expression position), ``solver_reduce_dt`` (IfStatic on the ``coupled``
    build flag — D7's dt_window/t_end exclusivity — plus Piecewise and a
    Min-reduction) and ``solver_march`` (the one While, with a post-loop
    ``solver_write_fields`` Call).

    Returns the UNRESOLVED procedures; ``solver_reduce_dt`` must be
    ``.resolve({"coupled": ...})``-d before either walker accepts it.
    """
    from zoomy_core.solver import Assign, Call, ForEachFace, IfStatic, Procedure, While

    f = sp.Symbol("f")
    lam_lo, lam_hi = sp.IndexedBase("lam_lo_f"), sp.IndexedBase("lam_hi_f")
    inradius = sp.IndexedBase("inradius_f")
    Q, Qaux, p_, nrm = sp.symbols("Q Qaux p face_normal")
    dt, dt_max, dt_window = sp.symbols("dt dt_max dt_window")
    lam, dt_f = sp.symbols("lam dt_f")
    time, t_end, iteration, i_snap = sp.symbols(
        "time t_end iteration i_snapshot")
    # CFL law: 1-D 0.9.  A literal here, never a runtime knob.
    cfl = sp.Rational(9, 10)
    eps = sp.Rational(1, 10**12)

    dt_pass = Procedure(
        name="solver_dt_pass",
        args=("variables", "aux_variables", "parameters", "face_cells",
              "face_normal", "n_faces", "lam_lo_f", "lam_hi_f"),
        doc=("v6 §1: dt is computed BEFORE the flux loop.  Stores the two "
             "wave-speed bounds per face; the eigenvalues slot stays an "
             "opaque backend kernel (_NON_RESOLVABLE)."),
        stmts=(
            ForEachFace("f", "n_faces", body=(
                Assign("c_own", sp.IndexedBase("face_cells")[2 * f],
                       ctype="int"),
                Assign("lam_lo_f[f]",
                       sp.Function("eigenvalues")(0, Q, Qaux, p_, nrm,
                                                  sp.Symbol("c_own")),
                       declare=False),
                Assign("lam_hi_f[f]",
                       sp.Function("eigenvalues")(1, Q, Qaux, p_, nrm,
                                                  sp.Symbol("c_own")),
                       declare=False),
            )),
        ),
    )

    reduce_dt = Procedure(
        name="solver_reduce_dt",
        args=("lam_lo_f", "lam_hi_f", "inradius_f", "n_faces", "dt_max",
              "dt_window"),
        returns="dt",
        doc=("D7: dt_window REPLACES the t_end clamp in coupled mode — never "
             "min-combined (that combination is a measured preCICE abort).  "
             "The choice is a BUILD-time IfStatic, not a runtime branch.  An "
             "all-dry domain yields dt = dt_max: zeroed lambda drops out of "
             "the min rather than spinning at dt = 0."),
        stmts=(
            IfStatic("coupled",
                     then=(Assign("dt", dt_window),),
                     otherwise=(Assign("dt", dt_max),)),
            ForEachFace("f", "n_faces", body=(
                Assign("lam", sp.Max(abs(lam_lo[f]), abs(lam_hi[f]))),
                Assign("dt_f", sp.Piecewise(
                    (dt_max, lam < eps),
                    (cfl * inradius[f] / sp.Max(lam, eps), True))),
                Assign("dt", sp.Min(dt, dt_f), declare=False),
            )),
        ),
    )

    march = Procedure(
        name="solver_march",
        args=("variables", "aux_variables", "t_end", "dt", "write_interval"),
        doc=("The ONE While in the design.  Carry = the march state; the "
             "final write is the post-loop shell block."),
        stmts=(
            Assign("time", sp.Integer(0)),
            Assign("iteration", sp.Integer(0), ctype="int"),
            Assign("i_snapshot", sp.Integer(0), ctype="int"),
            While(time < t_end,
                  carry=("time", "iteration", "i_snapshot"),
                  body=(
                      Assign("time", time + dt, declare=False),
                      Assign("iteration", iteration + 1, declare=False),
                      Assign("i_snapshot", sp.Piecewise(
                          (i_snap + 1,
                           time >= (i_snap + 1) * sp.Symbol("write_interval")),
                          (i_snap, True)), declare=False),
                  )),
            Call("solver_write_fields",
                 args=("variables", "aux_variables", "time", "i_snapshot")),
        ),
    )
    return {"solver_dt_pass": dt_pass, "solver_reduce_dt": reduce_dt,
            "solver_march": march}


def build_q01_procedure_c():
    """Q01: the C-family walker over the shared example procedures — both
    resolutions of the ``coupled`` build flag emitted, so the golden also
    pins that IfStatic really disappears (no runtime branch in either)."""
    from zoomy_core.transformation.procedure_c import CProcedurePrinter
    ex = solver_procedure_examples()
    printer = CProcedurePrinter()
    parts = ["== Q01 solver Procedure IR -> C family ==",
             printer.print_procedure(ex["solver_dt_pass"]).rstrip()]
    for coupled in (False, True):
        parts.append(f"-- solver_reduce_dt resolved with coupled={coupled} --")
        parts.append(printer.print_procedure(
            ex["solver_reduce_dt"].resolve({"coupled": coupled})).rstrip())
    parts.append(printer.print_procedure(ex["solver_march"]).rstrip())
    body = "\n\n".join(parts) + "\n"
    # The build flag must not survive into CODE (doc comments may name it):
    # an IfStatic is resolved away, never lowered to a runtime branch.
    code_lines = [ln for ln in body.splitlines()
                  if not ln.lstrip().startswith(("//", "--", "=="))]
    assert not any("coupled" in ln for ln in code_lines), (
        "a build-time branch leaked into C source")
    return body


# ── printer goldens (full emitted source, default options) ────────────────

def build_p01_foam_vam():
    """P01: foam <- the VAM NSM (N02 base).  Full emitted source: the
    SystemModel printer + the Numerics printer (REQ-187/183/185/190/81/91)."""
    from zoomy_core.numerics import NumericalSystemModel
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.transformation.to_openfoam import (
        FoamNumericsPrinter, FoamSystemModelPrinter)
    sm = SystemModel.from_model(vam_escalante_model())
    nsm = NumericalSystemModel.from_system_model(sm).derive()
    code_sm = FoamSystemModelPrinter(nsm).create_code()
    code_num = FoamNumericsPrinter(nsm.build_numerics()).create_code()
    body = ("== P01 foam emitted source (SystemModel printer) ==\n"
            + code_sm.rstrip() + "\n\n"
            "== P01 foam emitted source (Numerics printer) ==\n"
            + code_num.rstrip() + "\n")
    assert "Integral" not in body, "raw sympy Integral leaked into foam source"
    return body


def build_p02_generic_c_sme():
    """P02: generic-C / amrex path <- the SME NSM via GenericCppModel:
    REQ-81 P3_h column factor through CSE, scalarization, registry-derived
    arg lists, REQ-91 no Integral."""
    from zoomy_core.model.models import SME
    from zoomy_core.transformation.generic_c import GenericCppModel
    # the generic-C path needs a symbolic spectrum (numerical eigenvalues are
    # the default since 48ceef8)
    model = SME(level=1, dimension=2, closures=_std_closures(), symbolic_spectrum=True)
    code = GenericCppModel(model).create_code()
    body = ("== P02 generic-C emitted source (GenericCppModel <- SME(1)) ==\n"
            + code.rstrip() + "\n")
    assert "Integral" not in body, "raw sympy Integral leaked into C source"
    return body


# ── solver golden (X01) ────────────────────────────────────────────────────

def build_x01_numpy_wb_solver():
    """X01: end-to-end numpy solver path — SME(1) lake-at-rest over a bump
    bed, Audusse hook, PERIODIC wrap, order-1, 10 cells, 2 adaptive steps;
    WB <= 1e-11.  IC declared on the MODEL (threads REQ-103).  Folds the
    Bernoulli moving-equilibrium round-trip + exact-discharge assert
    (absorbs test_equilibrium_wb).  Golden pins the final state at 8 sig
    digits (the hard asserts live here, not in the snapshot)."""
    import numpy as np
    from zoomy_core.model.models import SME
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Periodic
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.mesh import BaseMesh
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel

    eta0, n_cells = 2.0, 10

    def bed(x):
        return 0.5 * np.exp(-x ** 2)

    def ic(xv):
        x = float(xv[0])
        b = bed(x)
        return np.array([b, eta0 - b, 0.0, 0.0])

    model = SME(level=1, equilibrium_reconstruction="audusse",
                closures=_std_closures(),
                boundary_conditions=BoundaryConditions(
                    [Periodic(tag="left", periodic_to_physical_tag="right"),
                     Periodic(tag="right", periodic_to_physical_tag="left")]))
    model.initial_conditions = IC.UserFunction(function=ic)   # IC on the MODEL
    sm = SystemModel.from_model(model)
    assert sm.initial_conditions is model.initial_conditions, (
        "REQ-103: model-declared IC must thread through from_model")
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))

    mesh = BaseMesh.create_1d(domain=(-5.0, 5.0), n_inner_cells=n_cells)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=1e9,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    solver.setup_simulation(mesh, nsm, write_output=False)
    dts = []
    for _ in range(2):
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_face_inradius,
            solver._sim_compute_max_abs_eigenvalue)
        solver.step(float(dt))
        dts.append(float(dt))
    Q = np.asarray(solver._sim_Q, float)[:, :n_cells]

    eta_dev = float(np.max(np.abs(Q[0] + Q[1] - eta0)))
    assert np.all(np.isfinite(Q))
    assert eta_dev < 1e-11, (
        f"Audusse hook must hold lake-at-rest through the periodic wrap "
        f"(max|eta-eta0| = {eta_dev:.3e})")

    # Folded Bernoulli moving-equilibrium round-trip (from test_equilibrium_wb):
    # single-signed sheared column reconstructs to its own bed as ~identity and
    # preserves the discharge q = h*alpha_0 EXACTLY on a shifted bed.
    from zoomy_core.fvm.bernoulli_wb import build_bernoulli_config, reconstruct
    from zoomy_core.model.boundary_conditions import Extrapolation
    sm2 = SystemModel.from_model(SME(
        level=2, closures=_std_closures(),
        boundary_conditions=BoundaryConditions(
            [Extrapolation(tag="left"), Extrapolation(tag="right")])))
    cfg = build_bernoulli_config(sm2, mode="bernoulli")
    h, b, a0, a1 = 1.5, 0.3, 0.6, 0.2
    Qb = np.array([[b], [h], [h * a0], [h * a1], [0.0]])
    Qs = reconstruct(Qb, np.array([b]), cfg)
    assert np.max(np.abs(Qs - Qb)) < 5e-4, "Bernoulli round-trip to own bed"
    Qs2 = reconstruct(Qb, np.array([b + 0.1]), cfg)
    assert abs(Qs2[2, 0] - Qb[2, 0]) < 1e-12, "discharge preserved exactly"

    out = ["== X01 numpy solver: SME(1) lake-at-rest, Audusse hook, periodic, "
           "order-1, 10 cells, 2 steps =="]
    out.append(f"state_names: {[str(s) for s in sm.state]}")
    out.append(f"dts: [{dts[0]:.8e}, {dts[1]:.8e}]")
    out.append("wb_max_eta_dev_lt_1e-11: True")
    out.append("bernoulli_roundtrip_ok: True")
    for i in range(Q.shape[0]):
        vals = ", ".join(f"{v:.8e}" for v in Q[i])
        out.append(f"Q[{i}] = [{vals}]")
    return "\n".join(out) + "\n"


# ── registry ───────────────────────────────────────────────────────────────

# name -> (builder, family, tier)   tier in {"gate", "t2-small", "t2-large"}
GOLDENS = {
    # model goldens — T1 gate members per the v3 delta
    "m01_swe_1d":          (build_m01_swe_1d, "model", "gate"),
    "m02_swe_2d":          (build_m02_swe_2d, "model", "gate"),
    "m03_sme_l1":          (build_m03_sme_l1, "model", "gate"),
    "m07_sme_inplane":     (build_m07_sme_inplane, "model", "gate"),
    "m09_mlswe":           (build_m09_mlswe, "model", "gate"),
    "m12_vam_escalante":   (build_m12_vam_escalante, "model", "gate"),
    # model goldens — T2
    "m04_sme_l2":          (build_m04_sme_l2, "model", "t2-small"),
    "m05_sme_2d":          (build_m05_sme_2d, "model", "t2-large"),
    "m06_elder_sme":       (build_m06_elder_sme, "model", "t2-small"),
    "m08_sme_bingham":     (build_m08_sme_bingham, "model", "t2-small"),
    "m10_mlsme":           (build_m10_mlsme, "model", "t2-small"),
    "m11_vam_1d":          (build_m11_vam_1d, "model", "t2-small"),
    "m13_vam_2d":          (build_m13_vam_2d, "model", "t2-large"),
    "m14_vam_inplane":     (build_m14_vam_inplane, "model", "t2-small"),
    "m15_mlvam":           (build_m15_mlvam, "model", "t2-small"),
    "m16_sigma3d":         (build_m16_sigma3d, "model", "t2-small"),
    # systemmodel goldens
    "s01_sme_freeze":      (build_s01_sme_freeze, "systemmodel", "gate"),
    "s02_chorin_split_bump": (build_s02_chorin_split_bump, "systemmodel", "gate"),
    "s03_chorin_split_mlvam": (build_s03_chorin_split_mlvam, "systemmodel",
                               "t2-small"),
    # NSM goldens
    "n01_swe2d_order2":    (build_n01_swe2d_order2, "nsm", "gate"),
    "n02_vam_nsm":         (build_n02_vam_nsm, "nsm", "gate"),
    "n03_sme_gate_guard":  (build_n03_sme_gate_guard, "nsm", "t2-small"),
    # printer goldens
    "p01_foam_vam":        (build_p01_foam_vam, "printer", "gate"),
    "p02_generic_c_sme":   (build_p02_generic_c_sme, "printer", "gate"),
    # solver-IR golden (the Procedure/Statement IR, C-family walker)
    "q01_procedure_c":     (build_q01_procedure_c, "procedure", "gate"),
    # solver golden
    "x01_numpy_wb_solver": (build_x01_numpy_wb_solver, "solver", "gate"),
}


def golden_params(family: str):
    """pytest.param list for one golden family — tier mapped to markers
    (gate -> gate+small, t2-small -> small, t2-large -> large).  Single
    source of truth for the golden test files."""
    import pytest
    tier_marks = {
        "gate":     lambda: [pytest.mark.gate, pytest.mark.small],
        "t2-small": lambda: [pytest.mark.small],
        "t2-large": lambda: [pytest.mark.large],
    }
    return [pytest.param(name, id=name, marks=tier_marks[tier]())
            for name, (_b, fam, tier) in GOLDENS.items() if fam == family]

TIME_BASELINE = GOLDEN_DIR / "n02_time_baseline.json"


def measure_n02_cpu_seconds() -> float:
    """CPU-time (ONE run, per the v3 ruling) of the N02 seam: NSM derive +
    numerics build + numpy opaque-kernel lowering on a warm-cached SM — the
    REQ-189 factor_terms-blowup canary."""
    import time
    from zoomy_core.numerics import NumericalSystemModel
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
    sm = SystemModel.from_model(vam_escalante_model())   # warm tier — untimed
    t0 = time.process_time()
    nsm = NumericalSystemModel.from_system_model(sm).derive()
    num = NonconservativeRusanov(nsm)
    num.to_runtime_numpy()
    return time.process_time() - t0


def write_time_baseline() -> dict:
    cpu = measure_n02_cpu_seconds()
    data = {"cpu_s": round(cpu, 3), "hostname": platform.node(),
            "cpu_model": platform.processor() or platform.machine(),
            "sympy": sp.__version__}
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BASELINE.write_text(json.dumps(data, indent=1) + "\n")
    return data
