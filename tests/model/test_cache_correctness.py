"""E2 — derivation/SystemModel cache correctness (spec §1b/§2 rank 2).

Goldens derive NO-CACHE by spec, so the entire cache-staleness class (wrong-key
collision, poisoned hits, stale disk/prebuilt entries, MRO source edits) is the
worst silent-wrong-physics bug and fully unguarded otherwise (REQ-163/188).

Merged from: test_derivation_cache.py + test_derivation_cache_keys.py +
test_sm_cache_req163.py + test_sm_cache_class_source_req188.py.

Small tier: key sensitivity, hit semantics, param-value sweep.
Rederive tier (same file): golden-vs-fresh srepr identity, subprocess
MRO-source re-bust, cross-process disk hit + prebuilt, env knobs.
"""
import pickle
import subprocess
import sys
import textwrap

import pytest
import sympy as sp

from zoomy_core.model.derivation.cache_keys import (
    cache_key, model_spec_key, op_sequence_key)
from zoomy_core.model.derivation.derivation_cache import derivation_cache
from zoomy_core.model.derivation.model import ResolveModes
from zoomy_core.model.derivation.projection import ExpandSums, EvaluateSums
from zoomy_core.model.derivation.operations import ChangeOfVariables
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.vam import VAM
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree)

pytestmark = [pytest.mark.model]

_h = sp.Function("h")(sp.Symbol("t"), sp.Symbol("x"))


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOMY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD", raising=False)
    # These are the CACHE tests, so they enable the cache EXPLICITLY rather
    # than inheriting the ambient default (OFF during model development).
    # A cache test that relies on the default is testing the default.
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "1")
    from zoomy_core.systemmodel import sm_cache
    sm_cache._MEMORY.clear()
    return tmp_path


# ── small: key sensitivity ──────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.gate
def test_key_sensitivity():
    """Keys are deterministic, cosmetic-insensitive, content-sensitive
    (op params / op order / model level / dimension / class), lambda CoVs are
    keyed by SYMBOLIC effect, and an exec-defined class is UNCACHEABLE
    (key None; fetch/store silent no-op) — never fingerprint-less staleness."""
    # deterministic + cosmetic-insensitive
    a = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    b = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    assert cache_key(a) == cache_key(b) and len(cache_key(a)) == 64
    assert cache_key(a) == cache_key(
        ResolveModes(index=sp.Symbol("k"), modes=[0, 1], name="renamed"))
    # content-sensitive: op params
    assert cache_key(a) != cache_key(
        ResolveModes(index=sp.Symbol("k"), modes=[0, 1, 2]))
    assert cache_key(a) != cache_key(
        ResolveModes(index=sp.Symbol("l"), modes=[0, 1]))
    # lambda CoV keyed by symbolic effect
    c1 = ChangeOfVariables("U", "q", lambda qi: qi / _h)
    c2 = ChangeOfVariables("U", "q", lambda qi: qi / _h)
    c3 = ChangeOfVariables("U", "q", lambda qi: qi / (2 * _h))
    assert cache_key(c1) == cache_key(c2)
    assert cache_key(c1) != cache_key(c3)
    # op ORDER matters
    assert op_sequence_key([ExpandSums(), EvaluateSums()]) \
        != op_sequence_key([EvaluateSums(), ExpandSums()])
    # model spec key: level / dimension / class sensitive, instance-insensitive
    assert model_spec_key(VAM(level=1, dimension=3)) \
        == model_spec_key(VAM(level=1, dimension=3))
    assert model_spec_key(VAM(level=1, dimension=3)) \
        != model_spec_key(VAM(level=2, dimension=3))
    assert model_spec_key(VAM(level=1, dimension=2)) \
        != model_spec_key(VAM(level=1, dimension=3))
    assert model_spec_key(VAM(level=1, dimension=3)) \
        != model_spec_key(SME(level=1, dimension=3))
    # BC/IC excluded by default, included on request
    from zoomy_core.model.boundary_conditions import Wall
    plain = VAM(level=1, dimension=3)
    walled = VAM(level=1, dimension=3,
                 boundary_conditions=[Wall("wall", on="momentum")])
    assert model_spec_key(plain) == model_spec_key(walled)
    assert model_spec_key(plain, include_bc_ic=True) \
        != model_spec_key(walled, include_bc_ic=True)
    # exec-defined class -> key None -> silent no-op (REQ-188 getsource guard)
    from zoomy_core.systemmodel import sm_cache
    ns = {}
    exec("from zoomy_core.model.models.swe import SWE\n"
         "class M(SWE):\n"
         "    def source(self):\n"
         "        return super().source()\n", ns)
    inst = ns["M"](dimension=1)
    assert sm_cache.cache_key(inst, lambda m: m) is None
    assert sm_cache.fetch(None) is None
    sm_cache.store(None, object())      # must be a silent no-op


# ── small: hit semantics ────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.gate
def test_hit_semantics(cache_dir):
    """A hit SKIPS the body (no ResolveModes re-run); changed args / version
    namespace miss; verify=True re-runs and asserts identity; and the sm_cache
    tier returns a FRESH MUTABLE copy — mutating a served SystemModel never
    poisons the next fetch."""
    calls = {"n": 0}

    @derivation_cache
    def stage(spec, basis, level):
        calls["n"] += 1
        return f"built:{spec}:{basis}:{level}"

    r1 = stage("vam", "legendre", 1)
    assert calls["n"] == 1 and stage.stats.misses == 1
    r2 = stage("vam", "legendre", 1)
    assert calls["n"] == 1, "2nd identical call must NOT re-run the body"
    assert stage.stats.hits == 1 and r1 is r2
    stage("vam", "legendre", 2)          # changed arg -> miss
    assert calls["n"] == 2

    # declarative-model args key on spec, not instance identity
    calls2 = {"n": 0}

    @derivation_cache
    def build(model):
        calls2["n"] += 1
        return (model.level, model.dimension)

    build(VAM(level=1, dimension=3))
    build(VAM(level=1, dimension=3))
    assert calls2["n"] == 1
    build(VAM(level=2, dimension=3))
    assert calls2["n"] == 2

    # verify=True re-runs the body and asserts srepr identity
    calls3 = {"n": 0}

    @derivation_cache(verify=True)
    def vstage(x):
        calls3["n"] += 1
        return sp.Symbol("a") * x

    vstage(sp.Integer(2))
    vstage(sp.Integer(2))
    assert calls3["n"] == 2

    # sm_cache: a hit returns a fresh mutable copy (no poisoning)
    a = SystemModel.from_model(SWE(dimension=1))
    ref = str(a.flux)
    a.flux = None                        # mutate the served object
    b = SystemModel.from_model(SWE(dimension=1))
    assert str(b.flux) == ref, "cache poisoned by mutating a served copy"


# ── small: parameter VALUES are runtime inputs, not identity ────────────────

@pytest.mark.small
@pytest.mark.gate
def test_param_value_sweep_hits(cache_dir):
    """Two instances differing ONLY in a parameter VALUE share one build:
    values are runtime solver inputs, not derivation identity."""
    a = SystemModel.from_model(SWE(dimension=1, parameters={"g": 9.81, "n_m": 0.0}))
    n_entries = len(list((cache_dir / "systemmodels").glob("*.pkl")))
    b = SystemModel.from_model(SWE(dimension=1, parameters={"g": 1.23, "n_m": 0.7}))
    assert len(list((cache_dir / "systemmodels").glob("*.pkl"))) == n_entries
    assert str(a.flux) == str(b.flux)
    assert float(b.parameter_values["g"] if hasattr(b.parameter_values, "__getitem__")
                 else b.parameter_values.g) == 1.23


# ── rederive tier: the cache-truth checks ───────────────────────────────────

@pytest.mark.rederive
def test_rederive_cached_equals_fresh_srepr():
    """srepr identity: the warm-cache VAM SystemModel operator content equals a
    fresh NO-CACHE derivation — the designed answer to 'goldens detect change,
    not wrongness' for the cache tiers."""
    import goldenlib
    _c = lambda: [Newtonian(), NavierSlip(), StressFree()]
    cached = SystemModel.from_model(VAM(level=1, dimension=2, closures=_c()))
    with goldenlib.no_cache():
        fresh = SystemModel.from_model(
            VAM(level=1, dimension=2, closures=_c()))
    for op in ("flux", "source", "nonconservative_matrix", "eigenvalues"):
        c, f = getattr(cached, op, None), getattr(fresh, op, None)
        if c is None or f is None:
            assert c is None and f is None, f"{op}: None mismatch"
            continue
        assert sp.srepr(sp.Matrix(sp.flatten(c))) == \
            sp.srepr(sp.Matrix(sp.flatten(f))), f"{op}: cached != fresh"


_DERIVE = """
import os, sys
os.environ['ZOOMY_CACHE_DIR'] = {cache!r}
os.environ.pop('ZOOMY_DERIVATION_REBUILD', None)
os.environ.pop('ZOOMY_DERIVATION_CACHE', None)
sys.path.insert(0, {moddir!r})
from mod_req188 import M
from zoomy_core.systemmodel.system_model import SystemModel
sm = SystemModel.from_model(M(dimension=1))
print('SOURCE=' + str(sm.source))
"""


def _write_model(path, marker: int):
    path.write_text(textwrap.dedent(f"""
        import sympy as sp
        from zoomy_core.model.models.swe import SWE
        from zoomy_core.model.basefunction import ZArray

        class M(SWE):
            def source(self):
                S = sp.Matrix.zeros(self.n_variables, 1)
                S[0, 0] = sp.Integer({marker})   # rainfall-like mass source
                return ZArray(S)
    """))


def _derive_in_subprocess(cache_dir, mod_dir):
    code = _DERIVE.format(cache=str(cache_dir), moddir=str(mod_dir))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("SOURCE="))
    return line[len("SOURCE="):]


@pytest.mark.rederive
def test_rederive_mro_source_rebust(tmp_path):
    """REQ-188: editing an overridden operator body (same class, same
    derive_model, same spec) busts the on-disk entry in a FRESH process."""
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    cache_dir = tmp_path / "cache"
    mod_file = mod_dir / "mod_req188.py"

    _write_model(mod_file, marker=7)
    first = _derive_in_subprocess(cache_dir, mod_dir)
    assert "7" in first and "13" not in first, first

    _write_model(mod_file, marker=13)
    second = _derive_in_subprocess(cache_dir, mod_dir)
    assert "13" in second, (
        "stale cache: edited source() body not reflected after re-derive "
        f"(got {second!r}, expected the new constant 13)")
    assert first != second
    # the MRO fingerprint chain is what makes this work
    from zoomy_core.systemmodel import sm_cache
    base = [f"{c.__module__}.{c.__qualname__}" for c in sm_cache._mro_classes(SWE)]
    assert base[0] == "zoomy_core.model.models.swe.SWE"
    assert "zoomy_core.model.basemodel.Model" in base
    assert sm_cache._mro_source_terms(SWE) is not None


@pytest.mark.rederive
def test_rederive_cross_process_disk_hit_and_prebuilt(cache_dir):
    """REQ-163: a SECOND python process with the same spec loads from disk,
    and the shipped ``_prebuilt`` tier is non-empty and unpicklable-sane."""
    SystemModel.from_model(SME(level=0, dimension=2,
                               closures=[Newtonian(), NavierSlip(),
                                         StressFree()]))
    code = (
        "import os, time\n"
        f"os.environ['ZOOMY_CACHE_DIR'] = {str(cache_dir)!r}\n"
        "from zoomy_core.model.models.sme import SME\n"
        "from zoomy_core.systemmodel.system_model import SystemModel\n"
        "t0 = time.time()\n"
        "from zoomy_core.model.models.closures import (\n"
        "    Newtonian, NavierSlip, StressFree)\n"
        "sm = SystemModel.from_model(SME(level=0, dimension=2, parameters={'g': 777.0},\n"
        "    closures=[Newtonian(), NavierSlip(), StressFree()]))\n"
        "dt = time.time() - t0\n"
        "assert sm.n_equations > 0\n"
        "assert dt < 2.0, f'expected disk hit, took {dt:.2f}s'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    from zoomy_core.systemmodel import sm_cache
    entries = sorted(sm_cache._prebuilt_dir().glob("*.pkl"))
    assert len(entries) >= 10, "prebuilt cache missing - run build_prebuilt_cache"
    for entry in entries:             # every shipped entry, not whichever glob lists first
        sm = pickle.loads(entry.read_bytes())
        assert hasattr(sm, "flux") and hasattr(sm, "state"), entry.name


@pytest.mark.small
def test_boundary_conditions_pickle_holds_no_param_internals():
    """A pickled `BoundaryConditions` carries no `param` Parameter objects: their slots differ
    between `param` versions (`default_factory` from 2.3), so the shipped `_prebuilt`
    `SystemModel` pickles would load only under the `param` that wrote them.
    """
    import pickletools
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, Dirichlet, Extrapolation)

    bcs = BoundaryConditions([Extrapolation(tag="b"), Dirichlet(tag="a", value=2.0)])
    blob = pickle.dumps(bcs)
    seen, globals_ = [], set()
    for op, arg, _ in pickletools.genops(blob):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            seen.append(arg)
        elif op.name == "STACK_GLOBAL":
            globals_.add((seen[-2], seen[-1]))
    assert not [g for g in globals_ if g[0] == "param.parameters"], globals_
    back = pickle.loads(blob)
    assert back.list_sorted_function_names == ["a", "b"]
    assert len(back._boundary_functions) == len(back._boundary_gradient_functions) == 2


@pytest.mark.rederive
def test_rederive_env_knobs(cache_dir, monkeypatch):
    """ZOOMY_DERIVATION_REBUILD forces a fresh build + user-tier write;
    ZOOMY_DERIVATION_CACHE=0 disables reads."""
    from zoomy_core.systemmodel import sm_cache
    monkeypatch.setenv("ZOOMY_DERIVATION_REBUILD", "1")
    SystemModel.from_model(SWE(dimension=1))
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD")
    entries = [f.stem for f in (cache_dir / "systemmodels").glob("*.pkl")]
    assert entries, "rebuild must write the user tier"
    key = entries[0]
    assert sm_cache.fetch(key) is not None
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "0")
    sm_cache._MEMORY.clear()
    assert sm_cache.fetch(key) is None          # reads disabled
    monkeypatch.delenv("ZOOMY_DERIVATION_CACHE")
    monkeypatch.setenv("ZOOMY_DERIVATION_REBUILD", "1")
    assert sm_cache.fetch(key) is None          # rebuild bypasses reads
