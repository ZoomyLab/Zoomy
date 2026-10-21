"""Function hooks (ANALYSIS 67): a model's operators call `hook(name)`, the solver setting
`functions` implements it on NumPy and JAX, foam / AMReX refuse it.  A Manning friction written
as a hook reproduces the built-in one (explicit, IMEX, point-implicit with each
`HookDerivative` mode); a hook flux equal to the SWE flux reproduces it on JAX.
"""
import inspect

import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SWE
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.numerics.numerical_system_model import SourceTreatment
from zoomy_core.symsolver.hooks import HookDerivative, HookError, HookFunction, hook, hook_function
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.symsolver.numpy_backend.kernels import NumpyKernels
from zoomy_core.symsolver.settings import Imex
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

pytestmark = pytest.mark.small

G, N, EPS = 9.81, 0.03, 1e-12
LIN = SourceTreatment.linearized
SOLVERS = {"numpy": SymbolicHyperbolicSolverNumpy, "jax": SymbolicHyperbolicSolverJax}


def f(h, hu):
    return -G * N**2 * hu * (hu * hu + EPS) ** 0.5 / h ** (7 / 3)


def df(h, hu):
    m = (hu * hu + EPS) ** 0.5
    return 7 / 3 * G * N**2 * hu * m / h ** (10 / 3), -G * N**2 * (m + hu * hu / m) / h ** (7 / 3)


class SWEHook(SWE):
    def source(self):
        return ZArray([[0], [0], [hook("manning")(self.Q.h, self.Q.hu)]])


class SWEFlux(SWE):
    def flux(self):
        F = sp.Matrix(super().flux())
        F[2, 0] = hook("pflux")(self.Q.h, self.Q.hu)
        return ZArray(F)


def nsm(cls=SWE, treatment=SourceTreatment.explicit, n_m=N, dam=False, order=1):
    bcs = BoundaryConditions([FromModel(tag=t, definition="wall") for t in ("left", "right")])
    sm = SystemModel.from_model(cls(dimension=1, parameters={"n_m": n_m}, boundary_conditions=bcs))
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0 * x[0], 1 + (0.5 * (x[0] < 5) if dam else 0.1 * np.exp(-(x[0] - 5) ** 2)),
         (0 if dam else 0.5) + 0 * x[0]]))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order), source_treatment=treatment)


def run(backend, model, functions=None, **kw):
    s = SOLVERS[backend](time_end=0.5, cfl=0.9, functions=functions, **kw)
    Q0, A = s.setup(LSQMesh.create_1d(domain=(0, 10), n_inner_cells=50), model)
    return np.asarray(s.run(Q0, A)[1])


def test_no_hook_kernels_unchanged():
    """No hook: every NumPy kernel's generated code is the same with the hook machinery on."""
    a, b = NumpyKernels(nsm()), NumpyKernels(nsm(), functions={"unused": f})
    for name, k in vars(a).items():
        low = getattr(k, "low", None)
        if low is not None:
            assert inspect.getsource(low.fn) == inspect.getsource(getattr(b, name).low.fn)


@pytest.mark.parametrize("backend", SOLVERS)
@pytest.mark.parametrize("imex", [False, Imex.lie, Imex.ssp2])
def test_manning_hook_is_builtin(backend, imex):
    ref = run(backend, nsm(), imex=imex)
    assert np.abs(run(backend, nsm(SWEHook), {"manning": f}, imex=imex) - ref).max() <= 1e-13
    if backend == "numpy" and not imex:          # the friction is felt
        assert np.abs(run(backend, nsm(n_m=0.0)) - ref).max() > 1e-4


@pytest.mark.parametrize("backend", SOLVERS)
def test_point_implicit_derivative_modes(backend):
    """SourceTreatment.linearized prints dS/dQ: user partials, a SymPy Lambda, JAX AD = built-in;
    `zero` drops the friction Jacobian; NumPy refuses `auto` of a callable at setup."""
    ref = run(backend, nsm(treatment=LIN))
    user = run(backend, nsm(SWEHook, LIN), {"manning": HookFunction(f, derivative=df)})
    split = hook_function(derivative=[lambda h, hu: df(h, hu)[0], lambda h, hu: df(h, hu)[1]])
    h, hu = sp.symbols("h hu")
    lam = sp.Lambda((h, hu), -G * N**2 * hu * sp.sqrt(hu**2 + EPS) / h ** sp.Rational(7, 3))
    for q in (user, run(backend, nsm(SWEHook, LIN), {"manning": split(f)}),
              run(backend, nsm(SWEHook, LIN), {"manning": lam})):
        assert np.abs(q - ref).max() <= 1e-13
    zero = run(backend, nsm(SWEHook, LIN),
               {"manning": HookFunction(f, derivative=HookDerivative.zero)})
    assert np.isfinite(zero).all() and 1e-9 < np.abs(zero - ref).max() < 1e-5
    if backend == "jax":
        assert np.abs(run(backend, nsm(SWEHook, LIN), {"manning": f}) - user).max() <= 1e-13
    else:
        with pytest.raises(HookError, match="operator 'source'.*'manning'.*jax only"):
            run(backend, nsm(SWEHook, LIN), {"manning": f})


@pytest.mark.parametrize("order", [1, 2])
def test_flux_hook_jax(order):
    """A hook flux = the SWE flux; quasilinear matrix and eigenvalues by JAX AD."""
    ref = run("jax", nsm(dam=True, order=order))
    q = run("jax", nsm(SWEFlux, dam=True, order=order),
            {"pflux": lambda h, hu: hu * hu / h + 0.5 * G * h * h})
    assert np.abs(q - ref).max() <= 1e-13


@pytest.mark.parametrize("module, name", [("zoomy_foam.symsolver", "SymbolicHyperbolicSolverFoam"),
                                          ("zoomy_amrex.symsolver",
                                           "SymbolicHyperbolicSolverAmrex")])
def test_cxx_refuses_hooks(module, name, tmp_path):
    S = getattr(pytest.importorskip(module), name)
    mesh = LSQMesh.create_1d(domain=(0, 10), n_inner_cells=8)
    with pytest.raises(NotImplementedError, match=r"function 'manning' \(called by source, "):
        S(time_end=0.1).run(mesh, nsm(SWEHook), np.ones((3, 8)), tmp_path)
    with pytest.raises(NotImplementedError, match="cannot call Python functions"):
        S(time_end=0.1, functions={"manning": f}).run(mesh, nsm(), np.ones((3, 8)), tmp_path)


def test_functions_are_validated():
    with pytest.raises(ValueError, match="not an identifier"):
        SymbolicHyperbolicSolverNumpy(functions={"1x": f})
    with pytest.raises(TypeError, match="not callable"):
        SymbolicHyperbolicSolverNumpy(functions={"manning": 1.0})
    with pytest.raises(TypeError, match="HookDerivative.zero"):
        HookFunction(f, derivative="zero")
    with pytest.raises(ValueError, match="functions has no 'manning'; did you mean 'maning'"):
        run("numpy", nsm(SWEHook), {"maning": f})
