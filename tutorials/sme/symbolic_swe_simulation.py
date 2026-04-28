"""1D dam-break with a SWE model whose flux() is READ from the symbolic
derivation (via ``SystemModel``) — no hand-coded ``q²/h + gh²/2`` anywhere.

For reference the script also runs the hand-coded ``SWEModelV2`` (the
classical Zoomy baseline) on the same initial condition and grid, and
plots both side by side.
"""
import sys
sys.path.insert(0, '/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/Zoomy-symbolic/library/zoomy_core')

import numpy as np
import sympy as sp
from sympy import Matrix

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.derivative_workflow import (
    DerivativeAwareSolver, StructuredDerivativeModel,
)

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Integrate, Inviscid, InterfaceKBC, Basis,
    SimplifyIntegrals, Recombine,
)
from zoomy_core.model.models.basisfunctions import LayeredBasis
from zoomy_core.model.models.sme_model import hydrostatic_scaling
from zoomy_core.model.models.system_model import SystemModel


# ---------------------------------------------------------------
# 1. Derive SWE symbolically and extract flux/NCP/source as sympy expressions
#    in symbols (h, hu, b, g).  These are the operators the solver will use.
# ---------------------------------------------------------------

def build_symbolic_swe_operators():
    state = StateSpace(dimension=2)  # (t, x, z)
    t, x, z = state.t, state.x, state.z
    basis = Basis(state, LayeredBasis, interfaces=[state.b, state.eta])

    model = FullINS(state)
    model.momentum.z.apply(hydrostatic_scaling(state)).simplify()
    model.momentum.z.apply(Integrate(z, z, state.eta, method='analytical'))
    model.momentum.z.apply({state.p.subs(z, state.eta): 0}).simplify()
    model.momentum.x.apply(model.momentum.z.solve_for(state.p))
    model.momentum.z.remove()
    model.apply(Inviscid(state)).simplify()

    swe = model.branch(name='SWE')
    swe.apply(Integrate(z, state.b, state.eta, method='auto'))
    swe.apply(InterfaceKBC(state, state.b)).simplify()
    swe.apply(InterfaceKBC(state, state.eta)).simplify()
    swe.apply(basis.layer_expand(state.u, 0)).simplify()
    swe.apply(SimplifyIntegrals(state)).simplify()

    alpha = basis.alpha.alpha_0
    hu_fn = sp.Function('hu', real=True)(t, x)
    swe.apply({alpha: hu_fn / state.H})
    swe.apply(Recombine(vars=[t, x]))

    h_sym = sp.Symbol('h', positive=True)
    hu_sym = sp.Symbol('hu', real=True)
    b_sym = sp.Symbol('b', real=True)
    g_sym = sp.Symbol('g', positive=True)

    sm = SystemModel(
        swe,
        state_substitutions={state.H: h_sym, hu_fn: hu_sym, state.b: b_sym},
        state_variables=[b_sym, h_sym, hu_sym],
        equation_variable={('continuity',): h_sym,
                           ('momentum', 'x'): hu_sym},
        time_var=t, coords=[x],
    )
    # Zoomy's sympy "g" parameter is also named g; but the derivation used
    # state.g which is Symbol("g", positive=True).  Same name, so flux already
    # uses g correctly.
    F = sm.flux()
    A = sm.nonconservative_matrix()
    S = sm.source()
    print('Extracted SWE operators (state order: [b, h, hu]):')
    print(f'  F[h , 0] = {sp.simplify(F[1, 0])}')
    print(f'  F[hu, 0] = {sp.simplify(F[2, 0])}')
    for (i, j) in [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        v = sp.simplify(A[i, j, 0])
        if v != 0:
            print(f'  A[{i}, {j}, 0] = {v}')
    remainders = sm.untagged_remainders()
    print(f'  untagged remainders: {remainders if remainders else "(none)"}')
    return F, A, h_sym, hu_sym, b_sym


# ---------------------------------------------------------------
# 2. Wrap the extracted operators in a StructuredDerivativeModel.
#    The model declares the same state as the hand-coded baseline
#    (variables = ["h", "hu"]) and builds flux/NCP by substituting
#    the symbolic operators with self.Q.* and self.params.*.
# ---------------------------------------------------------------

_F_sym, _A_sym, _H_PH, _HU_PH, _B_PH = build_symbolic_swe_operators()


class SymbolicSWE(StructuredDerivativeModel):
    """SWE whose flux and NCP are read from the symbolic derivation."""

    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def _symbol_to_runtime(self, expr):
        expr = sp.sympify(expr)
        return expr.subs({
            _H_PH: self.Q.h,
            _HU_PH: self.Q.hu,
            _B_PH: 0,                     # flat bed for this test
            sp.Symbol('g', positive=True): self.params.g,
        })

    def flux(self):
        F = Matrix.zeros(self.n_variables, self.dimension)
        # Extractor ordering was [b, h, hu]; we only evolve h and hu.
        F[0, 0] = self._symbol_to_runtime(_F_sym[1, 0])
        F[1, 0] = self._symbol_to_runtime(_F_sym[2, 0])
        return ZArray(F)

    def nonconservative_matrix(self):
        A = Matrix.zeros(self.n_variables, self.n_variables)
        # Map extractor (b, h, hu) rows/cols -> Zoomy (h, hu).  Drop b row/col.
        def _map_col(j):
            return {1: 0, 2: 1}.get(j, None)
        def _map_row(i):
            return {1: 0, 2: 1}.get(i, None)
        for i in (1, 2):
            for j in (1, 2):
                val = self._symbol_to_runtime(_A_sym[i, j, 0])
                if val != 0:
                    A[_map_row(i), _map_col(j)] = val
        # Wrap into the (n_vars, n_vars, dim) shape solvers expect.
        from sympy.tensor.array import MutableDenseNDimArray
        T = MutableDenseNDimArray.zeros(self.n_variables, self.n_variables, self.dimension)
        for r in range(self.n_variables):
            for c in range(self.n_variables):
                T[r, c, 0] = A[r, c]
        return ZArray(T)

    def source(self):
        return ZArray.zeros(self.n_variables)


class HandCodedSWE(StructuredDerivativeModel):
    """Classical hand-coded SWE baseline for cross-check."""
    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu**2 / h + 0.5 * g * h**2
        return ZArray(F)


# ---------------------------------------------------------------
# 3. Classic dam-break: h=2 left, h=1 right, u=0.  Evolve to t=0.1.
# ---------------------------------------------------------------

def ic_func(x):
    Q = np.zeros(2, dtype=float)
    Q[0] = 2.0 if x[0] < 5.0 else 1.0
    Q[1] = 0.0
    return Q


def make_model(cls):
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )
    return cls(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_func),
    )


def run(cls, label):
    print(f'--- running {label} ---')
    model = make_model(cls)
    mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=200)
    Q0 = np.empty((model.n_variables, mesh.n_cells), dtype=float)
    Q0 = model.initial_conditions.apply(mesh.cell_centers, Q0)
    settings = Zstruct(output=Zstruct(
        directory=f"/tmp/dambreak_{label.lower().replace(' ', '_')}",
        filename="dambreak", snapshots=2, clean_directory=True,
    ))
    solver = DerivativeAwareSolver(
        time_end=0.5,
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
    )
    Qnew, Qaux = solver.solve(mesh, model, write_output=False)
    h = Qnew[0]
    hu = Qnew[1]
    print(f'  h range: [{float(h.min()):.4f}, {float(h.max()):.4f}]')
    print(f'  hu range: [{float(hu.min()):.4f}, {float(hu.max()):.4f}]')
    return mesh.cell_centers[:, 0], h, hu


if __name__ == '__main__':
    x_sym, h_sym, hu_sym = run(SymbolicSWE, 'Symbolic SWE')
    x_hc, h_hc, hu_hc = run(HandCodedSWE, 'Hand-coded SWE')
    print(f'\ngrid sizes: symbolic={len(h_sym)}  hand-coded={len(h_hc)}')
    n = min(len(h_sym), len(h_hc))
    dh = float(np.abs(h_sym[:n] - h_hc[:n]).max())
    dhu = float(np.abs(hu_sym[:n] - hu_hc[:n]).max())
    print(f'\nmax |h_symbolic - h_handcoded|  = {dh:.3e}')
    print(f'max |hu_symbolic - hu_handcoded| = {dhu:.3e}')
    if dh < 1e-10 and dhu < 1e-10:
        print('  MATCH — symbolic operators reproduce hand-coded SWE bit-for-bit.')
    elif dh < 1e-4 and dhu < 1e-4:
        print('  CLOSE — small numerical divergence (likely path-conservative vs pure-flux split).')
    else:
        print('  DIVERGENT — investigate.')
