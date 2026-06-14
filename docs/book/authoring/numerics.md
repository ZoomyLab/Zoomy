# Numerics — printers, the NumPy runtime, and Zoomy's SymPy layer

Between a `SystemModel` (frozen operator form, see
[`system-model.md`](system-model.md)) and a running solver sits one thin
layer: code generation. A *printer* walks the operator matrices, resolves
Zoomy's SymPy extensions (opaque basis atoms, runtime limiters, field
handles), and emits either a callable Python runtime (NumPy) or a source
file (C++, GLSL, JS, OpenFOAM, AMReX, UFL). The Riemann-solver registry is
built on the same machinery — every numerical flux is a `SystemModel`-aware
symbolic object the printer compiles like any other operator.

## 0. NumericalSystemModel — the bundle a solver consumes

A solver does not consume a `SystemModel` directly. `NumericalSystemModel`
(`zoomy_core/numerics/numerical_system_model.py`) is its **numerical sibling**:
a read-only wrapper around the frozen `SystemModel` plus everything a solver
needs that is *not* symbolic-PDE-shaped. Build it by **selecting existing
specs** — never by editing a solver:

```python
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
nsm = NumericalSystemModel.from_system_model(
    sm,
    riemann=NonconservativeRusanov,                              # a Numerics class (§4)
    reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"),
    diffusion=DiffusionSpec(...),
    regularization=RegularizationSpec(eigenvalue_eps=1e-8),
)
```

It carries the Riemann choice, reconstruction/limiter, diffusion and
regularization specs, and any sub-system splits. So the full pipeline is
**`Model → SystemModel → NumericalSystemModel → Solver`**; the printers below are
the code-generation the runtime/solver uses to execute it.

## 1. Printers

### 1a. Inventory

All printer modules live under
`library/zoomy_core/zoomy_core/transformation/`:
`to_numpy.py` (`NumpyRuntimeModel`, lambdified NumPy runtime — the
reference execution path); `to_c.py` (`CppModel`, `CppNumerics`;
templated C++ header `template <typename T>`); `to_glsl.py`
(`GenericGlslBase`; WebGL2 GLSL ES 3.00 shaders with `out` array
parameters); `to_js.py` (`GenericJsBase`; allocation-free JavaScript
kernels for the browser solver); `to_openfoam.py` (`FoamModel`;
OpenFOAM solver kernels with `Foam::scalar` / `Foam::List`);
`to_amrex.py` (`AmrexModel`, `AmrexNumerics`; AMReX C++ using
`amrex::SmallMatrix` and GPU macros); `to_ufl.py` (`UFLRuntimeModel`;
Unified Form Language for Firedrake / FEniCSx).

The package `__init__.py` carries no public re-exports; printers are
imported by full module path. `helpers.py` is the shared substrate
(`regularize_denominator`, `substitute_sympy_attributes_with_symbol_matrix`).

### 1b. Base layer — `generic_c.py`

`GenericCppBase` (CXX11 SymPy printer subclass), `GenericCppModel`
(model-side header) and `GenericCppNumerics` (Riemann-side header) form
the canonical C-family pipeline. `OutParamCodePrinter` is the sibling
base for languages without array return values (GLSL, JS).

Class-level configuration: `_output_subdir` (subdir under
`settings.output.directory`), `_wrapper_name` (outer `struct` name),
`_is_template_class` (wrap in `template <typename T>`), `real_type`
(`"double"` / `"T"` / `"amrex::Real"`), `math_namespace` (`"std::"` /
`"amrex::Math::"`), `gpu_enabled` (emit `PORTABLE_FN` / CUDA
host-device macros).

Per-method generation hooks a subclass may override:
`_generate_signature_from_function`, `wrap_function_signature`,
`get_includes`, `get_simple_array_def`, `get_array_type` /
`get_array_declaration`, `format_accessor`, `format_assignment`,
`format_array_initialization`, `convert_expression_body`, `_print_Pow`,
`_print_Indexed`, `_print_Function`, `_print_Symbol`, `doprint`.

### 1c. Building a backend printer

Both patterns are consumed by `cls.write_code(target, settings)`.

**Pattern A — thin config wrapper.** When the target is templated C++
and the base behaviour is correct, only the configuration attributes
move. `library/zoomy_core/zoomy_core/transformation/to_c.py`:

```python
from zoomy_core.transformation.generic_c import (
    GenericCppModel, GenericCppNumerics,
)

class CppModel(GenericCppModel):
    """Configuration wrapper. Does not override generation logic."""
    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"
# CppNumerics(GenericCppNumerics) is the same shape on the Numerics side.
```

**Pattern B — mixin + override.** When the backend has structural
differences (different array type, namespacing, kernel decoration), put
them in a mixin and compose. `to_amrex.py:11`:

```python
class AmrexCore:
    """All AMReX-specific syntax rules. Mixed in with
    GenericCppModel or GenericCppNumerics."""
    def __init__(self, *args, **kwargs):
        self.real_type = "amrex::Real"
        self.math_namespace = "amrex::Math::"
        super().__init__(*args, **kwargs)
    # overrides every print hook + signature/wrapper hooks listed in 1b.

class AmrexModel(AmrexCore, GenericCppModel):
    _output_subdir = ".amrex_interface"
    _is_template_class = False
```

The signature override at `to_amrex.py:128` is representative — AMReX
kernels take `amrex::SmallMatrix const&` instead of `const T*` because
the runtime owns `SmallMatrix` value objects passed by const reference
rather than raw pointer buffers:

```python
def _generate_signature_from_function(self, func_obj):
    """Use const references instead of raw pointers."""
    decls = []
    for key, obj in func_obj.args.items():
        cpp_name = self.ARG_MAPPING.get(key, key)
        if cpp_name in ["Q", "Q_minus", "Q_plus"]:
            t_val = self.get_array_type((self.n_dof_q,))
            decls.append(f"{t_val} const& {cpp_name}")
        # Qaux, n, X, p, gradQ the same const-ref pattern;
        # scalars (time, dX, dt, dx, bc_idx) stay by value.
```

`AmrexNumerics(AmrexCore, GenericCppNumerics)` at `to_amrex.py:223`
is the same pattern on the Numerics side.

**Minimal custom printer:**

```python
class MyBackendModel(GenericCppModel):
    _output_subdir = ".mybackend_interface"
    _is_template_class = False

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.real_type = "myreal_t"
        self.math_namespace = "mb::"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        ret_type = self.get_array_type(shape)
        return (f"MB_KERNEL static {ret_type} {name}({args_str}) noexcept\n"
                f"{{\n{body_str}\n}}\n")
```

The contract: (1) consume a `SystemModel` (or a `Model`,
`SystemModel.from_model`-normalised); (2) emit a runtime object or a
source file via `cls.write_code(target, settings, filename=...)`;
(3) register output under a stable `_output_subdir`.

## 2. `NumpyRuntimeModel.from_system_model(sm)`

The lambdify endpoint at
`library/zoomy_core/zoomy_core/transformation/to_numpy.py:289`:

```python
@classmethod
def from_system_model(cls, sm, *, module=None, printer=None):
    """Build a runtime by lambdifying a SystemModel's stored matrices."""
```

The returned object exposes the operator surface as plain callables
taking `(Q, Qaux, p)` and returning NumPy arrays:
`flux` → `(n_eq, n_dim, n_cells)`; `nonconservative_matrix` →
`(n_eq, n_state, n_dim, n_cells)`; `source` / `source_explicit` →
`(n_eq, n_cells)`; `hydrostatic_pressure` → `(n_eq, n_dim, n_cells)`;
`mass_matrix` → `(n_eq, n_state, n_cells)`; `quasilinear_matrix` /
`source_jacobian` per [`system-model.md`](system-model.md);
`eigenvalues(Q, Qaux, p, n)` → `(n_eq, n_cells)`; and
`boundary_conditions(bc_idx, time, position, distance, Q, Qaux, p, n)`
returning the per-face state.

Each operator is compiled through `_lambdify_function`, which routes
through `_vectorize_expression` so constant entries broadcast cleanly
on full-grid arrays (the `zeros_like` / `ones_like` anchor trick).
`nonconservative_matrix` and `quasilinear_matrix` are emitted per-axis
slabs and `np.stack`'d along the trailing axis; `diffusion_matrix` is
rank-4 and stacked along the trailing two. `from_system_model` is the
right entry when the pipeline starts from a `SystemModel` that may not
have a backing `Model` (e.g. a splitter output); the
`NumpyRuntimeModel(model)` constructor remains preferred when the
model's full `functions` registry is required. This is what every NumPy
solver internally invokes — see
[`../backends/numpy.md`](../backends/numpy.md).

## 3. Zoomy's SymPy extensions

SymPy subclasses that look opaque to the printer but carry the
metadata the runtime needs to resolve them.

**`Basisfunction`** —
`library/zoomy_core/zoomy_core/model/models/basisfunctions.py:12`.
Opaque sympy `Function` subclasses with a `_basis` back-reference;
each `phi_k(ζ)` routes back to its basis cache for integration:

```python
self.phi_fn = type(symbol, (sympy.Function,),
                   {"_basis": self, "nargs": 2})
# Sum(amp(k) · basis.phi_fn(k, ζ), (k, 0, L))
```

`EvaluateIntegrals` walks an integrand, reads `func._basis` on every
opaque atom, and dispatches to that basis's `evaluate_integral`.
Distinct bases get distinct sympy classes via the `symbol` argument,
so mixing is automatic and registry-free.

**`limit()`** — `library/zoomy_core/zoomy_core/model/numerics.py:43`.
Inert `sympy.Function` wrapper the runtime resolves to a TVD slope
limiter (`minmod`, `venkatakrishnan`, `barth_jespersen`):

```python
from zoomy_core.model.numerics import limit
S[1, 0] = - 2 * P_1 * limit(sp.Derivative(b, x), sp.Symbol("minmod"))
```

`.doit()` does not strip it; `expose_aux_atoms` lifts each `limit(...)`
to a fresh aux symbol `{target}_{axes}__{scheme}`, and the runtime
aux-refresh applies the named limiter to the LSQ gradient before
evaluating the source.

**`FieldHandle`** —
`library/zoomy_core/zoomy_core/fvm/riemann_solvers.py:22`. Resolves a
named field (`"h"`, `"b"`, `"hinv"`) to either state `Q` or auxiliary
`Qaux` at construction time, exposing `minus` / `plus` / `state`
Symbols on the L / R / cell-centre side of a face. Riemann code reads
`h.access(qL, auxL)` and stays agnostic to whether bathymetry lives in
the state (legacy SWE) or `Qaux` (chain-DAE).

**`AffineProjection`** — FEM-style affine reference-element step
($z \in [b, b+h] \to \zeta \in [0, 1]$). Canonical name contract for
the FEM mapping operation; coord-flavoured aliases are not used.

**Regularization** — `(h + ε)^(-n)` rewrite for inverse h-powers near
wet/dry interfaces; declared at construction via the `regularization=`
keyword on the subclass (e.g. `HyperbolicSME(regularization=...)`).
The printer then sees bounded denominators only.

## 4. Riemann solver registry

`library/zoomy_core/zoomy_core/fvm/riemann_solvers.py:80`:

```python
class Numerics(param.Parameterized, SymbolicRegistrar):
    """Symbolic numerics over a SystemModel."""
    name  = param.String(default="NumericsV2")
    model = param.Parameter(default=None)
```

A `Numerics` consumes a `SystemModel` (a `Model` is normalised once via
`SystemModel.from_model`). It builds per-face placeholder vectors
(`Q_minus`, `Q_plus`, `Qaux_minus`, `Qaux_plus`, `flux_minus`,
`flux_plus`, `source_term`), auto-locates standard named fields via
`find_field("h" | "b" | "hinv")`, and registers `numerical_flux` and
`numerical_fluctuations` as symbolic functions the printer lambdifies
exactly like a model operator.

Built-ins: `Rusanov` (local Lax-Friedrichs; identity dissipation,
bed-row zeroed when `b` lives in `Q`); `PositiveRusanov`
(Audusse-Bristeau-Klein hydrostatic reconstruction on top); `HLL`
(Davis-bounds two-wave; falls back to LLF when `eigenvalues is None`);
`HLLC` (HLL plus contact / shear wave for the free-surface family;
requires `h`); `NonconservativeRusanov` (Rusanov with NCP fluctuation
viscosity from `quasilinear_matrix`). Higher compositions
(`PositiveHLL`, `PositiveNonconservativeRusanov`,
`PositiveNonconservativeHLL`, `QuasilinearRusanov`,
`PositiveQuasilinearRusanov`) live alongside.

**User extension point** — `SymbolicRegistrar.register_symbolic_function`
from `library/zoomy_core/zoomy_core/model/basefunction.py:46`:

```python
def register_symbolic_function(self, name, method_ref, sig_struct):
    """Register a symbolic function under `self.functions[name]`,
    install a `self.call[name]` proxy that emits an indexed function
    call (`Model<T>::name(...)` or `name(...)`)."""
```

`SymbolicRegistrar` is a mixin; both `Model` and `Numerics` inherit
it. A custom Riemann variant subclasses `Numerics`, overrides
`numerical_flux` (and optionally `numerical_fluctuations`), and the
parent `_initialize_functions` registers them — the printer picks them
up automatically:

```python
import sympy as sp
import param
from zoomy_core.fvm.riemann_solvers import Rusanov

class MyDampedRusanov(Rusanov):
    name = param.String(default="MyDampedRusanov")
    damping = param.Number(default=0.5)

    def numerical_flux(self):
        return sp.Float(self.damping) * super().numerical_flux()
```

For a truly custom callable used inside an operator expression (not a
Riemann flux), call `register_symbolic_function(name, method_ref, sig)`
directly on the model so the printer resolves the symbolic
`Function(name)(...)` atom against a registered kernel.

## Running example — SWE end-to-end

From `thesis/notebooks/modeling/swe/simple_swe_v2.py`
(`SWEModelV2` and `make_model()`):

```python
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from simple_swe_v2 import make_model

m  = make_model()                              # author's Model
sm = SystemModel.from_model(m)                 # freeze
rt = NumpyRuntimeModel.from_system_model(sm)   # lambdify

flux_arr = rt.flux(Q, Qaux, p)                 # (n_eq, n_dim, n_cells)
```

`rt.boundary_conditions`, `rt.aux_boundary_conditions`,
`rt.boundary_gradients`, `rt.eigenvalues` populate on the same object.
For the upstream derivation that produces `SystemModel.flux`, see
[`model.md`](model.md); for the operator tag catalogue and post-freeze
surface, [`system-model.md`](system-model.md). The SME + $\sigma$
thread that pages A and B use is the advanced reference for chain-DAE
printing.

## Authoring checklist

- After `SystemModel.from_model(...)`, no naked `Function` atoms remain
  in the operators — `expose_aux_atoms` has routed them all to
  `aux_state` (cross-check `sm.aux_registry` if a printer emits an
  unresolved symbol).
- Every custom callable used inside an operator expression is registered
  via `SymbolicRegistrar.register_symbolic_function(...)` on the
  originating object so the printer resolves the function atom to a
  real kernel.
- Inverse h-powers are regularised at the `Model` layer (e.g.
  `HyperbolicSME(regularization=...)`); never paper over
  division-by-zero downstream of the printer.
- Targeting a non-NumPy backend: subclass `GenericCppModel` /
  `GenericCppNumerics` (or `OutParamCodePrinter` for languages without
  array return values), set `_output_subdir`, override only the
  language-specific hooks, and validate that the emitted source
  compiles in the target environment.
- Cross-backend invariant: every printer consumes the *same*
  `SystemModel`. If a backend needs different operators, fix the
  derivation, not the printer.
