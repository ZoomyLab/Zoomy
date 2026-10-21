# SystemModel

A `SystemModel` freezes the model into operator slots of one balance law:

$$
M\,\partial_t Q + \nabla\cdot\big(F + P\big) + \sum_d B_d\,\partial_d Q
= S + \nabla\cdot\big(A : \nabla Q\big).
$$

**What you write:** usually nothing but `SystemModel.from_model(model)` and the
initial conditions.  The solvers read the slots; they never inspect the model.

| slot | shape | treatment |
|---|---|---|
| `flux` $F$ | `(n_eq, n_dim)` | explicit |
| `hydrostatic_pressure` $P$ | `(n_eq, n_dim)` | explicit (well-balanced with the NCP) |
| `nonconservative_matrix` $B$ | `(n_eq, n_state, n_dim)` | explicit, path-conservative |
| `source` $S$ | `(n_eq, 1)` | implicit slot: implicit under IMEX, else in the RK step |
| `source_explicit` | `(n_eq,)` | always explicit |
| `diffusion_matrix` $A$ | `(n_eq, n_state, n_dim, n_dim)` | implicit slot, as `source` |
| `diffusion_matrix_explicit` | same | always explicit |
| `eigenvalues` | `(n_eq, 1)` or `None` | `None`: numerical wave speed (the Gelfand upper bound of the spectral radius) |
| `mass_matrix` $M$ | `(n_eq, n_state)` | |

Spatial derivatives the operators need are registered as aux variables
(`dhdx`, ...) and filled by the solver with least-squares gradients.

The slots of a model are printed by `sm.describe()`: [Model and SystemModel](../tutorials/ipynb/components/model_system_model.ipynb).

| you set | how |
|---|---|
| initial state | `sm.initial_conditions = IC.UserFunction(function=f)` (`zoomy_core.model.initial_conditions`; `f(x)` returns the state at a point) |
| initial aux | `sm.aux_initial_conditions` |
| parameter values | `sm.parameter_values` |
| pressure split (Chorin) | `VAM.chorin_split(dt, system_model=sm)` gives `SM_pred`, `SM_press`, `SM_corr` |

**Connects to:** `NumericalSystemModel.from_system_model(sm, ...)`
([NumericalSystemModel](numerical-system-model.md)).  `sm.describe()` renders
the slots.

API: {doc}`zoomy_core.systemmodel <../api/_autodoc2/zoomy_core/zoomy_core.systemmodel>`.
