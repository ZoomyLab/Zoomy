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
| `eigenvalues` | `(n_eq, 1)` or `None` | `None`: numerical wave speed (power iteration) |
| `mass_matrix` $M$ | `(n_eq, n_state)` | |

Spatial derivatives the operators need are registered as aux variables
(`dhdx`, ...) and filled by the solver with least-squares gradients.

The C1 bump of the [quick start](quickstart.md), inspected
(`docs/book/examples/inspect_bump.py`):

```text
state       ['b', 'h', 'q_x_0', 'q_y_0']
aux_state   ['dq_x0dx', 'dhdx', 'dq_y0dy', 'dhdy', 'dbdx', 'dbdy', 'hinv']
parameters  {'g': 9.81, 'rho': 1.0, 'nu': 0.0, 'lambda_s': 0.0, 'e_x': 0.0, 'n': 0.0}
flux                       shape (4, 2), nonzero 4
hydrostatic_pressure       shape (4, 2), nonzero 4
nonconservative_matrix     shape (4, 4, 2), nonzero 4
source                     shape (4, 1), nonzero 4
source_explicit            None
diffusion_matrix           None
diffusion_matrix_explicit  None
eigenvalues                None
```

| you set | how |
|---|---|
| initial state | `sm.initial_conditions = IC.UserFunction(function=f)` (`zoomy_core.model.initial_conditions`; `f(x)` returns the state at a point) |
| initial aux | `sm.aux_initial_conditions` |
| parameter values | `sm.parameter_values` |
| pressure split (Chorin) | `VAM.chorin_split(dt, system_model=sm)` gives `SM_pred`, `SM_press`, `SM_corr` |

**Connects to:** `NumericalSystemModel.from_system_model(sm, ...)`
([NumericalSystemModel](numerical-system-model.md)).  `sm.describe()` renders
the slots.

API: {doc}`zoomy_core.systemmodel <../api/_apidoc_zoomy_core/zoomy_core.systemmodel>`.
