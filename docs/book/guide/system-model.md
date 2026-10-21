# SystemModel

A `SystemModel` freezes the model into the operator slots of one balance law:

$$
M\,\partial_t Q + \nabla\cdot\big(F + P\big) + \sum_d B_d\,\partial_d Q
= S + \nabla\cdot\big(A : \nabla Q\big).
$$

`SystemModel.from_model(model)` derives it, the solvers read the slots and never inspect the model.

| slot | role |
|---|---|
| `flux` $F$ | explicit |
| `hydrostatic_pressure` $P$ | explicit, well-balanced with the nonconservative term |
| `nonconservative_matrix` $B$ | explicit, path-conservative |
| `source` $S$ | implicit under IMEX, else part of the explicit step |
| `source_explicit` | always explicit |
| `diffusion_matrix` $A$ | implicit under IMEX, else part of the explicit step |
| `diffusion_matrix_explicit` | always explicit |
| `eigenvalues` | wave speeds, `None` for the numerical wave speed |
| `mass_matrix` $M$ | |

Spatial derivatives the operators need are aux variables, filled by the solver with least-squares gradients.

| you set | how |
|---|---|
| initial state | `initial_conditions` on the model or the SystemModel: `Constant`, `RP` (Riemann problem), `UserFunction`, `RestartFromHdf5` of `zoomy_core.model.initial_conditions` |
| aux initial state | `aux_initial_conditions` |
| parameter values | `parameter_values` |
| pressure split (Chorin) | `model.chorin_split(system_model=sm)` gives `SM_pred`, `SM_press`, `SM_corr` |

`sm.describe()` renders the slots.
Tutorial: {doc}`Model and SystemModel <../tutorials/ipynb/components/model_system_model>`.

API: {doc}`zoomy_core.systemmodel <../api/_autodoc2/zoomy_core/zoomy_core.systemmodel>`.
