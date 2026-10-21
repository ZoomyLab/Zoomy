# Multilayer VAM

`MLVAM` is a non-hydrostatic model for dispersive free-surface flow.
The depth is split into `n_layers` layers that move with the free surface, and each layer is a vertically averaged and moment (VAM) column: horizontal velocity, vertical velocity and non-hydrostatic pressure as polynomials in the layer.
Papers: the VAM column of [Escalante, Morales de Luna, Cantero-Chinchilla and Castro-Orgaz 2024](https://doi.org/10.1016/j.jcp.2024.112882), the layer decomposition with interface mass transfer of [Escalante, Fernández-Nieto, Morales de Luna and Castro 2019](https://doi.org/10.1007/s10915-018-0849-9).

## Starting point

The incompressible Navier-Stokes system in $(x, z)$ for velocity $(u, w)$, total pressure $p_T$ and deviatoric stress $\sigma$:

<!-- Escalante et al. 2024, eq. (1) -->
$$
\begin{cases}
\partial_x u + \partial_z w = 0, \\
\partial_t u + u \partial_x u + w \partial_z u + \partial_x p_T = \partial_x \sigma_{xx} + \partial_z \sigma_{xz}, \\
\partial_t w + u \partial_x w + w \partial_z w + \partial_z p_T = -g + \partial_x \sigma_{zx} + \partial_z \sigma_{zz}
\end{cases}
$$

with the stress at the free surface $\eta = h + b$ and at the bed $b$, the hydrostatic and non-hydrostatic pressure, and the kinematic conditions

<!-- eq. (2) and Sect. 2 -->
$$
\sigma_{xz}|_{z=\eta} = 0, \quad \sigma_{xz}|_{z=b} = \tau, \qquad
p_T = p_H + p, \quad \partial_z p_H = -g, \qquad
u|_{z=b} \partial_x b = w|_{z=b}, \quad \partial_t \eta + u|_{z=\eta} \partial_x \eta = w|_{z=\eta}
$$

The depth is split into layers of fixed fractions, which exchange mass across the interface (two layers in the paper):

<!-- Escalante et al. 2019, Sect. 2, eq. (10a), (10b), (11) -->
$$
h_\alpha = l_\alpha h, \quad l_1 + l_2 = 1, \qquad
\begin{cases}
\partial_t h_1 + \partial_x (h_1 u_1) = -\Gamma_I, \\
\partial_t h_2 + \partial_x (h_2 u_2) = \Gamma_I,
\end{cases} \qquad
\Gamma_I = l_1 \partial_x (h_2 u_2) - l_2 \partial_x (h_1 u_1)
$$

In each layer, on the Legendre basis $\phi_j$ of the layer coordinate $\xi \in [0, 1]$ with $\phi_j(0) = 1$:

<!-- Escalante et al. 2024, Sect. 2 -->
$$
u = u_0\phi_0 + u_1\phi_1, \quad w = w_0\phi_0 + w_1\phi_1 + w_2\phi_2, \quad p = p_0\phi_0 + p_1\phi_1 + p_2\phi_2
$$

The equations are the mass and momentum balances tested with $\phi_j$ and integrated over the layer.

## Assumptions

- The in-plane viscous stresses are dropped, only $\sigma_{xz}$ is kept.
- Per layer $u \in \mathbb{P}_1$ and $w, p \in \mathbb{P}_2$. Zoomy generalizes this to $u \in \mathbb{P}_N$, $w, p \in \mathbb{P}_{N+1}$ with $N$ = `level`.
- The non-hydrostatic pressure vanishes at the free surface.
- The layers are fixed fractions of the depth. Zoomy takes any number of layers and keeps the pressure continuous across an interface, the paper allows a jump (its (16)).
- The interface velocity of the exchange term is the mean of the two layers (the paper's choice, Zoomy's default `MeanInterface`).

## Generated equations

```{include} ../../../_generated/model_mlvam.md
```

Notation: layer $\ell$ has depth $h_\ell = l_\ell h$ (parameter `l_1`, $l_2 = 1 - l_1$), $q_{\ell,k} = (-1)^k h_\ell u_{\ell,k}$, $r_{\ell,k} = (-1)^k h_\ell w_{\ell,k}$, $P_{\ell,k} = (-1)^k \rho p_{\ell,k}$ (Zoomy's basis has $\phi_k(1) = 1$).
The top modes $w_{\ell,N+1}$ are closed upward from the bed kinematic condition, $p_{\ell,N+1}$ downward from $p = 0$ at the surface, the interface mass flux by the layer fractions.
`MLVAM(n_layers=1, level=1)` is the paper's VAM system (4) with the constraints (5) row by row, inviscid and with $e_x = 0$: the row of $q_{1,1}$, $r_{1,1}$ is $-1$ times the paper's, the constraint rows are $-1$ times (5).

## Options

| setting | |
|---|---|
| `n_layers` | number of layers |
| `level` | $N$: $u \in \mathbb{P}_N$, $w, p \in \mathbb{P}_{N+1}$ per layer |
| `dimension` | `2`: (x, z). `3`: (x, y, z), state `q_x_l_k`, `q_y_l_k` |
| `closures` | stress and interface pieces, for example `[Newtonian(), NavierSlip(), StressFree()]`, plus `UpwindInterface()` for an upwinded interface velocity |
| `stress` | `Stress.shallow` the vertical shear, `Stress.full` the whole Newtonian tensor |
| `vertical_traction` | with `Stress.full`: the interface trace of the shear in the vertical rows (`VerticalTraction.shared`, `one_sided`, `function`) |
| `parameters` | `g`, `rho`, `nu`, `lambda_s`, `e_x`, layer fractions `l_1`, ..., `l_{n_layers-1}` |

The non-hydrostatic pressure makes the system a DAE: `chorin_split` gives the predictor, pressure and corrector systems of the Chorin solver.

```python
from zoomy_core.model.models import MLVAM, Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel import SystemModel

mlvam = MLVAM(n_layers=2, level=1, dimension=2,
              closures=[Newtonian(), NavierSlip(), StressFree()])
model = SystemModel.from_model(mlvam)
split = mlvam.chorin_split(system_model=model)
```

## In the GUI

There is no MLVAM card.
Compose, Model card **VAM**, then in the `## Model` section of `case.py` import `MLVAM` instead of `VAM` and replace `VAM(` by `MLVAM(n_layers=2, `.
Run with the **Split / Chorin (NumPy)** solver, which takes the card's `split`.
