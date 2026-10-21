# Shallow moment equations

`SME` resolves the vertical profile of the horizontal velocity of a shallow free-surface flow as a polynomial: the depth mean plus `level` moments.
`level=0` is shallow water.
Paper: [Kowalski and Torrilhon 2019, Moment approximations and model cascades for shallow flow](https://doi.org/10.4208/cicp.OA-2017-0263).

## Starting point

The reduced balance laws for velocity $(u, v, w)$, pressure $p$ and deviatoric stress $\sigma$, constant density $\rho$, gravity $g(e_x, e_y, e_z)$, bed $h_b$ and free surface $h_s$:

<!-- Kowalski and Torrilhon 2019, eq. (3.1) -->
$$
\partial_x u + \partial_y v + \partial_z w = 0
$$

<!-- eq. (3.2) -->
$$
\partial_t u + \partial_x (u^2) + \partial_y (uv) + \partial_z (uw) = -\frac{1}{\rho} \partial_x p + \frac{1}{\rho} \partial_z \sigma_{xz} + g e_x
$$

<!-- eq. (3.3) -->
$$
\partial_t v + \partial_x (uv) + \partial_y (v^2) + \partial_z (vw) = -\frac{1}{\rho} \partial_y p + \frac{1}{\rho} \partial_z \sigma_{yz} + g e_y
$$

<!-- eq. (3.4) -->
$$
p(t,x,y,z) = (h_s(t,x,y) - z) \rho g e_z
$$

with the kinematic conditions at the free surface and at the bed

<!-- eq. (A.3) -->
$$
\partial_t h_s + (u(t,x,y,h_s), v(t,x,y,h_s))^T \cdot \nabla h_s = w(t,x,y,h_s)
$$

<!-- eq. (A.4) -->
$$
\partial_t h_b + (u(t,x,y,h_b), v(t,x,y,h_b))^T \cdot \nabla h_b = w(t,x,y,h_b)
$$

the Newtonian shear stress, the stress-free surface and the slip condition at the bed with slip length $\lambda$

<!-- eq. (3.28), (3.30), (3.32) -->
$$
\sigma_{xz} = \mu \partial_z u, \qquad
\partial_z u|_{z=h_s} = 0, \qquad
(u - \lambda \partial_z u)|_{z=h_b} = 0
$$

and the same for $v$.
In the mapped coordinate $\zeta \in [0, 1]$ the velocity is the mean plus a polynomial deviation

<!-- eq. (3.5), (4.3) -->
$$
\zeta = \frac{z - h_b(t, x, y)}{h(t, x, y)}, \qquad
u(x,y,t,\zeta) = u_m(x,y,t) + \sum_{j=1}^{N} \alpha_j(x,y,t) \phi_j(\zeta)
$$

with $h = h_s - h_b$ and the scaled Legendre polynomials $\phi_j$, orthogonal on $[0, 1]$ with $\phi_j(0) = 1$.
The moment equations are the momentum balance tested with $\phi_i$ and integrated over $\zeta$.

## Assumptions

- Shallow flow: the in-plane deviatoric stresses $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{xy}$ are neglected.
- Hydrostatic pressure (3.4), constant density.
- Newtonian shear stress, stress-free surface, Navier slip at the bed, enforced weakly through the integrated stress terms.
- Velocity profile: a polynomial of degree $N$ in $\zeta$ (4.3).

## Generated equations

```{include} ../../../_generated/model_sme.md
```

Notation: $q_0 = h u_m$ and $q_k = (-1)^k h \alpha_k$ (Zoomy's basis has $\phi_k(1) = 1$), $\lambda_s = \rho \nu / \lambda$, $b = h_b$, $e_z = 1$, `level` $= N$.
The row `momentum_x_k` is $(-1)^k$ times the paper's equation for $\alpha_k$.
With a flat bed and $e_x = 0$, `level=0` is the paper's (4.13) and `level=1` its (4.14) with (4.15).
`tests/model/test_model_references.py` in zoomy_core checks levels 1 to 3 row by row against (4.7) and (4.10).

## Options

| setting | |
|---|---|
| `level` | moments above the mean, `0` is shallow water |
| `dimension` | `2`: (x, z), state `b, h, q_0, ..., q_level`. `3`: (x, y, z), state `b, h, q_x_k, q_y_k` |
| `closures` | for example `[Newtonian(), NavierSlip(), StressFree()]`. Empty leaves the shear stress unclosed |
| `stress` | `Stress.shallow` keeps the vertical shear (the paper's model), `Stress.full` the whole Newtonian tensor |
| `scaling` | a `ShallowScaling`: the regime of the stress truncation |
| `small_slope` | `True` shallow boundary tractions, `False` slope-aware ones |
| `quadrature_order` | Gauss-Legendre order for non-polynomial closures, `0` off |
| `parameters` | `g`, `rho`, `nu`, `lambda_s`, `e_x` |

```python
from zoomy_core.model.models import SME, Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel import SystemModel

model = SME(level=1, dimension=2, closures=[Newtonian(), NavierSlip(), StressFree()],
            parameters={"nu": 0.1, "lambda_s": 0.5})
sm = SystemModel.from_model(model)
```

## In the GUI

Compose, Model card **Shallow Moments**: set `level` in its parameters panel.
Pick a mesh and the **Hyperbolic (NumPy)** solver and run.
The card writes `SME(...)` with `[Newtonian(), NavierSlip(), StressFree()]` into the `## Model` section of `case.py`.
