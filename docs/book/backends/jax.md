# JAX backend

`zoomy_jax` runs the same finite-volume HyperbolicSolver on JAX, giving JIT,
vectorisation, autodiff, and GPU/TPU acceleration.

**Highlights**
- Second-order MUSCL reconstruction + Venkatakrishnan limiter.
- Crank–Nicolson diffusion (IMEX) for parabolic terms.
- 1D / 2D shallow-water and SME models verified against the NumPy reference.

**Install**

```bash
pip install zoomy_core zoomy_jax
```

**Repository**: [`library/zoomy_jax`](https://github.com/ZoomyLab/zoomy-jax)  
**API reference**: see [zoomy_jax](../api/_apidoc_zoomy_jax/zoomy_jax.rst).
