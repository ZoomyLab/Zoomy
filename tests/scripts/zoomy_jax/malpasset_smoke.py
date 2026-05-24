"""Smoke test: MalpassetSWE on a tiny 2D structured mesh, JAX path,
ONE step.  Goal — confirm the model + setup_simulation + flux op
chain runs end-to-end in under 1 minute (target: ~10 s)."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malpasset_swe_model import MalpassetSWE  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.models.system_model import SystemModel  # noqa: E402
from zoomy_core.numerics import NumericalSystemModel  # noqa: E402
from zoomy_core.numerics.numerical_system_model import (
    ReconstructionSpec,
)  # noqa: E402
from zoomy_jax.fvm.solver_jax import HyperbolicSolver  # noqa: E402


def main():
    NX, NY = 16, 8
    DOMAIN = (0.0, 1600.0, 0.0, 800.0)   # rough Malpasset extent (m)

    t0 = time.perf_counter()
    mesh_np = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
    model = MalpassetSWE()
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(order=1),
    )
    solver = HyperbolicSolver()
    Q, Qaux = solver.setup_simulation(mesh_np, nsm)
    t_setup = time.perf_counter() - t0

    n_cells = int(solver._rt_mesh.n_inner_cells)
    print(f"setup OK: n_cells={n_cells}, Q.shape={Q.shape}, "
          f"Qaux.shape={Qaux.shape}, t_setup={t_setup:.2f}s")

    # ── Set IC: water height ~10 m everywhere, no flow, flat bathymetry.
    Q = Q.at[0].set(0.0)        # b = 0
    Q = Q.at[1].set(10.0)       # h = 10 m
    Q = Q.at[2].set(0.0)        # hu = 0
    Q = Q.at[3].set(0.0)        # hv = 0
    # Refresh Qaux from new Q.
    Qaux = solver.update_qaux(
        Q, Qaux, Q, Qaux, solver._rt_mesh, solver._rt_model,
        solver._rt_parameters, 0.0, 1.0,
    )

    # ── One forward-Euler step.
    dt = 0.04                   # ~CFL on shallow lake
    t0 = time.perf_counter()
    Q_new = solver.step(dt, 0.0, Q, Qaux)
    Q_new.block_until_ready()
    t_step1 = time.perf_counter() - t0

    # Second step (warm cache).
    t0 = time.perf_counter()
    Q_new = solver.step(dt, dt, Q_new, Qaux)
    Q_new.block_until_ready()
    t_step2 = time.perf_counter() - t0

    print(f"step 1: {t_step1:.3f}s  (compile + run)")
    print(f"step 2: {t_step2:.3f}s  (warm)")
    print(f"final h range: [{float(Q_new[1].min()):.3f}, "
          f"{float(Q_new[1].max()):.3f}]")
    assert jnp.isfinite(Q_new).all(), "non-finite values"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
