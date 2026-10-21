"""Release smoke test: imports every package, then a 1-D dam break on NumPy and on JAX.

Run from a scratch directory in a venv that holds only the built wheels:
    python tests/release/smoke.py
"""
import importlib

import numpy as np

for name in ("zoomy_core", "zoomy_jax", "zoomy_plotting", "zoomy_prepost", "zoomy_server"):
    path = importlib.import_module(name).__file__
    print(name, path)
    assert "site-packages" in path, f"{name} is not the installed wheel"

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME, NavierSlip, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

model = SME(
    level=0,
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=BC.BoundaryConditions([BC.Wall(tag="left"), BC.Wall(tag="right")]),
    initial_conditions=IC.RP(
        high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),
        low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),
        jump_position_x=5.0,
    ),
)
nsm = NumericalSystemModel.from_system_model(
    SystemModel.from_model(model), reconstruction=ReconstructionSpec(order=1))
mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)

heights = {}
for label, solver in (("numpy", SymbolicHyperbolicSolverNumpy(time_end=0.5, cfl=0.9)),
                      ("jax", SymbolicHyperbolicSolverJax(time_end=0.5, cfl=0.9, mood=False))):
    Q, Qaux = solver.setup(mesh, nsm)
    time, Q, Qaux = solver.run(Q, Qaux)
    h = np.asarray(Q)[1]
    heights[label] = h
    print(f"{label}: t = {float(time)}  h in [{h.min():.5f}, {h.max():.5f}]  sum h = {h.sum():.6f}")
    assert np.isfinite(h).all() and abs(h.sum() - 75.0) < 1e-3, f"{label}: mass is not conserved"

assert np.abs(heights["numpy"] - heights["jax"]).max() < 1e-3, "numpy and jax disagree"
print("smoke ok")
