# %% [markdown] zoomy={"role":"meta","title":"SME dam break (1D)","description":"Shallow moments level 2."}
# # SME dam break (1D)
# 
# Shallow moments level 2.

# %% [markdown] zoomy={"role":"heading","section":"model"}
# ## Model

# %% zoomy={"role":"model","class_path":"zoomy_core.model.models.SME","init":{"level":2}}


# %% [markdown] zoomy={"role":"heading","section":"mesh"}
# ## Mesh

# %% zoomy={"role":"mesh","spec":{"x_min":0,"x_max":10,"n_cells":80}}


# %% [markdown] zoomy={"role":"heading","section":"settings"}
# ## Solver settings

# %% zoomy={"role":"settings","settings":{"backend":"numpy"}}
settings = {
  "backend": "numpy"
}

# %% [markdown] zoomy={"role":"heading","section":"run"}
# ## Run

# %% zoomy={"role":"run"}
# Runs IN-PROCESS with the numpy solver (no server needed) — works in
# JupyterLite, a backend container's JupyterLab, any env with zoomy_core,
# or standalone as `python run.py` in a materialized case folder (the
# zoomy-server generic runner executes exactly this file).
# For other backends, submit this file via the Zoomy GUI / zoomy-server.
import json
import os

# Folder form: the notebook defines model/mesh/settings in the cells above;
# a standalone run.py loads them from the sibling case-folder files.
if "model" not in globals():
    exec(open("model.py").read())
if "mesh" not in globals():
    exec(open("mesh.py").read())
if "settings" not in globals():
    settings = json.load(open("settings.json"))

from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
from zoomy_core.misc.misc import Zstruct

# effective settings = general keys + this backend's branch
_eff = {k: v for k, v in settings.items() if not isinstance(v, dict)}
_eff.update(settings.get("numpy", {}))

nsm = NumericalSystemModel.from_system_model(
    model,
    reconstruction=ReconstructionSpec(
        order=_eff.get("reconstruction_order", 1),
        limiter=_eff.get("limiter", "venkatakrishnan")))
solver = FreeSurfaceFlowSolver(
    time_end=_eff.get("time_end", 0.1),
    compute_dt=ts.adaptive(CFL=_eff.get("cfl", 0.45)),
    settings=Zstruct(output=Zstruct(
        directory=os.getcwd(), filename="simulation",
        snapshots=_eff.get("output_snapshots", 10), clean_directory=False)))

mesh.write_to_hdf5("simulation.h5")   # mesh first; the solver appends /fields
Q, Qaux = solver.solve(mesh, nsm, write_output=True)
print("done -> simulation.h5")

# %% [markdown] zoomy={"role":"heading","section":"visualization"}
# ## Visualization

# %% zoomy={"role":"visualization"}
import matplotlib
import matplotlib.pyplot as plt
import zoomy_plotting as zp

store = zp.read_hdf5("simulation.h5")
time_step = store.n_snapshots - 1        # last snapshot
field_name = None                        # None -> first field
try:
    display                              # provided by Jupyter
except NameError:
    display = lambda *a: None            # plain-script fallback
field = next(iter(store.field.keys()))
with zp.apply_style():
    fig, ax = plt.subplots()
    zp.MatplotlibPlotter(store).plot(ax, time_step=time_step, field=field,
                                     **({} if store.dim == 1 else {"cmap": "viridis", "colorbar": True}))
    if store.times is not None and len(store.times):
        ax.set_title(f"{field} — t = {float(store.times[time_step]):.3f}")
fig.savefig("simulation.png", dpi=150, bbox_inches="tight")
display(fig)
print("figure -> simulation.png")
