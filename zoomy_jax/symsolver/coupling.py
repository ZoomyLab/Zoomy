"""A reduced arm of a coupling to a resolved free-surface solver (VOF), as one preCICE
participant on the JAX symbolic solver: the column exchange (`ColumnPreciceAdapter`), the
Riemann problem towards the characteristic ghost (`CoupledFace.characteristic`) and the depth
row held over each window at the value the peer charges (`frozen_mass_row`).

The data contract is `zoomy_foam.vof`'s: the fields `[b, h, u, v, w, p]` written as
`<field>_S` and read as `<field>_V`, `nz` cell-centre heights per coupled face.
"""
from __future__ import annotations

from zoomy_core.symsolver.discretization.time import CflWaveSpeed
from zoomy_core.symsolver.strategies.hyperbolic import CoupledFace
from zoomy_jax.symsolver.runtime import ColumnPreciceAdapter
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

FIELDS = ("b", "h", "u", "v", "w", "p")


def column_participant(nsm, mesh, *, participant, config, mesh_name, tag, nz, store,
                       dt_write, cfl=0.9, cfl_wave_speed=CflWaveSpeed.faces,
                       frozen_mass_row=True):
    """March `nsm` on `mesh` as participant `participant` of the preCICE configuration
    `config`, coupled on the boundary tag `tag` (preCICE mesh `mesh_name`) through the column
    on `nz` heights; adaptive steps inside each window (`cfl`, `cfl_wave_speed`); snapshots
    every `dt_write` into the HDF5 store `store`. Returns `store`.
    """
    adapter = ColumnPreciceAdapter(participant, str(config), mesh_name,
                                   [f"{f}_S" for f in FIELDS], [f"{f}_V" for f in FIELDS], nz)
    solver = SymbolicHyperbolicSolverJax(
        cfl=cfl, cfl_wave_speed=cfl_wave_speed, subcycling=True, dt_write=dt_write,
        output=str(store),
        couple=dict(adapter=adapter, tag=tag, face=CoupledFace.characteristic,
                    frozen_mass_row=frozen_mass_row))
    Q, Qaux = solver.setup(mesh, nsm)
    solver.run(Q, Qaux)
    adapter.finalize()
    return store


__all__ = ["FIELDS", "column_participant"]
