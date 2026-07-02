# zoomy-prepost

Shared pre/postprocessing conversions for Zoomy — the general place for format
conversion so backends don't each ship custom writers.

Today it converts **VTK / ParaView `.pvd` → zoomy HDF5** (the format
`zoomy_plotting.read_hdf5` and the GUI read), so any backend that emits VTK
(firedrake `.pvd`, foam / dmplex `.vtu`/`.vtk`) does one call and the normal
postprocessing just works.

```python
from zoomy_prepost import vtk_to_hdf5
vtk_to_hdf5("simulation.pvd", "simulation.h5")   # .pvd collection, .vtu, or [frames]
```

```bash
zoomy-convert simulation.pvd simulation.h5
```

Output layout (`zoomy_core.misc.io` convention):

    /mesh/{dimension, type, n_cells, n_inner_cells,
           vertex_coordinates:(dim,n_vert), cell_vertices:(k,n_cells)}
    /fields/iteration_i/{time, Q:(n_vars,n_cells)}

Install it only in the containers that need it (firedrake, foam, dmplex). Mesh
conversion helpers can grow here too.
