"""What each layer of the C1 bump holds: Model -> SystemModel -> NSM."""
from bump import bump

mesh, nsm = bump(order=2, nx=40)
sm = nsm.sm                                         # the SystemModel the NSM was built from

print("state      ", [str(s) for s in sm.state])
print("aux_state  ", [str(s) for s in sm.aux_state])
print("parameters ", {k: sm.parameter_values[k] for k in sm.parameters.keys()})
for slot in ("flux", "hydrostatic_pressure", "nonconservative_matrix", "source",
             "source_explicit", "diffusion_matrix", "diffusion_matrix_explicit", "eigenvalues"):
    op = getattr(sm, slot)
    if op is None:
        print(f"{slot:26s} None")
    else:
        print(f"{slot:26s} shape {tuple(op.shape)}, nonzero {sum(1 for e in op if e != 0)}")
print("riemann        ", getattr(nsm.riemann, "__name__", type(nsm.riemann).__name__))
print("reconstruction ", nsm.reconstruction)
print("mesh           ", type(mesh).__name__, int(mesh.n_inner_cells), "cells")
