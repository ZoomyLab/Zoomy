from common import *
for dim in (1, 2):
    m = SWE(dimension=dim, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in (("left","right") if dim==1 else ("left","right","bottom","top"))]))
    nsm = build_nsm(m)
    sm = nsm.sm
    print(dim, "update_variables", nsm.update_variables if hasattr(nsm,'update_variables') else None)
    print("  sm.update_variables", getattr(sm, "update_variables", None))
    print("  update_aux", getattr(sm, "update_aux_variables", None))
    print("  ncm", sm.nonconservative_matrix)
    print("  hydrostatic_pressure", sm.hydrostatic_pressure)
    print("  flux", sm.flux)
    print("  eqr", getattr(sm, "equilibrium_reconstruction", None), "aux_registry", getattr(sm, "aux_registry", None))
    print("  diffusion", getattr(sm, "diffusion_matrix", None))
    rt_np, rt_jx, p = runtimes(nsm)
    print("  jax ces", getattr(rt_jx, "numerical_cell_edge_source", None) is not None, "jax ncm", getattr(rt_jx, "nonconservative_matrix", None) is not None)
