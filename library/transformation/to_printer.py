def to_jax(model, printer='jax'):
    runtime_pde = model._get_pde(printer=printer)
    runtime_bcs = model._get_boundary_conditions(printer=printer)
    return runtime_pde, runtime_bcs