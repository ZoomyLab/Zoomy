import os

def transform_model_in_place(model, printer='jax'):
    runtime_pde = model.get_pde(printer=printer)
    runtime_bcs = model.get_boundary_conditions(printer=printer)

    return runtime_pde, runtime_bcs

def save_model_to_C(model, settings):
    _ = model.create_c_interface(
        path=os.path.join(settings.output.directory, "c_interface")
    )
    _ = model.create_c_boundary_interface(
        path=os.path.join(settings.output.directory, "c_interface")
        )