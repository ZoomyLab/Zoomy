import subprocess
import os


def build(
    dimension=2,
    n_boundary_conditions=4,
    n_elements=100,
    n_fields=3,
    n_fields_aux=0,
    path_settings="outputs/output_c/settings.hdf5",
    path_mesh="outputs/output_c/mesh.hdf5",
    path_fields="outputs/output_c/fields.hdf5",
):
    main_dir = os.getenv("SMS")
    path = f"{main_dir}/library/solver"
    dimension = dimension
    n_boundary_conditions = n_boundary_conditions
    n_elements = n_elements
    n_fields = n_fields
    n_fields_aux = n_fields_aux
    path_settings = os.path.join(main_dir, path_settings)
    path_mesh = os.path.join(main_dir, path_mesh)
    path_fields = os.path.join(main_dir, path_fields)

    command = "make clean"
    make_process = subprocess.Popen(
        command, shell=True, stderr=subprocess.STDOUT, cwd=path
    )
    if make_process.wait() != 0:
        print(subprocess.STDOUT)

    command = " ".join(
        [
            "make",
            f"DIMENSION={dimension}",
            f"N_BOUNDARY_CONDITIONS={n_boundary_conditions}",
            f"N_ELEMENTS={n_elements}",
            f"N_FIELDS={n_fields}",
            f"N_FIELDS_AUX={n_fields_aux}",
            f"PATH_SETTINGS={path_settings}",
            f"PATH_MESH={path_mesh}",
            f"PATH_FIELDS={path_fields}",
            f"TIMESTEPPER=Adaptive",
            f"TIMESTEPPER_PARAM={0.45}",
            f"ODE_FLUX={1}",
            f"ODE_SOURCE={-1}",
        ]
    )
    print(f"make command : {command}")
    # command = ["make", f"dimension={dimension}",  f"arch={architecture}"]
    # command = "make"
    make_process = subprocess.Popen(
        command, shell=True, stderr=subprocess.STDOUT, cwd=path
    )
    if make_process.wait() != 0:
        print(subprocess.STDOUT)


def run_driver():
    main_dir = os.getenv("SMS")
    path = f"{main_dir}/bin"
    command = " ".join(["./volkos;"])
    # command = " ".join(["mpicc ./volkos;"])
    make_process = subprocess.Popen(
        command, shell=True, stderr=subprocess.STDOUT, cwd=path
    )
    if make_process.wait() != 0:
        print(subprocess.STDOUT)


if __name__ == "__main__":
    build()
    # run_driver()
