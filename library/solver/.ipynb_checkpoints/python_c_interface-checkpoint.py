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
    model_path="outputs/output_c/c_interface",
    debug=False,
    profiling=False,
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

    #TODO add make options from python (timestepper, ODE)
    command = " ".join(
        [
            "make",
            # "arch=cpu",
            ### WORKING ARCHITECTURE ON CLUSTER: Kokkos_ARCH_VOLTA70
            "arch=gpu",
            "device=Kokkos_ARCH_VOLTA70", 
            # "device=Volta70", 
            # "device=Volta100", 
            # "device=cuda", 
            # "device=sm_90",
            # "device=Kokkos_ARCH_TURING75",
            # "device=Tesla100", 
            f"DIMENSION={dimension}",
            f"N_BOUNDARY_CONDITIONS={n_boundary_conditions}",
            f"N_ELEMENTS={n_elements}",
            f"N_FIELDS={n_fields}",
            f"N_FIELDS_AUX={n_fields_aux}",
            f"PATH_SETTINGS={path_settings}",
            f"PATH_MESH={path_mesh}",
            f"PATH_FIELDS={path_fields}",
            f"MODEL_PATH={os.path.join(main_dir, model_path)}",
            f"TIMESTEPPER=Adaptive",
            f"TIMESTEPPER_PARAM={0.45}",
            f"ODE_FLUX={1}",
            f"ODE_SOURCE={-1}",
            f"DEBUG={int(debug)}",
            f"PROFILING={int(profiling)}",
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
