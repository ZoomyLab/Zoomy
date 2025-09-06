import numpy as np
import pytest
from types import SimpleNamespace
import os
from sympy import Matrix

from library.fvm.precice_solver import PreciceHyperbolicSolver
from library.model.models.shallow_moments import ShallowMoments
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
import library.misc.io as io
from library.misc.misc import Settings, Zstruct
import library.fvm.timestepping as timestepping

# from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import argparse

main_dir = os.getenv("ZOOMY_DIR")

class MySME(ShallowMoments):
    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        return out



@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d(
    settings,
    level=0,
    process="",
    case="",
    c_nut=1.0,
    c_bl=1.0,
    c_slipmod=1.0,
    lamda=7.0,
    nut=0.0000145934315,
    nut_bl=0.0000145934315,
):
    print("==============================================")
    print("==============================================")
    print("==============================================")
    print(f"Running process {process} with level {level}")
    print("==============================================")
    print("==============================================")
    print("==============================================")


    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),
            BC.Wall(
                physical_tag="right",
                momentum_field_indices=[[1+i] for i in range(0, level + 1)],
            ),
        ]
    )
    ic = IC.RP(
        high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
        low=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    )
    model = MySME(
        level=level,
        parameters=Zstruct(
            g= 9.81,
            nu= 0.000001,
            rho= 1.000,
            lamda= lamda,
            C= 30.0,
            kst= 100,
            eta= 1.0,
            c_nut= c_nut,
            c_bl= c_bl,
            c_slipmod= c_slipmod,
            nut = nut,
            nur_bl= nut_bl,
        ),
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 5), 300)

    solver = PreciceHyperbolicSolver(
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=10,
        config_path=os.path.join(main_dir, f"library/precice_configs/of_to_zoomy.xml"))

    # precice_fvm(mesh, model, settings, ode_solver_source=RK1)
    solver.solve(mesh, model)
    return model


if __name__ == "__main__":
    settings = Settings(
    output=Zstruct(
        directory="outputs/precice", filename="sim", clean_directory=True,
    ),
    )
    nut = 0.0000145934315
    nut = 0.0000125934315
    model = test_smm_1d(
        settings,
        level=0,
        process="",
        case="again",
        c_nut=1.0,
        c_bl=1.0,
        c_slipmod=1,
        lamda=70,
        nut=nut,
        nut_bl=0.000001,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.output.filename}.h5"))
    postprocessing.vtk_interpolate_3d(model, settings, Nz=20, filename='out_3d')


    # test_smm_1d(level=6, process='_1', case='again', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=4, process='_2', case='again', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=2, process='_3', case='again', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=1, process='_4', case='again', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=8, process='_5', case='again', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)

    # nut = 0.0000155934315
    # test_smm_1d(level=6, process='_1', case='r-no', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.0000001)
    # nut = 0.0000165934315
    # test_smm_1d(level=6, process='_2', case='r-no2', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.0000012)
    # nut = 0.0000175934315
    # test_smm_1d(level=6, process='_3', case='r-no3', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.0000014)

    # test_smm_1d(level=8, process='_2', case='nut_bl-r', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=4, process='_3', case='nut_bl-r', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=2, process='_4', case='nut_bl-r', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=1, process='_5', case='nut_bl-r', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)

    # test_smm_1d(level=0, process='_1', case='manning80', c_nut = 0., c_bl=0., c_slipmod=0, lamda=70, nut=0, nut_bl=0)
    # test_smm_1d(level=0, process='_2', case='manning100', c_nut = 0., c_bl=0., c_slipmod=0, lamda=70, nut=0, nut_bl=0)

    # test_smm_1d(level=8, process='_1', case='lvl8-recover', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
    # test_smm_1d(level=8, process='_2', case='lvl8-off', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.00000)
    # test_smm_1d(level=8, process='_3', case='lvl8-higher', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000002)
    # test_smm_1d(level=8, process='_4', case='lvl8-lower', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.0000005)

    # test_smm_1d(level=8, process='_1', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=700)
    # test_smm_1d(level=8, process='_2', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)
    # test_smm_1d(level=4, process='_3', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=700)
    # test_smm_1d(level=4, process='_4', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)
    # test_smm_1d(level=2, process='_5', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=700)
    # test_smm_1d(level=2, process='_6', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)

    # test_smm_1d(level=0, process='_1', case='manning80', c_nut = 0., c_bl=0., c_slipmod=0, lamda=700)
    # test_smm_1d(level=0, process='_2', case='manning80', c_nut = 0., c_bl=0., c_slipmod=0, lamda=70)
    # test_smm_1d(level=1, process='_5', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=700)
    # test_smm_1d(level=1, process='_6', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)

    # test_smm_1d(level=0, process='_1', case='chezy30', c_nut = 0., c_bl=0., c_slipmod=0, lamda=700)
    # test_smm_1d(level=0, process='_2', case='manning160', c_nut = 0., c_bl=0., c_slipmod=0, lamda=70)

    # test_smm_1d(level=6, process='_1', case='drag', c_nut = 2., c_bl=1., c_slipmod=1, lamda=70)
    # test_smm_1d(level=6, process='_2', case='drag', c_nut = 2., c_bl=1., c_slipmod=1, lamda=7)
    # test_smm_1d(level=6, process='_3', case='', c_nut = 1., c_bl=1., c_slipmod=1, lamda=7)

    # test_smm_1d(level=6, process='_1', case='pt45', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)
    # test_smm_1d(level=6, process='_2', case='pt40', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)
    # test_smm_1d(level=6, process='_3', case='pt35', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)

    # test_smm_1d(level=6, process='_1', case='h_of_integrated', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70)

    ### This does not do the job. seems like I need to modify the friction at the boudnary -> nut_bl
    # test_smm_1d(level=6, process='_1', case='nut_high', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=0.00003, nut_bl=0.)

    # test_smm_1d(level=6, process='_1', case='nut_bl_high', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.0000003)
    # test_smm_1d(level=6, process='_2', case='nut_bl_low', c_nut = 1., c_bl=1., c_slipmod=1, lamda=70, nut=nut, nut_bl=0.000001)
