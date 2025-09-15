import numpy as np
import pytest
from types import SimpleNamespace
import os
from sympy import Matrix


from library.fvm.precice_solver import PreciceHyperbolicSolver, PreciceHyperbolicSolverBidirectional, PreciceHyperbolicSolverAUP, PreciceTestSolver, PreciceHyperbolicSolverAUP_while
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
        # out += self.slip()
        out += self.slip_mod()
        # out += self.newtonian()
        # out += self.newtonian_turbulent()
        # out += self.newtonian_boundary_layer_classic()
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
            lamda = 7.,
            rho=1000,
            c_slipmod= 1.0,
        ),
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 2), 2000)

    solver = PreciceHyperbolicSolver(
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=10,
        config_path=os.path.join(main_dir, f"library/precice_configs/of_to_zoomy.xml"))

    # precice_fvm(mesh, model, settings, ode_solver_source=RK1)
    solver.solve(mesh, model)
    return model

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d_bidirectional(
    settings,
    level=0,
):

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),
            BC.Right(physical_tag="left"),

            # BC.Wall(
            #     physical_tag="right",
            #     momentum_field_indices=[[1+i] for i in range(0, level + 1)],
            # ),
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
            rho= 1000.,
            lamda = 0.0001,

        ),
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 5), 500)

    solver = PreciceHyperbolicSolverBidirectional(
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=10,
        config_path=os.path.join(main_dir, f"library/precice_configs/of_zoomy_bidirectional.xml"))

    # precice_fvm(mesh, model, settings, ode_solver_source=RK1)
    solver.solve(mesh, model)
    return model

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d_from_tut(
    settings,
    level=0,
):

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),

            BC.Extrapolation(physical_tag="right"),
            # BC.Wall(
            #     physical_tag="right",
            #     momentum_field_indices=[[1+i] for i in range(0, level + 1)],
            # ),
        ]
    )
    
    def custom_ic(x):
        Q = np.zeros(2+level, dtype=float)
        Q[0] = np.where(x[0] > 0.7, 0.06, 0.02)
        Q[1]  = -1.5

        return Q
    ic = IC.UserFunction(custom_ic)
    model = MySME(
        level=level,
        parameters=Zstruct(
            g= 9.81,
            nu= 0.000001,
            lamda = 1.,
            rho=1000,
            c_slipmod= 1.0,
            c_nut=1.0,
            nut = 0.0000125934315,
            c_bl=1.0,
            nut_bl=0.0000145934315,
            eta=1,
        ),
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 1), 500)

    solver = PreciceHyperbolicSolverAUP_while(
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt = timestepping.constant(dt=0.001),
        time_end=10,
        config_path=os.path.join(main_dir, f"library/precice_configs/of_zoomy_bidirectional.xml"))

    # precice_fvm(mesh, model, settings, ode_solver_source=RK1)
    _, _,  = solver.solve(mesh, model)
    return model


@pytest.mark.critical
@pytest.mark.unfinished
def test_precice(
    settings,
    level=0,
):

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
            rho= 1000.,
            lamda = 0.0001,

        ),
        boundary_conditions=bcs,
        initial_conditions=ic,
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 1), 500)

    solver = PreciceTestSolver(
        settings=settings,
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=10,
        config_path=os.path.join(main_dir, f"library/precice_configs/from_tut.xml"))

    # precice_fvm(mesh, model, settings, ode_solver_source=RK1)
    _, _= solver.solve(mesh, model)
    return model



if __name__ == "__main__":
    settings = Settings(
    output=Zstruct(
        directory="outputs/precice", filename="sim", clean_directory=True,
    ),
    )
    nut = 0.0000145934315
    nut = 0.0000125934315
    
    # model = test_smm_1d(
    #     settings,
    #     level=0,
    #     process="",
    #     case="again",
    #     c_nut=1.0,
    #     c_bl=1.0,
    #     c_slipmod=1,
    #     lamda=70,
    #     nut=nut,
    #     nut_bl=0.000001,
    # )
    
    # settings = Settings(
    # output=Zstruct(
    #     directory="outputs/precice_bidirectional", filename="sim", clean_directory=True,
    # ),
    # )
    
    # model = test_smm_1d_bidirectional(
    #     settings,
    #     level=2,
    # )
    
    level=0
    settings = Settings(
    output=Zstruct(
        directory=f"outputs/precice_from_tut_{level}", filename="sim", clean_directory=True,
    ),
    )
    
    model = test_smm_1d_from_tut(
        settings,
        level=level,
    )
    
    # settings = Settings(
    # output=Zstruct(
    #     directory="outputs/precice_from_tut", filename="sim", clean_directory=True,
    # ),
    # )
    
    # model = test_precice(
    #     settings,
    #     level=0,
    # )
    

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
