import numpy as np
import sympy as sym
import pytest
from types import SimpleNamespace
from scipy.integrate import quad

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
# from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import argparse

Cf = 0.0036
# Cr = 0.00
theta = 0.05011
phi = 22.76
Cr = 0.00035

# u0 = 0.0
# u1 = 0.5
# h0 = 0.1
# u = lambda z: u0 + u1 * (-2*z + 1)
# P = h0 * quad(lambda z: (u(z)-u0)**2, 0, 1)[0]

phi1 = lambda z: -2*z +1
phi2 = lambda z: 1-6*z+6*z**2

basis = Legendre_shifted(order=10)

def load_tc(exp):
    if exp == 2:
        time_end = 1.
        h0 = 0.1
        u0 = 0.5
        u1 = lambda x: -0.5 
        u2 = lambda x: 0
        h0 = 0.1
        u = lambda x, z: u0 + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0)**2, 0, 1)[0]
    elif exp == 3:
        time_end = 3.
        h0 = 0.1
        u0 = 0.0
        u1 = lambda x: -0.5 
        u2 = lambda x: 0
        h0 = 0.1
        u = lambda x, z: u0 + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0)**2, 0, 1)[0]
    else:
        time_end = 1.5
        h0 = 0.1
        u0 = 0
        u1 = lambda x: -0.5 * np.cos(x/10*np.pi)
        u2 = lambda x: -0.5 * np.cos(x/10*np.pi)
        u = lambda x, z: u0 + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0)**2, 0, 1)[0]
    return h0, u0, u1, u2, P, time_end

def hybrid_SMM_SFF(exp=2, level=1):
    offset = level+1
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux = nonconservative_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=5,
        output_snapshots=200,
        output_dir = f'outputs/eccomas/coupled_SMM_SFF-{level}'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )

    if exp == 2:
        h0, u0, u1, u2, P, time_end = load_tc(2)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            Q[1] = np.where(x > 0, h0 * u0, -h0*u0)
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            return Q
    elif exp == 3:
        h0, u0, u1, u2, P, time_end = load_tc(3)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            # if level > 1:
            #     Q[3] = h0 * u2(x[0])
    elif exp == 4:
        time_end = 20
        settings.time_end=time_end
        shear_slope = 0.1
        h0 = lambda x: 0.1 
        u0 = lambda x: 0.1
        u1 = lambda x: shear_slope *np.cos(x/10*np.pi)
        u2 = lambda x: 0
        u = lambda x, z: u0(x) + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0(x))**2, 0, 1)[0]
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0(x)
            Q[1] = h0(x) * u0(x)
            if level > 0:
                Q[2] = np.where(x > 0, h0(x) * u1(x), h0(x) * u1(x))
            Q[-1] = P(x)
            return Q
    elif exp == 97:
        time_end = 20
        h0 = 0.1
        u0 = 0
        shear_slope = 0.8
        u1 = lambda x: - shear_slope * np.cos(x/10*np.pi)
        u2 = lambda x: 0
        u = lambda x, z: u0 + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0)**2, 0, 1)[0]
        settings.time_end = time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), h0*u1(x))
            # if level > 1:
            #     Q[3] = h0 * u2(x[0])
            # Q[-1] = 1/3 * (shear_slope*u1(x))**2
            Q[-1] = P(x)
            return Q
    elif exp == 98:
        h0, u0, u1, u2, P, time_end = load_tc(2)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            # if level > 1:
            #     Q[3] = h0 * u2(x[0])
            Q[-1] = 4/3 * u1(x)**2
            return Q
    elif exp == 99:
        h0, u0, u1, u2, P, time_end = load_tc(2)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            # if level > 1:
            #     Q[3] = h0 * u2(x[0])
            return Q
    else:
        h0, u0, u1, u2, P, time_end = load_tc(0)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3+(level), dtype=float)
            Q[0] = h0
            Q[1] = h0 * u0
            if level > 0:
                Q[2] = h0 * u1(x[0])
            if level > 1:
                Q[3] = h0 * u2(x[0])
            return Q

    ic = IC.UserFunction(function = custom_ic)

    model = HybridSFFSMM(
        dimension=1,
        fields=2 + level + 1,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ['chezy_ssf']},
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level + 1)),
    )
    print(model.sympy_flux)
    print(model.sympy_nonconservative_matrix)
    print(sym.simplify(model.sympy_eigenvalues))

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    # jax_fvm_unsteady_semidiscrete(
    #     mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    # )
    solver_price_c(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


def SMM_enforce_SFF(level=1):
    offset = level+1
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=5,
        output_snapshots=100,
        output_dir = f'outputs/eccomas/SMMWS-{level}'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )

    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    #     low=lambda n_field: np.array([0.01, 0.0] + [0.0 for l in range(level)]),
    # )
    # ic = IC.Constant(lambda n_fields: [h0, u0, u1] + [0 for l in range(level-1)])
    # ic = IC.RP(
    #     high=lambda n_field: np.array([h0, h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    #     low=lambda n_field: np.array([h0, -h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    # )
    def custom_ic(x):
        Q = np.zeros(2+(level), dtype=float)
        Q[0] = h0
        Q[1] = h0 * u0
        Q[2] = h0 * u1(x[0])
        return Q
    ic = IC.UserFunction(function = custom_ic)

    model = ShallowMomentsAugmentedSSF(
        dimension=1,
        fields=2 + level + 1,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ['chezy_ssf']},
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level + 1)),
    )
    print(model.sympy_eigenvalues)

    main_dir = os.getenv("SMS")
    # mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    # jax_fvm_unsteady_semidiscrete(
    #     mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    # )
    # io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

def SMMWS(level=0):
    offset = level+1
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=5,
        output_snapshots=100,
        output_dir = f'outputs/eccomas/SMMWS-{level}'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )

    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    #     low=lambda n_field: np.array([0.01, 0.0] + [0.0 for l in range(level)]),
    # )
    # ic = IC.Constant(lambda n_fields: [h0, u0, u1] + [0 for l in range(level-1)])
    # ic = IC.RP(
    #     high=lambda n_field: np.array([h0, h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    #     low=lambda n_field: np.array([h0, -h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    # )
    def custom_ic(x):
        Q = np.zeros(2+(level), dtype=float)
        Q[0] = h0
        Q[1] = h0 * u0
        Q[2] = h0 * u1(x[0])
        return Q
    ic = IC.UserFunction(function = custom_ic)

    model = ShallowMomentsSSF(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ['chezy_ssf']},
        settings={"friction": []},
        basis=BasisNoHOM(basis=Legendre_shifted(order=level)),
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

def SMM(level=0, exp=0):
    offset = level+1
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.5),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=5,
        output_snapshots=100,
        output_dir = f'outputs/eccomas/SMM-{level}'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )

    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    #     low=lambda n_field: np.array([.01, 0.0] + [0.0 for l in range(level)]),
    # )
    # ic = IC.Constant(lambda n_fields: [h0, u0, u1] + [0 for l in range(level-1)])
    # ic = IC.RP(
    #     high=lambda n_field: np.array([h0, h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    #     low=lambda n_field: np.array([h0, -h0*u0, h0*u1] + [0.0 for l in range(level-1)])[:level+2],
    # )
    if exp == 2:
        h0, u0, u1, u2, P, time_end = load_tc(2)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(2+(level), dtype=float)
            Q[0] = h0
            Q[1] = np.where(x > 0, h0 * u0, -h0*u0)
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            return Q
    elif exp == 3:
        h0, u0, u1, u2, P, time_end = load_tc(3)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(2+(level), dtype=float)
            Q[0] = h0
            if level > 0:
                Q[2] = np.where(x > 0, h0 * u1(x), -h0*u1(x))
            # if level > 1:
            #     Q[3] = h0 * u2(x[0])
            return Q
    elif exp == 4:
        time_end = 2
        settings.time_end=time_end
        shear_slope = 0.1
        h0 = lambda x: 0.1 
        u0 = lambda x: 0.1
        u1 = lambda x: shear_slope *np.cos(x/10*np.pi)
        u2 = lambda x: 0
        u = lambda x, z: u0(x) + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0(x))**2, 0, 1)[0]
        def custom_ic(x):
            Q = np.zeros(2+(level), dtype=float)
            Q[0] = h0(x)
            Q[1] = h0(x) * u0(x)
            if level > 0:
                Q[2] = np.where(x > 0, h0(x) * u1(x), h0(x) * u1(x))
            return Q
    else:
        h0, u0, u1, u2, P, time_end = load_tc(0)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(2+(level), dtype=float)
            Q[0] = h0
            Q[1] = h0 * u0
            if level > 0:
                Q[2] = h0 * u1(x[0])
            if level > 1:
                Q[3] = h0 * u2(x[0])
            return Q

    ic = IC.UserFunction(function = custom_ic)

    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ['chezy_ssf']},
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    # jax_fvm_unsteady_semidiscrete(
    #     mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    # )
    print(model.sympy_quasilinear_matrix)
    solver_price_c(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


def SSF(exp=0):
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr, "theta":theta, "phi": phi},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=5,
        output_snapshots=100,
        output_dir = 'outputs/eccomas/SSF'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )


    P11_0 = 0.0

    # ic = IC.RP(
    #     high=lambda n_field: np.array([0.02, 0.0, P11_0]),
    #     low=lambda n_field: np.array([.01, 0.0, P11_0]),
    # )
    # ic = IC.Constant(lambda n_fields: [h0, u0, P])
    # ic = IC.RP(
    #     high=lambda n_field: np.array([h0, h0*u0, P]),
    #     low=lambda n_field: np.array([h0, -h0*u0, P]),
    # )
    if exp == 2:
        h0, u0, u1, u2, P, time_end = load_tc(2)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3, dtype=float)
            Q[0] = h0
            Q[1] = np.where(x > 0, h0 * u0, -h0*u0)
            Q[2] = P(x[0])
            return Q
    elif exp == 3:
        h0, u0, u1, u2, P, time_end = load_tc(3)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3, dtype=float)
            Q[0] = h0
            Q[2] = P(x[0])
            return Q
    elif exp == 4:
        time_end = 20
        settings.time_end=time_end
        shear_slope = 0.1
        h0 = lambda x: 0.1 
        u0 = lambda x: 0.1
        u1 = lambda x: shear_slope *np.cos(x/10*np.pi)
        u2 = lambda x: 0
        u = lambda x, z: u0(x) + u1(x) * basis.eval(1, z) + u2(x) * basis.eval(2, z)
        P = lambda x: quad(lambda z: (u(x, z)-u0(x))**2, 0, 1)[0]
        def custom_ic(x):
            Q = np.zeros(3, dtype=float)
            Q[0] = h0(x)
            Q[1] = h0(x) * u0(x)
            Q[2] = P(x)
            return Q
    else:
        h0, u0, u1, u2, P, time_end = load_tc(0)
        settings.time_end=time_end
        def custom_ic(x):
            Q = np.zeros(3, dtype=float)
            Q[0] = h0
            Q[1] = h0 * u0
            Q[2] = P(x[0])
            return Q
    ic = IC.UserFunction(function = custom_ic)

    model = ShearShallowFlow(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        # settings={"friction": ["chezy"]},
        settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    print(model.sympy_flux)
    print(model.sympy_nonconservative_matrix)
    print(sym.simplify(model.sympy_eigenvalues))

    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


def SSFPathconservative():
    settings = Settings(
        parameters={"g": 9.81, "C": 1/Cf, "Cf": Cf, "Cr": Cr},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        # compute_dt=timestepping.constant(dt=0.1),
        time_end=5,
        output_snapshots=100,
        output_dir = 'outputs/eccomas/SSFPathconservative'
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag="left", periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
        ]
    )

    ic = IC.RP(
        high=lambda n_field: np.array([0.02, 0., 0., 0., 0., 0.]),
        low=lambda n_field: np.array([.01, 0., 0., 0., 0., 0.]),
    )

    model = ShearShallowFlowPathconservative(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy"]},
        # settings={"friction": ["newtonian"]},
        settings={"friction": []},
        # settings={"friction": ["friction_paper"]},
    )
    
    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

if __name__ == "__main__":
    # SMMWS(level=1)
    # SMM(level=1)
    # SSF()
    # SSFPathconservative()
    # SMM_enforce_SFF(level=0)
    # SMM_enforce_SFF(level=1)
    # SMM_enforce_SFF(level=2)
    # SMM_enforce_SFF(level=3)
    # coupled_SMM_SFF(exp=99, level=0)
    # coupled_SMM_SFF(exp=99, level=1)
    # coupled_SMM_SFF(exp=97, level=2)

    # hybrid_SMM_SFF(exp=4, level=0)
    # hybrid_SMM_SFF(exp=4, level=1)
    # hybrid_SMM_SFF(exp=4, level=2)
    # hybrid_SMM_SFF(exp=4, level=3)
    # SSF(exp=4)
    # SMM(exp=4, level=0)
    SMM(exp=4, level=1)
    # SMM(exp=4, level=2)
    # SMM(exp=4, level=3)