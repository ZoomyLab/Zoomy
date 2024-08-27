import os
import numpy as np
import pyvista as pv
import panel as pn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from copy import deepcopy
import seaborn 
seaborn.set_context('talk')
from IPython.display import Math

main_dir = os.getenv("SMS")
import pytest
from types import SimpleNamespace

from library.model.model import *
from library.pysolver.solver import *
from library.pysolver.solver import jax_fvm_unsteady_semidiscrete as fvm_unsteady
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import panel as pn

settings = Settings(
    reconstruction=recon.constant,
    num_flux=flux.LLF(),
    nc_flux=nonconservative_flux.segmentpath(1),
    compute_dt=timestepping.adaptive(CFL=.45),
    time_end=0.1,
    output_snapshots=100,
    output_clean_dir=True,
    output_dir="outputs/output_introduction/swe",
)


def simulate():
    mesh = petscMesh.Mesh.from_gmsh(os.path.join(main_dir, "meshes/quad_2d/mesh_finest.msh"))
    print(f"physical tags in gmsh file: {mesh.boundary_conditions_sorted_names}")
    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag='left', periodic_to_physical_tag='right'),
            BC.Periodic(physical_tag="right", periodic_to_physical_tag='left'),
            BC.Periodic(physical_tag='top', periodic_to_physical_tag='bottom'),
            BC.Periodic(physical_tag="bottom", periodic_to_physical_tag='top'),
        ]
    )
    
    
    def custom_ic(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = np.where(x[0]**2 + x[1]**2 < 0.1, 2., 1.)
        return Q
    
    ic = IC.UserFunction(custom_ic)
    
    

    model = ShallowWater2d(
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"]},
        parameters = {"C": 1.0, "g": 9.81, "ez": 1.0}
    )
    
    # display(Math(r'\large{' + 'Flux \, in \, x' + '}'))
    # display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_flux[0])) + '}'))
    # display(Math(r'\large{' + 'Nonconservative \, matrix \, in \, x' + '}'))
    # display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_nonconservative_matrix[0])) + '}'))
    # display(Math(r'\large{' + 'Eigenvalues' + '}'))
    # display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_eigenvalues)) + '}'))
    # display(Math(r'\large{' + 'Source' + '}'))
    # display(Math(r'\large{' + sympy.latex(sympy.simplify(model.sympy_source)) + '}'))
    
    # settings = Settings(
    #     reconstruction=recon.constant,
    #     num_flux=flux.LLF(),
    #     nc_flux=nonconservative_flux.segmentpath(1),
    #     compute_dt=timestepping.adaptive(CFL=.45),
    #     time_end=0.1,
    #     output_snapshots=100,
    #     output_clean_dir=True,
    #     output_dir="outputs/output_introduction/swe",
    # )
    
    
    fvm_unsteady(
        mesh, model, settings, ode_solver_source=RK1
    )
    
    io.generate_vtk(os.path.join(os.path.join(main_dir, settings.output_dir), f'{settings.name}.h5'))

def plot(p):
    out_0 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.0.vtk'))
    out_10 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.10.vtk'))
    # out_98 = pv.read(os.path.join(os.path.join(main_dir, settings.output_dir), 'out.98.vtk'))
    field_names = out_0.cell_data.keys()
    print(f'Field names: {field_names}')
    # return 'done'
    # p.subplot(0, 1)
    p.add_mesh(out_10, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=0.1)'})
    p.enable_parallel_projection()
    p.enable_image_style()
    p.view_xy()

    # p.subplot(0, 0)
    # p.add_mesh(out_0, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=0)'})
    # p.enable_parallel_projection()
    # p.enable_image_style()
    # p.view_xy()
    # 
    # p.subplot(0, 1)
    # p.add_mesh(out_10, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=0.1)'})
    # p.enable_parallel_projection()
    # p.enable_image_style()
    # p.view_xy()
    
    # p.subplot(0, 2)
    # p.add_mesh(out_98, scalars='0', show_edges=False, scalar_bar_args={'title': 'h(t=1)'})
    # p.enable_parallel_projection()
    # p.enable_image_style()
    # p.view_xy()
    
    # p.show(jupyter_backend='static')
    return p
pn.extension('vtk')

# text = pn.widgets.TextInput(value='Ready')
#
# def b(event):
#     text.value = 'Clicked {0} times'.format(button.clicks)
#     
# button.on_click(b)
# pn.Row(button, text)

# pn.panel("Hello World")

plotter = pv.Plotter()
button = pn.widgets.Button(name='Simulate', button_type='primary')
button_plot = pn.widgets.Button(name='Plot ', button_type='primary')
text = pn.widgets.TextInput(value='Ready')
# p = pv.Plotter(shape=(1,3), notebook=True)
# plotter = pv.Plotter() # we define a pyvista plotter
plotter.background_color = (0.1, 0.2, 0.4)
# we create a `VTK` panel around the render window
geo_pan_pv = pn.panel(plotter.ren_win, width=500, height=500) 
# p = plot(p)
# geo_pan_pv = pn.panel(p)


def b(event):
    result = 'Simulate'
    text.value = f'{result}'
    simulate()
    # geo_pan_pv = pn.panel(p.ren_win, width=500, height=500)
    result = 'Simulation done'
    text.value = f'{result}'

# def b_plot(event):
    # result = 'Plot'
    # text.value = f'{result}'
    # plotter = plot(plotter)

def b_plot(event):
    result = 'Plot'
    text.value = f'{result}'
    return plot(plotter)
    # return f'You have clicked me {clicks} times'

pn.Column(
    button_plot,
    pn.bind(b_plot, button_plot),
)
    

button.on_click(b)
# button_plot.on_click(b_plot)

pn.Row(button, text)

button.servable()
button_plot.servable()
text.servable()
geo_pan_pv.servable()
# p.servable()
