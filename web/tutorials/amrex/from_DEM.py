# ---
# title: "Simulate from DEM"
# author: Ingo Steldermann
# date: 08/28/2025
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown] vscode={"languageId": "raw"}
# # Simulate from FEM

# %% [markdown]
# ## Imports
#
#
# :::{.callout-warning}
# You need to modify  `ZOOMY_AMREX_HOME` below to point to your local AMReX installation
# :::

# %%
import os
from pathlib import Path
import sys
import rasterio
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), '../../..')
os.environ['ZOOMY_DIR'] = os.path.join(os.getcwd(), '../../..')
os.environ['ZoomyLog'] = 'Default'
os.environ['ZoomyLogLevel'] = 'INFO'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['ZOOMY_AMREX_HOME'] = '/home/is086873/MBD/Git/amrex'


project_root = Path.cwd().parents[2]  # 0=current, 1=.., 2=../..
sys.path.append(str(project_root))

# %%
# | code-fold: true
# | code-summary: "Load packages"
# | output: false

import os
import numpy as np
from sympy import Matrix

from library.fvm.solver import Settings
from library.model.models.shallow_water import ShallowWaterEquations
from library.model.models.shallow_moments_topo import ShallowMomentsTopo, ShallowMomentsTopoNumerical


import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.misc.misc import Zstruct
import library.transformation.to_c as trafo
from tutorials.amrex.helper import create_artificial_raster, show_raster, transform_tiff

import argparse

parser = argparse.ArgumentParser(description="Argparse.")
parser.add_argument("-l", "--level", type=int, default=0, help="Level of the SME")

args = parser.parse_args()

level = args.level

# %% [markdown]
# ## Read raster data

# %%
main_dir = os.getenv("ZOOMY_DIR")
dem_path = os.path.join(main_dir, 'data/ca_elev.tif')
ic_water_path = os.path.join(main_dir, 'data/ca_debrisflow.tif')
angle = 0.0

# dem_path = os.path.join(main_dir, 'data/evel_artificial.tif')
# ic_water_path = os.path.join(main_dir, 'data/release_artificial.tif')
# N = 100
# dx = 5.
# M = N * dx
# create_artificial_raster(lambda x, y: 1*np.exp((-(x+0.5*M)**2-y**2)/M/10), (-M, M, -M, M), dx, ic_water_path)
# create_artificial_raster(lambda x, y: 100 * np.exp(-(x+M)**2/M**2), (-M, M, -M, M), dx, dem_path)

# create_artificial_raster(lambda x, y: 1.*np.exp((-(x-1000)**2-(y-3000)**2)/10000.), (0, 2000, 0, 4000), 5, ic_water_path)
# ic_water_path, _ = transform_tiff(ic_water_path, tilt=False)


zoom = [[0, 800], [700,1100]]  # [ymin,ymax], [xmin,xmax]
dem_path, angle = transform_tiff(dem_path, tilt=True, scale=1, zoom=zoom)
ic_water_path, _ = transform_tiff(ic_water_path, tilt=False, scale=1, zoom=zoom)
print(f'Tilt angle (degrees): {angle:.2f}')

# show_raster(dem_path)
# show_raster(ic_water_path)


# %% [markdown]
# ## Model definition

# %%
# Currently, BCs are not implemented in AMReX and periodic BCs are applied
bcs = BC.BoundaryConditions(
    [
        BC.Extrapolation(physical_tag="N"),
        BC.Extrapolation(physical_tag="S"),
        BC.Extrapolation(physical_tag="E"),
        BC.Extrapolation(physical_tag="W"),
    ]
)


class MyModel(ShallowMomentsTopoNumerical):
    def source(self):
        out = Matrix([0 for i in range(self.n_variables)])
        out += self.inclination()
        out += self.newtonian()
        out += self.slip_mod()
        # out += self.chezy()
        return self.substitute_precomputed_denominator(out, self.variables[1], self.aux_variables.hinv)
         

model = MyModel(
    level=level,
    boundary_conditions=bcs,
    parameters=Zstruct(ey=np.sin(np.radians(-angle)), ez=np.cos(np.radians(-angle)), nu=0.000001, lamda=1/1000., rho=1000, c_slipmod=1/30, C=300),
    # aux_variables = ['hinv'] + [f'dalpha_{i}_dx' for i in range(level+1)] + [f'dbeta_{i}_dy' for i in range(level+1)],
    aux_variables = ['hinv'],
    
)



# %%
print(model.parameters.keys())
print(model.parameter_values)

# %%
model.source()

# %%
(model.eigenvalues()[2])

# %%
model.quasilinear_matrix()

# %% [markdown]
# ## Code transformation and AMReX compilation
#
# currently, we always "clean" and compile from scratch. Comment out "make clean" to disable.

# %%

import shutil
from pathlib import Path
settings = Settings(name="ShallowMoments", output=Zstruct(directory=f"outputs/amrex_{level}"))
source_dir = Path(os.path.join(main_dir, 'library/amrex/Exec'))
output_dir = Path(os.path.join(main_dir, settings.output.directory))

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

trafo.to_c(model, settings)
main_dir = os.getenv("ZOOMY_DIR")
os.environ['ZOOMY_AMREX_MODEL'] = os.path.join(main_dir, os.path.join(settings.output.directory, '.c_interface'))

# %%
import subprocess

base = os.path.join(os.environ['ZOOMY_DIR'], 'library/amrex/')
cmds = [
    f"cp ../../{settings.output.directory}/.c_interface/model.h ./Source/model.h",
    "source ~/.zshrc",
    "source setup.sh",
    "cd Exec",
    # "make clean",
    "make"
]

subprocess.run(" && ".join(cmds), shell=True, executable="/bin/zsh", cwd=base)



# %% [markdown]
# ## Prepare raster data for AMReX
#
# ... and copy stuff to the output directory ...

# %%


output_dir.mkdir(parents=True, exist_ok=True)  # make sure output folder exists

for item in source_dir.iterdir():
    dest = output_dir / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)
        
shutil.copy(dem_path, output_dir)
shutil.copy(ic_water_path, output_dir)

print("The simulation output will be written to: ", output_dir)

from library.amrex.preprocess_rasterdata import preprocess
preprocess(os.path.join(output_dir, 'inputs'), dem_path, ic_water_path)


# %% [markdown]
# ## Do the Simulation
#
# run in the notebook or as a batch job

# %%
run_in_notebook = False

# %%
if (run_in_notebook):
    base = os.path.join(os.environ['ZOOMY_DIR'], 'library/amrex/')
    cmds = [
        "source ~/.zshrc",
        "source setup.sh",
        f"cd ../../{settings.output.directory}",
        "./clean.sh",
        "mpiexec -np 4 ./main3d.gnu.MPI.ex inputs",
    ]

    subprocess.run("\n".join(cmds), shell=True, executable="/bin/zsh", cwd=base)
else:
    
    base = os.path.join(os.environ['ZOOMY_DIR'], 'library/amrex/')
    cmds = [
        "source ~/.zshrc",
        "source setup.sh",
        f"cd ../../{settings.output.directory}",
        "./clean.sh",
        "sbatch batch.sh < inputs",
    ]

    subprocess.run("\n".join(cmds), shell=True, executable="/bin/zsh", cwd=base)

        

# %%
