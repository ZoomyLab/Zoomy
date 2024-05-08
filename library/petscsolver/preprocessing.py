import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

import library.mesh.mesh as petscMesh

def generate_fields(plex, Qsize, Qauxsize):
    numFields = 2 #Q, Qaux
    numComp = [Qsize, Qauxsize]
    numDof = [0, 0, Qsize] + [0, 0, Qauxsize]
    plex.setNumFields(numFields)
    sec = plex.createSection(numComp, numDof, None, None, None, None)
    sec.setFieldName(0, "Q")
    sec.setFieldName(1, "Qaux")
    sec.setName('Fields')
    plex.setSection(sec)
    return plex

def apply_initial_condition_Q(func, plex):
    (cStart, cEnd) = plex.getHeightStratum(0)
    for i, c in enumerate(range(cStart, cEnd)):
        volume, centroid, normal = plex.computeCellGeometryFVM(c)



if __name__ == "__main__":
    path = '/home/ingo/Git/SMM/shallow-moments-simulation/meshes/quad_2d/mesh_coarse.msh'
    plex, boundary_dict, ghost_cells_dict = petscMesh.load_gmsh(path)
