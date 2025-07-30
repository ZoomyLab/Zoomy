import os
import numpy as np
from typing import Union, Type

from library.model.models.base import Model, ModelGUI
from library.model.models.advection import Advection
from library.model.models.shallow_water import ShallowWater, ShallowWater2d
from library.model.models.shallow_water_topo import ShallowWaterTopo, ShallowWaterTopo2d
# 

# from library.model.models.shallow_moments_sediment import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.mesh.fvm_mesh import Mesh


def create_default_mesh_and_model(
    dimension: int = 1,
    cls: Type[Model] = Advection,
    fields: Union[int, list] = 1,
    aux_fields: Union[int, list] = 0,
    parameters: Union[int, list, dict] = 0,
    settings: dict = {},
):
    main_dir = os.getenv("SMS")
    assert main_dir != ""
    ic = IC.Constant()

    bc_tags = ["left", "right", "top", "bottom"][: 2 * dimension]
    bcs = BC.BoundaryConditions([BC.Wall(physical_tag=tag) for tag in bc_tags])
    if dimension == 1:
        mesh = Mesh.create_1d((-1, 1), 10)
    elif dimension == 2:
        mesh = Mesh.load_mesh(
            os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
            "quad",
            2,
            bc_tags,
        )
    else:
        assert False
    model = cls(
        dimension=dimension,
        fields=fields,
        aux_fields=aux_fields,
        parameters=parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings=settings,
    )
    n_elements = mesh.n_elements

    Q = np.linspace(1, fields * n_elements, fields * n_elements).reshape(
        n_elements, fields
    )
    Qaux = np.zeros((Q.shape[0], model.aux_variables.length()))
    parameters = model.parameter_values
    num_normals = mesh.element_n_neighbors
    normals = np.array(
        [mesh.element_face_normals[:, i] for i in range(mesh.n_faces_per_element)]
    )

    model.initial_conditions.apply(Q, mesh.element_center)
    # model.boundary_conditions.apply()
    return mesh, model, Q, Qaux, parameters, num_normals, normals
