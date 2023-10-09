from library.models.base import *

# from library.models.swe import *
# from library.models.smm import *

import library.initial_conditions as IC
import library.boundary_conditions as BC
from library.mesh import Mesh


def create_default_mesh_and_model(
    dimension: int, cls: Model, fields, aux_fields, parameters, momentum_eqns
):
    main_dir = os.getenv("SMS")
    ic = IC.Constant()

    bc_tags = ["left", "right", "top", "bottom"][: 2 * dimension]
    bcs = BC.BoundaryConditions(
        [BC.Wall(physical_tag=tag, momentum_eqns=momentum_eqns) for tag in bc_tags]
    )
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
    )
    n_ghosts = model.boundary_conditions.initialize(mesh)

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, fields * n_all_elements, fields * n_all_elements).reshape(
        n_all_elements, fields
    )
    Qaux = np.zeros((Q.shape[0], aux_fields))
    parameters = model.parameter_defaults
    num_normals = mesh.element_n_neighbors
    normals = np.array(
        [mesh.element_edge_normal[:, i] for i in range(mesh.n_nodes_per_element)]
    )

    model.initial_conditions.apply(Q, mesh.element_centers)
    model.boundary_conditions.apply(Q)
    return mesh, model, Q, Qaux, parameters, num_normals, normals
