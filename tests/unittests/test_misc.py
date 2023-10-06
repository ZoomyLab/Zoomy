import numpy as np

from library.misc import *


def test_projection_in_normal_transverse_direction_and_back():
    dim = 2
    n_fields = 1 + 2 * dim
    momentum_eqns = np.array(list(range(1, n_fields)))
    N = 10
    Q = np.linspace(1, n_fields * N, n_fields * N).reshape((N, n_fields))
    angles = [0.0, np.pi / 2.0, np.pi / 4.0]
    compute_normal = lambda angle: np.array([np.cos(angle), -np.sin(angle), 0.0])
    normals = [compute_normal(angle)[:dim] for angle in angles]
    cartesian = compute_normal(0.0)[:dim]
    cartesian = np.repeat(cartesian[np.newaxis, :], N, axis=0)

    for normal in normals:
        normal = np.repeat(normal[np.newaxis, :], N, axis=0)
        Qn, Qt = projection_in_normal_and_transverse_direction(Q, momentum_eqns, normal)
        Qnew = project_in_x_y_and_recreate_Q(Qn, Qt, Q, momentum_eqns, normal)
        assert np.allclose(Q, Qnew)


if __name__ == "__main__":
    test_projection_in_normal_transverse_direction_and_back()

