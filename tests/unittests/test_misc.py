import numpy as np
import pytest

from library.misc.misc import *

@pytest.mark.unittest
def test_iterable_namespace():
    ns = IterableNamespace(a=lambda x: x, b=2, c='3')
    assert ns.length() == 3
    assert ns.get_list()[2] == '3'
    assert ns.b == 2
    assert ns.c == '3'
    
@pytest.mark.unittest
def test_settings():
    try: 
        ns = Settings(a=lambda x: x, b=2, c='3')
        assert True, "Settings should raise ValueError if 'name' is not provided"
    except:
        ns = Settings(name='test', a=lambda x: x, b=2, c='3')
        
    assert ns.length() == 4
    assert ns.get_list()[3] == '3'
    assert ns.name == 'test'
    assert ns.b == 2
    assert ns.c == '3'

@pytest.mark.unittest
def test_projection_in_normal_transverse_direction_and_back():
    dim = 2
    n_fields = 1 + 2 * dim
    momentum_eqns = np.array(list(range(1, n_fields)))
    N = 10
    Q = np.linspace(1, n_fields * N, n_fields * N).reshape((N, n_fields))
    angles = [0.0, np.pi / 2.0, np.pi / 4.0]
    compute_normal = lambda angle: np.array([np.cos(angle), -np.sin(angle), 0.0])
    normals = [compute_normal(angle)[:dim] for angle in angles]

    for i, normal in enumerate(normals):
        Qn, Qt = projection_in_normal_and_transverse_direction(
            Q[i], momentum_eqns, normal
        )
        Qnew = project_in_x_y_and_recreate_Q(Qn, Qt, Q[i], momentum_eqns, normal)
        assert np.allclose(Q[i], Qnew)


if __name__ == "__main__":
    test_iterable_namespace()
    test_settings()
    test_projection_in_normal_transverse_direction_and_back()

