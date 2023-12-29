import os
import numpy as np

# import scipy.interpolate as interp
# from functools import wraps

from attr import define
from typing import Callable, Optional, Any
from types import SimpleNamespace

from sympy import MatrixSymbol

from library.misc.custom_types import FArray


@define(slots=True, frozen=False)
class IterableNamespace(SimpleNamespace):
    iterable_obj: list[Any]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterable_obj = list(self.__dict__.values())

    def __getitem__(self, key):
        return self.iterable_obj[key]

    def length(self):
        return len(self.iterable_obj)

    def get_list(self):
        return self.iterable_obj


def require(requirement):
    """
    Decorator to check if a requirement is met before executing the decorated function.

    Parameters:
    - requirement (str): The requirement string to evaluate. Should evaluate to True or False.

    Returns:
    - wrapper: The decorated function that will check the requirement before executing.
    """

    # decorator to check the assertion given in requirements given the settings
    def req_decorator(func):
        @wraps(func)
        def wrapper(settings, *args, **kwargs):
            requirement_evaluated = eval(requirement)
            if not requirement_evaluated:
                print("Requirement {}: {}".format(requirement, requirement_evaluated))
                assert requirement_evaluated
            return func(settings, *args, **kwargs)

        return wrapper

    return req_decorator


def all_class_members_identical(a, b):
    members = [
        attr
        for attr in dir(a)
        if not callable(getattr(a, attr)) and not attr.startswith("__")
    ]
    for member in members:
        m_a = getattr(a, member)
        m_b = getattr(b, member)
        if type(m_a) == np.ndarray:
            if not ((getattr(a, member) == getattr(b, member)).all()):
                print(getattr(a, member))
                print(getattr(b, member))
                assert False
        else:
            if not ((getattr(a, member) == getattr(b, member))):
                print(getattr(a, member))
                print(getattr(b, member))
                assert False


def compute_transverse_direction(normal):
    dim = normal.shape[0]
    if dim == 1:
        return np.zeros_like(normal)
    elif dim == 2:
        transverse = np.zeros((2), dtype=float)
        transverse[0] = -normal[1]
        transverse[1] = normal[0]
        return transverse
    elif dim == 3:
        cartesian_x = np.array([1, 0, 0], dtype=float)
        transverse = np.cross(normal, cartesian_x)
        return transverse
    else:
        assert False


def extract_momentum_fields_as_vectors(Q, momentum_fields, dim):
    num_fields = len(momentum_fields)
    num_momentum_eqns = int(num_fields / dim)
    Qnew = np.empty((num_momentum_eqns, dim))
    for i_eq in range(num_momentum_eqns):
        for i_dim in range(dim):
            Qnew[ i_eq, i_dim] = Q[
                momentum_fields[i_dim * num_momentum_eqns + i_eq]
            ]
    return Qnew


def projection_in_normal_and_transverse_direction(Q, momentum_fields, normal):
    dim = normal.shape[0]
    transverse_directions = compute_transverse_direction(normal)
    Q_momentum_eqns = extract_momentum_fields_as_vectors(Q, momentum_fields, dim)
    Q_normal = np.empty(
        (Q_momentum_eqns.shape[0]), dtype=float
    )
    Q_transverse = np.empty(
        (Q_momentum_eqns.shape[0]), dtype=float
    )
    for i in range(Q_momentum_eqns.shape[0]):
        Q_normal[i] = np.dot(Q_momentum_eqns[i, :], normal[:])
        Q_transverse[i] = np.dot(
            Q_momentum_eqns[i :], transverse_directions[:]
        )
    return Q_normal, Q_transverse


def projection_in_x_y_direction(Qn, Qt, normal):
    dim = normal.shape[0]
    num_momentum_fields = Qn.shape[0]
    transverse_directions = compute_transverse_direction(normal)
    Q = np.empty((num_momentum_fields * dim), dtype=float)
    for i in range(num_momentum_fields):
        for d in range(dim):
            Q[i + d * num_momentum_fields] = (
                Qn[i] * normal[d] + Qt[i] * transverse_directions[d]
            )
    return Q


def project_in_x_y_and_recreate_Q(Qn, Qt, Qorig, momentum_eqns, normal):
    Qnew = np.array(Qorig)
    Qnew[momentum_eqns] = projection_in_x_y_direction(Qn, Qt, normal)
    # Qnew = np.concatenate(
    #     [Qorig[:, 0][:, np.newaxis], projection_in_x_y_direction(Qn, Qt, normal)],
    #     axis=1,
    # )
    return Qnew


def vectorize(
    func: Callable[[list[FArray]], FArray], n_arguments=3
) -> Callable[[list[FArray]], FArray]:
    """ Note that besides vectorization, we also convert the output to a numpy array and erase the trailing 1 in the dimension for vectors (stored in sympy as matrices) """
    if n_arguments == 3:
        
        # probe has format [n_dim, [N, n_fields, 1 or n_fields (Vector or Matrix)]]
        def f(Q, Qaux, param):
            probe = np.array(func(Q[0], Qaux[0], param))
            Qout = np.zeros((Q.shape[0],) + probe.shape, dtype=probe.dtype)
            for i, (q, qaux) in enumerate(zip(Q, Qaux)):
                Qout[i] = np.array(func(q, qaux, param))
            return Qout

    elif n_arguments == 4:

        def f(Q, Qaux, normals, param):
            probe = np.array(func(Q[0], Qaux[0], normals[0], param))
            Qout = np.zeros((Q.shape[0],) + probe.shape, dtype=probe.dtype)
            for i, (q, qaux, normals) in enumerate(zip(Q, Qaux, normals)):
                Qout[i] = func(q, qaux, normals, param)
            return np.squeeze(Qout)

    return f


# def load_npy(filepath=main_dir + "/output/", filename="mesh.npy", filenumber=None):
#     if filenumber is not None:
#         full_filename = filepath + filename + "." + str(int(filenumber))
#     else:
#         full_filename = filepath + filename
#     if not os.path.exists(full_filename):
#         print("File or file path to: ", full_filename, " does not exist")
#         assert False
#     data = np.load(full_filename)
#     return data


# def write_field_to_npy(
#     field, filepath=main_dir + "/output/", filename="mesh.npy", filenumber=None
# ):
#     if filenumber is not None:
#         full_filename = filepath + filename + "." + str(int(filenumber))
#     else:
#         full_filename = filepath + filename
#     os.makedirs(filepath, exist_ok=True)
#     # the extra step over 'open' is to allow for a filename with filenumber
#     with open(full_filename, "wb") as f:
#         np.save(f, field)


# def interpolate_field_to_mesh(field, mesh_field, mesh_out):
#     interpolator = interp.interp1d(mesh_field, field)
#     return interpolator(mesh_out)
