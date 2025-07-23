import os
import numpy as np

# import scipy.interpolate as interp
# from functools import wraps

from attr import define
from typing import Callable, Optional, Any
from types import SimpleNamespace

from sympy import MatrixSymbol

from library.misc.custom_types import FArray
from library.misc.static_class import register_static_pytree





@register_static_pytree
@define(slots=True, frozen=False, kw_only=True)
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

    def to_value_dict(self, values):
        out = {k: values[i] for i, k in enumerate(vars(self).keys())}
        return out
    
@register_static_pytree
@define(slots=True, frozen=False, kw_only=True)
class Settings(IterableNamespace):
    """
    Settings class for the application.
    
    Args: 
        **kwargs: Arbitrary keyword arguments to set as attributes.
        
    Returns:
        An `IterableNamespace` instance.
    """
    
    def __init__(self, **kwargs):
        # assert that kwargs constains name
        if 'name' not in kwargs:
            raise ValueError("Settings must have a 'name' attribute.")
        super().__init__(**kwargs)


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
            Qnew[i_eq, i_dim] = Q[momentum_fields[i_dim * num_momentum_eqns + i_eq]]
    return Qnew


def projection_in_normal_and_transverse_direction(Q, momentum_fields, normal):
    dim = normal.shape[0]
    transverse_directions = compute_transverse_direction(normal)
    Q_momentum_eqns = extract_momentum_fields_as_vectors(Q, momentum_fields, dim)
    Q_normal = np.zeros((Q_momentum_eqns.shape[0]), dtype=float)
    Q_transverse = np.zeros((Q_momentum_eqns.shape[0]), dtype=float)
    for d in range(dim):
        Q_normal += Q_momentum_eqns[:, d] * normal[d]
        Q_transverse += Q_momentum_eqns[:, d] * transverse_directions[d]
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
    return Qnew
