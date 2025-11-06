import os
import numpy as np

# import scipy.interpolate as interp
# from functools import wraps

from attr import define
from typing import Callable, Optional, Any
from types import SimpleNamespace

from sympy import MatrixSymbol
from sympy import MutableDenseNDimArray as ZArray

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.logger_config import logger






@define(slots=True, frozen=False, kw_only=True)
class Zstruct(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        return self.values()[key]

    def length(self):
        return len(self.values())

    def get_list(self, recursive: bool = True):
        if recursive:
            output = []
            for item in self.values():
                if hasattr(item, 'get_list'):
                    # If item is a Zstruct or similar, call get_list recursively
                    output.append(item.get_list(recursive=True))
                else:
                    output.append(item)
            return output
        else:
            return self.values()
        
    def as_dict(self, recursive: bool = True):
        if recursive:
            output = {}
            for key, value in self.items():
                if hasattr(value, 'as_dict'):
                    # If value is a Zstruct or similar, call as_dict recursively
                    output[key] = value.as_dict(recursive=True)
                else:
                    output[key] = value
            return output
        else:
            return self.__dict__

    
    def items(self, resursive: bool = False):
        return self.as_dict(recursive=resursive).items()
    
    def keys(self):
        return list(self.as_dict(recursive=False).keys())
    
    def values(self):
        return list(self.as_dict(recursive=False).values())
    
    def contains(self, key):
        if self.as_dict(recursive=False).get(key) is not None:
            return True
        return False
    
    def update(self, zstruct, recursive: bool = True):
        """
        Update the current Zstruct with another Zstruct or dictionary.
        """
        if not isinstance(zstruct, Zstruct):
            raise TypeError("zstruct must be a Zstruct or a dictionary.")
        
        if recursive:
            # Update each attribute recursively
            for key, value in zstruct.as_dict(recursive=False).items():
                if hasattr(self, key):
                    current_value = getattr(self, key)
                    if isinstance(current_value, Zstruct) and isinstance(value, Zstruct):
                        # If both are Zstructs, update recursively
                        current_value.update(value, recursive=True)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        else:
            # Update only the top-level attributes
            for key, value in zstruct.as_dict(recursive=False).items():
                setattr(self, key, value)
                
    @classmethod
    def from_dict(cls, d):
        """
        Create a Zstruct recursively from a dictionary.
        
        Args:
            d (dict): Dictionary.
        
        Returns:
            Zstruct: An instance of Zstruct.
        """
        if not isinstance(d, dict):
            raise TypeError("Input must be a dictionary.")
        
        # Convert the dictionary to a Zstruct
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = Zstruct.from_dict(v)     
        return cls(**d)

    
@define(slots=True, frozen=False, kw_only=True)
class Settings(Zstruct):
    """
    Settings class for the application.
    
    Args: 
        **kwargs: Arbitrary keyword arguments to set as attributes.
        
    Returns:
        An `IterableNamespace` instance.
    """
    
    def __init__(self, **kwargs):
        # assert that kwargs constains name
        if 'output' not in kwargs or not isinstance(kwargs['output'], Zstruct):
            logger.warning("No 'output' Zstruct found in Settings. Default: Zstruct(directory='output', filename='simulation', clean_directory=False)")
            kwargs['output'] = Zstruct(directory='output', filename='simulation', clean_directory=True)
        output = kwargs['output']
        if not output.contains('directory'):
            logger.warning("No 'directory' attribute found in output Zstruct. Default: 'output'")
            kwargs['output'] = Zstruct(directory='output', **output.as_dict())
        if not output.contains('filename'):
            logger.warning("No 'filename' attribute found in output Zstruct. Default: 'simulation'")
            kwargs['output'] = Zstruct(filename='simulation', **output.as_dict())
        if not output.contains('clean_directory'):
            logger.warning("No 'clean_directory' attribute found in output Zstruct. Default: False")
            kwargs['output'] = Zstruct(clean_directory=False, **output.as_dict())
        super().__init__(**kwargs)
        
    @classmethod
    def default(cls):
        """
        Returns a default Settings instance.
        """
        return cls(
            output=Zstruct(directory='output', filename='simulation', clean_directory=False)
        )
    

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
