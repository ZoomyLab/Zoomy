"""Shared utilities: SymPy-friendly arrays, dynamic structs, and runtime paths."""

import os
import numpy as np
import subprocess
import sys
import json
from types import SimpleNamespace
from typing import Callable, Optional, Any
from sympy.core.basic import Basic
from collections.abc import Iterable
import itertools

import sympy as sp
from sympy import MutableDenseNDimArray, tensorcontraction, tensorproduct

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.logger_config import logger

class ZArray(MutableDenseNDimArray):
    """
    Enhanced SymPy Array that supports matrix multiplication (@)
    and vector algebra (+, -) with lists and Zstructs.
    """

    def __reduce__(self):
        """Pickle support (REQ-163 SystemModel cache): sympy's NDimArray
        reduce protocol does not satisfy ZArray's required `iterable`
        argument — reconstruct from the nested list form instead.  Also fixes
        `applyfunc`-style reconstruction paths that hand `__new__` a
        `map` (see the reshape below)."""
        return (type(self), (self.tolist() if self.shape else [self._array[0]],))

    def __new__(cls, iterable, *args, **kwargs):
        # Accept lazy iterators (e.g. sympy applyfunc passes a ``map``) by
        # materializing them first; shape (if given) is handled by the
        # flattener below.
        if isinstance(iterable, map):
            iterable = list(iterable)
        # 1. Normalize Input
        if hasattr(iterable, "tolist"):
            data = iterable.tolist()
        elif hasattr(iterable, "values") and callable(iterable.values):
            data = list(iterable.values())
        elif isinstance(iterable, (list, tuple)):
            data = iterable
        else:
            raise TypeError(
                f"ZArray constructor does not support type: {type(iterable)}"
            )

        flat_list = []

        # 2. Recursive Flattening
        def flatten_and_get_shape(item, current_depth=0, shape_acc=None):
            if shape_acc is None:
                shape_acc = []

            if hasattr(item, "tolist"):
                item = item.tolist()
            elif hasattr(item, "values") and callable(item.values):
                item = list(item.values())

            if isinstance(item, (list, tuple)):
                if current_depth >= len(shape_acc):
                    shape_acc.append(len(item))
                elif shape_acc[current_depth] != len(item):
                    pass

                for sub in item:
                    flatten_and_get_shape(sub, current_depth + 1, shape_acc)
            else:
                flat_list.append(item)
            return shape_acc

        inferred_shape = tuple(flatten_and_get_shape(data))

        final_shape = inferred_shape
        if args:
            final_shape = args[0]
        elif "shape" in kwargs:
            # FIX: Use .pop() to remove 'shape' from kwargs so it isn't passed twice!
            final_shape = kwargs.pop("shape")

        return super().__new__(cls, flat_list, final_shape, **kwargs)

    def _to_array(self, other):
        # sympy MatrixBase must go through ``.tolist()`` — ``.values()``
        # returns a flat list which would strip the (rows, cols) shape
        # and break shape-matched element-wise operations.
        if isinstance(other, sp.MatrixBase):
            return sp.Array(other.tolist())
        if hasattr(other, "tolist"):
            return sp.Array(other.tolist())
        if hasattr(other, "values"):
            return sp.Array(list(other.values()))
        if isinstance(other, (list, tuple)):
            return sp.Array(other)
        return other

    def __matmul__(self, other):
        """
        Matrix Multiplication (@).
        Prioritizes robust Matrix Algebra if inputs are compatible.
        """
        # 1. Robust Matrix Path: If 'other' is a Matrix, use Matrix algebra
        #    This fixes the (N,N) @ (N,1) crash by avoiding tensorcontraction
        if isinstance(other, sp.MatrixBase) and self.rank() == 2:
            return ZArray(self.tomatrix() @ other)

        # 2. Tensor Path: Fallback for Arrays/Lists
        other_arr = self._to_array(other)
        if isinstance(other_arr, sp.NDimArray):
            # Rank 2 @ Rank 1 -> Vector (Contraction on last axis of self, first of other)
            if self.rank() == 2 and other_arr.rank() == 1:
                return ZArray(tensorcontraction(tensorproduct(self, other_arr), (1, 2)))

            # Rank 1 @ Rank 1 -> Scalar (Dot product)
            if self.rank() == 1 and other_arr.rank() == 1:
                return tensorcontraction(tensorproduct(self, other_arr), (0, 1))

            # Rank 2 @ Rank 2 -> Matrix (Standard Matrix Mult)
            if self.rank() == 2 and other_arr.rank() == 2:
                return ZArray(tensorcontraction(tensorproduct(self, other_arr), (1, 2)))

        return NotImplemented

    def __sub__(self, other):
        return ZArray(super().__add__(-1 * self._to_array(other)))

    def __add__(self, other):
        return ZArray(super().__add__(self._to_array(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return ZArray(sp.Array(self._to_array(other)) + (-1 * self))

    def tomatrix(self):
        """Return a SymPy `Matrix` (rank-2 arrays only)."""
        if self.rank() != 2:
            raise ValueError("Only rank-2 ZArrays can be converted to Matrix.")
        return sp.Matrix(self.tolist())

    def __setitem__(self, index, value):
        # 1. Determine if we are dealing with slices
        has_slice = False
        if isinstance(index, slice):
            has_slice = True
            index = (index,)
        elif isinstance(index, tuple):
            has_slice = any(isinstance(i, slice) for i in index)

        # 2. Case A: Simple Integer Indexing (Scalar Assignment)
        # This allows qauxL[0] = ... to work
        if not has_slice:
            idx = self._parse_index(index)
            # Guard against assigning a list/array to a single slot
            if isinstance(
                value, (Iterable, sp.MatrixBase, sp.NDimArray)
            ) and not isinstance(value, Basic):
                raise ValueError("Cannot assign an iterable to a single ZArray slot.")
            self._array[idx] = sp.sympify(value)
            return

        # 3. Case B: Slice Indexing
        # This restores the functionality of JacF[:, :, d] = ...
        shape = self.shape
        ranges = []
        for i, ind in enumerate(index if isinstance(index, tuple) else [index]):
            if isinstance(ind, slice):
                start, stop, step = ind.indices(shape[i])
                ranges.append(range(start, stop, step))
            else:
                ranges.append([ind])

        for i in range(len(ranges), len(shape)):
            ranges.append(range(shape[i]))

        target_coords = list(itertools.product(*ranges))

        # Prepare the values to be assigned
        if isinstance(
            value, (Iterable, sp.MatrixBase, sp.NDimArray)
        ) and not isinstance(value, Basic):
            # --- CRITICAL FIX: Ensure value is flattened to scalars ---

            if isinstance(value, sp.NDimArray):
                # SymPy Arrays (and ZArray) store data flat in _array. Use that.
                # list(value) returns sub-arrays (rows), which caused the size mismatch.
                val_list = list(value._array)

            elif isinstance(value, sp.MatrixBase):
                # SymPy Matrices iterate flat by default
                val_list = list(value)

            elif hasattr(value, "flat"):
                # NumPy arrays
                val_list = list(value.flat)

            else:
                # Fallback for nested lists: Recursive flatten
                val_list = []

                def smooth(v):
                    for elt in v:
                        if isinstance(elt, (list, tuple, sp.NDimArray)):
                            smooth(elt)
                        else:
                            val_list.append(elt)

                smooth(value)

            # Validate size
            if len(val_list) != len(target_coords):
                raise ValueError(
                    f"Source value length ({len(val_list)}) does not match slice size ({len(target_coords)})"
                )

            # Assign
            for coord, val in zip(target_coords, val_list):
                self._array[self._parse_index(coord)] = sp.sympify(val)
        else:
            # Broadcast scalar
            sym_val = sp.sympify(value)
            for coord in target_coords:
                self._array[self._parse_index(coord)] = sym_val
                
    def flatten(self):
        """
        Flattens the N-dimensional array into a 1D ZArray.
        Uses the internal _array list which is inherently flat in SymPy NDimArrays.
        """
        return ZArray(list(self._array))
    
    def subs(self, *args, **kwargs):
        """
        Substitutes expressions within the ZArray.
        Supports SymPy's standard subs API:
        .subs(old, new), .subs({old: new}), or .subs([(old, new)])
        """
        # 1. Store the shape
        current_shape = self.shape

        # 2 & 3. Flatten implicitly (using _array) and substitute for each item
        new_elements = []
        for item in self._array:
            if hasattr(item, "subs"):
                new_elements.append(item.subs(*args, **kwargs))
            else:
                new_elements.append(item)

        # 4 & 5. Restore the shape and return a new ZArray
        return ZArray(new_elements, shape=current_shape)

    def xreplace(self, rule):
        """Element-wise xreplace — sympy's fast pattern-free substitution.

        ZArray inherits from `MutableDenseNDimArray` which does not
        define `xreplace`; we apply it per-element and rebuild.
        """
        current_shape = self.shape
        new_elements = []
        for item in self._array:
            if hasattr(item, "xreplace"):
                new_elements.append(item.xreplace(rule))
            else:
                new_elements.append(item)
        return ZArray(new_elements, shape=current_shape)

    def replace(self, query, value, **kwargs):
        """Element-wise `sympy.Basic.replace` — pattern-based substitution.

        `MutableDenseNDimArray` defines no `replace`; we apply sympy's
        per-element `replace(query, value, …)` and rebuild, exactly as
        `xreplace` / `subs` do.  `query` / `value` may be the
        callable form (`lambda expr: bool` + `lambda expr: expr`) so a
        whole *family* (e.g. every `h**(-n)`) is rewritten in one pass.
        """
        current_shape = self.shape
        new_elements = []
        for item in self._array:
            if hasattr(item, "replace"):
                new_elements.append(item.replace(query, value, **kwargs))
            else:
                new_elements.append(item)
        return ZArray(new_elements, shape=current_shape)

    def doit(self, **kwargs):
        """Element-wise `.doit()` — evaluates unevaluated Derivative
        / Integral / Function calls.  Same pattern as `xreplace`."""
        current_shape = self.shape
        new_elements = []
        for item in self._array:
            if hasattr(item, "doit"):
                new_elements.append(item.doit(**kwargs))
            else:
                new_elements.append(item)
        return ZArray(new_elements, shape=current_shape)

    def expand(self, **kwargs):
        """Element-wise `sp.expand`."""
        current_shape = self.shape
        new_elements = [sp.expand(item, **kwargs) if hasattr(item, "expand")
                        else item for item in self._array]
        return ZArray(new_elements, shape=current_shape)

    def diff(self, *args, **kwargs):
        """Element-wise differentiation — returns a ZArray of the same shape."""
        current_shape = self.shape
        new_elements = [sp.diff(item, *args, **kwargs) for item in self._array]
        return ZArray(new_elements, shape=current_shape)

    @property
    def is_zero_matrix(self):
        """All entries are symbolically zero.  Mirrors
        `sp.MatrixBase.is_zero_matrix` so consumers that branch on
        zero operators (e.g. splitter's algebraic-row detection)
        work uniformly."""
        return all(item == 0 for item in self._array)

    def __eq__(self, other):
        """Element-wise equality.  Cross-type comparison with sympy
        `MatrixBase` works via shared `.tolist()` representation
        (both produce nested lists of sympy expressions).  Falls back
        to the inherited NDimArray equality for ZArray-to-ZArray."""
        if isinstance(other, sp.MatrixBase):
            try:
                return self.tolist() == other.tolist()
            except Exception:
                return NotImplemented
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    @property
    def flat(self):
        """Flat."""
        return list(self)

def get_main_directory():
    # 1. Environment variable overrides everything
    """Get main directory."""
    env_dir = os.getenv("ZOOMY_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    # 2. Detect Git root if available
    try:
        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return git_root
    except Exception:
        pass

    # 3. Handle Jupyter notebooks vs scripts
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip and hasattr(ip, "run_line_magic"):
            notebook_dir = ip.run_line_magic("pwd", "")
            if notebook_dir:
                return notebook_dir
    except Exception:
        pass

    # 4. Fallback: use the directory of the current script
    if hasattr(sys.modules["__main__"], "__file__"):
        return os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__))

    # 5. Ultimate fallback: CWD
    return os.getcwd()


class Zstruct(SimpleNamespace):
    """
    Dynamic structure that behaves like a Namespace but supports
    iteration and index access (based on insertion order).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._symbolic_name = None

    def __repr__(self):
        clean_dict = self._filter_dict()
        args = ", ".join(f"{k}={v!r}" for k, v in clean_dict.items())
        return f"Zstruct({args})"

    def __getitem__(self, key):
        """Hook `__getitem__`.

        Supports lookup by string name, by integer index, or by any
        object with a `.name` attribute that matches a key (e.g. a
        sympy `Symbol`).  The Symbol path lets callers write
        `z[sym]` after iterating `for sym in z` — the iterator
        yields *values* (Symbols), so re-indexing them needs to round-
        trip through `Symbol.name`.
        """
        if isinstance(key, str):
            return getattr(self, key)
        # Symbol (or any object with .name) → look up by name when the
        # name matches a key.
        name = getattr(key, "name", None)
        if isinstance(name, str) and name in self.keys():
            return getattr(self, name)
        return self.values()[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            setattr(self, key, value)
        elif isinstance(key, int):
            keys = self.keys()
            if 0 <= key < len(keys):
                setattr(self, keys[key], value)
            else:
                raise IndexError(f"Index {key} out of range")
        else:
            raise TypeError("Indices must be integers or strings")

    def _filter_dict(self):
        """Returns internal dict excluding private/hidden attributes."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def length(self):
        return len(self._filter_dict())

    def get_list(self, recursive: bool = True):
        """Nested list view of values (used when flattening symbolic trees)."""
        if recursive:
            output = []
            for item in self.values():
                if hasattr(item, "get_list"):
                    output.append(item.get_list(recursive=True))
                else:
                    output.append(item)
            return output
        else:
            return self.values()

    def as_dict(self, recursive: bool = True):
        """Serialize to plain dicts (recursive for nested `Zstruct`)."""
        data = self._filter_dict()
        if recursive:
            output = {}
            for key, value in data.items():
                if hasattr(value, "as_dict"):
                    output[key] = value.as_dict(recursive=True)
                else:
                    output[key] = value
            return output
        else:
            return data

    def items(self, recursive: bool = False):
        """Items."""
        return self.as_dict(recursive=recursive).items()

    def keys(self):
        """Keys."""
        return list(self._filter_dict().keys())

    def values(self):
        """Values."""
        return list(self._filter_dict().values())

    def contains(self, key):
        """Contains."""
        return hasattr(self, key)

    def update(self, zstruct, recursive: bool = True):
        """Update."""
        if not hasattr(zstruct, "as_dict") and not isinstance(zstruct, dict):
            raise TypeError("Input must be a Zstruct or a dictionary.")
        data = (
            zstruct.as_dict(recursive=False) if hasattr(zstruct, "as_dict") else zstruct
        )
        if recursive:
            for key, value in data.items():
                if hasattr(self, key):
                    current_value = getattr(self, key)
                    if isinstance(current_value, Zstruct) and isinstance(
                        value, Zstruct
                    ):
                        current_value.update(value, recursive=True)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        else:
            for key, value in data.items():
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, d):
        """Build a `Zstruct` tree from nested dicts."""
        if not isinstance(d, dict):
            raise TypeError("Input must be a dictionary.")
        data = d.copy()
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = Zstruct.from_dict(v)
        return cls(**data)


class Settings(Zstruct):
    """Run configuration (output paths, snapshots, etc.) for drivers and solvers."""

    def __init__(self, **kwargs):
        if "output" not in kwargs:
            logger.warning("No 'output' Zstruct found in Settings. Defaulting.")
            kwargs["output"] = Zstruct(
                directory="output",
                filename="simulation",
                snapshots=2,
                clean_directory=True,
            )
        elif not isinstance(kwargs["output"], Zstruct):
            if isinstance(kwargs["output"], dict):
                kwargs["output"] = Zstruct(**kwargs["output"])

        output = kwargs["output"]
        # ``clean_directory`` defaults to TRUE.  A run that inherits stale
        # timestep directories does not fail -- it reports numbers computed
        # from someone else's leftovers.  That has already happened: a coupling
        # run picked up 118 stale directories (t up to 30 s), its mass audit
        # read a t=30 frame as the "final time", and it reported a FABRICATED
        # +76.7 % mass drift; the same pairing gives -2.487e-13 once the
        # directory is wiped.  A silent wrong answer is worse than a lost output
        # directory.  The previous defaults were also self-inconsistent:
        # omitting ``output`` entirely gave True, while passing an ``output``
        # Zstruct without the key gave False.
        defaults = {
            "directory": "output",
            "filename": "simulation",
            "clean_directory": True,
            "snapshots": 2,
        }

        for key, val in defaults.items():
            if not output.contains(key):
                logger.warning(
                    f"No '{key}' attribute found in Settings.output Zstruct. Default: '{val}'"
                )
                setattr(output, key, val)

        super().__init__(**kwargs)

    @classmethod
    def default(cls):
        """Reasonable defaults for local runs and tests."""
        return cls(
            output=Zstruct(
                directory="output",
                filename="simulation",
                snapshots=2,
                clean_directory=True,
            )
        )

    @classmethod
    def from_json(cls, filepath):
        """Load settings from a JSON file on disk."""
        with open(filepath, "r") as f:
            data = json.load(f)
        if "output" in data:
            data["output"] = Zstruct(**data["output"])
        return cls(**data)

    def write_json(self, filepath=None):
        """Persist settings next to the simulation output unless `filepath` is given."""
        if filepath is None:
            filepath = os.path.join(
                get_main_directory(), self.output.directory, "settings.json"
            )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        settings_dict = self.as_dict()
        with open(filepath, "w") as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings exported to {filepath}")
