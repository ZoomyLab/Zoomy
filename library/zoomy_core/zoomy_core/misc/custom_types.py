"""NumPy typing aliases used across mesh, I/O, and model code."""

import numpy as np
import numpy.typing as npt

FArray = npt.NDArray[np.float64]
IArray = npt.NDArray[np.int64]
CArray = npt.NDArray[np.str_]
