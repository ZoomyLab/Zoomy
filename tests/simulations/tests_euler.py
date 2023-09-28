import os
import pytest
import matplotlib.pyplot as plt
import inspect
import numpy as np

from library.solver.controller import Controller
from library.solver.model import *
from library.solver.mesh import Mesh1D, Mesh2D
import library.solver.misc as misc
import library.visualization.matplotlibstyle

main_dir = os.getenv("SMPYTHON")
