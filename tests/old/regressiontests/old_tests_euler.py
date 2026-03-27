import os
import pytest
import matplotlib.pyplot as plt
import inspect
import numpy as np

from solver.controller import Controller
from solver.model import *
from solver.mesh import Mesh1D, Mesh2D
import solver.misc as misc
import gui.visualization.matplotlibstyle

main_dir = os.getenv("SMPYTHON")
