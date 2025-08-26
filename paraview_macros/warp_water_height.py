# trace generated using paraview version 5.10.0-451-g61c424a2e6
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
out_3dvtkseries = FindSource('out_3d.vtk.series')

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=out_3dvtkseries)
cellDatatoPointData1.CellDataArraytoprocess = ['0', '1', '2', '3', '4', 'aux_0']

UpdatePipeline(time=4.000198159390415, proxy=cellDatatoPointData1)

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=cellDatatoPointData1)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
programmableFilter1.Script = """import numpy as np
from vtk.util import numpy_support as ns

pdi = self.GetInput()
pdo = self.GetOutput()
pdo.ShallowCopy(pdi)

points = pdo.GetPoints()
coords = ns.vtk_to_numpy(points.GetData())

# Fetch a point-data array called "0"
arr_vtk = pdi.GetPointData().GetArray("0")
if arr_vtk is None:
    raise RuntimeError("Point data array \'0\' not found!")

arr = ns.vtk_to_numpy(arr_vtk).astype(float)  # ensure float

# Now apply expression: new Z = Z * arr
coords[:,2] = coords[:,2] * arr

points.SetData(ns.numpy_to_vtk(coords))
"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''
