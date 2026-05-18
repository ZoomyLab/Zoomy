# trace generated using paraview version 5.13.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
plt_3d_0000000000 = GetActiveSource()

# Properties modified on plt_3d_0000000000
plt_3d_0000000000.CellArrayStatus = ['b', 'h', 'p', 'u', 'v', 'w']

UpdatePipeline(time=0.0, proxy=plt_3d_0000000000)

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=plt_3d_0000000000)

# Properties modified on threshold1
threshold1.Scalars = ['CELLS', 'h']
threshold1.UpperThreshold = 0.01
threshold1.ThresholdMethod = 'Above Upper Threshold'

UpdatePipeline(time=0.0, proxy=threshold1)

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=threshold1)

# Properties modified on resampleToImage1
resampleToImage1.SamplingDimensions = [100, 100, 16]

UpdatePipeline(time=0.0, proxy=resampleToImage1)

# set active source
SetActiveSource(threshold1)

# destroy resampleToImage1
Delete(resampleToImage1)
del resampleToImage1

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=threshold1)

# Properties modified on resampleToImage1
resampleToImage1.SamplingDimensions = [100, 100, 16]

UpdatePipeline(time=0.0, proxy=resampleToImage1)

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(registrationName='WarpByScalar1', Input=resampleToImage1)

# Properties modified on warpByScalar1
warpByScalar1.ScaleFactor = 0.0

UpdatePipeline(time=0.0, proxy=warpByScalar1)

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=warpByScalar1)

# Properties modified on programmableFilter1
programmableFilter1.Script = """import numpy as np
from vtk.util import numpy_support as ns

pdi = self.GetInput()
pdo = self.GetOutput()
pdo.ShallowCopy(pdi)

points = pdo.GetPoints()
coords = ns.vtk_to_numpy(points.GetData())

# Fetch a point-data array called "0"
b = pdi.GetPointData().GetArray("b")
if b is None:
    raise RuntimeError("Point data array \'0\' not found!")
h = pdi.GetPointData().GetArray("h")
if h is None:
    raise RuntimeError("Point data array \'1\' not found!")

b = ns.vtk_to_numpy(b).astype(float)  # ensure float
h = ns.vtk_to_numpy(h).astype(float)  # ensure float

# Now apply expression: new Z = Z * arr
coords[:,2] = b + coords[:,2] * h

points.SetData(ns.numpy_to_vtk(coords))
"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

UpdatePipeline(time=0.0, proxy=programmableFilter1)

# set active source
SetActiveSource(plt_3d_0000000000)

# create a new 'Resample To Image'
resampleToImage2 = ResampleToImage(registrationName='ResampleToImage2', Input=plt_3d_0000000000)

# Properties modified on resampleToImage2
resampleToImage2.SamplingDimensions = [400, 800, 1]

UpdatePipeline(time=0.0, proxy=resampleToImage2)

# create a new 'Warp By Scalar'
warpByScalar2 = WarpByScalar(registrationName='WarpByScalar2', Input=resampleToImage2)

# Properties modified on warpByScalar2
warpByScalar2.ScaleFactor = 0.99

UpdatePipeline(time=0.0, proxy=warpByScalar2)