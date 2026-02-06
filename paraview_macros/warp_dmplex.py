# trace generated using paraview version 5.12.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
sol_3dvtuseries = FindSource('sol_3d.vtu.series')

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=sol_3dvtuseries)

UpdatePipeline(time=0.1063296, proxy=cellDatatoPointData1)

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=cellDatatoPointData1)

# Properties modified on programmableFilter1
programmableFilter1.Script = """import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa

# 1. Get the raw VTK input object
raw_input = self.GetInputDataObject(0, 0)

# 2. Initialize the output with the input's topology and fields
output.ShallowCopy(raw_input)

# 3. Wrap input/output for easy NumPy-style access
input_wrapped = dsa.WrapDataObject(raw_input)
output_wrapped = dsa.WrapDataObject(output)

# 4. Access the fields (automatically converted to NumPy-like arrays)
b = input_wrapped.PointData['State3DState.b']
h = input_wrapped.PointData['State3DState.h']

# 5. Get the coordinates and transform
# input_wrapped.Points is the coordinate array
old_coords = input_wrapped.Points
new_coords = np.copy(old_coords)

# Apply the math: z_new = b + z_old * h
new_coords[:, 2] = b + old_coords[:, 2] * h

# 6. Set the transformed points back to the output
output_wrapped.Points = new_coords"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

UpdatePipeline(time=0.1063296, proxy=programmableFilter1)