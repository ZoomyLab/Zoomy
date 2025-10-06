# trace generated using paraview version 5.10.0-451-g61c424a2e6
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
programmableFilter1 = FindSource('ProgrammableFilter1')

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=programmableFilter1)
calculator1.Function = ''

# find source
cellDatatoPointData1 = FindSource('CellDatatoPointData1')

# Properties modified on calculator1
calculator1.Function = '"1" * iHat + "2" * jHat + "3" * kHat'
