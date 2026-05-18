# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
plt_2d_00 = GetActiveSource()

# Properties modified on plt_2d_00
plt_2d_00.CellArrayStatus = ['b', 'h', 'hu_0', 'hu_1', 'hv_0', 'hv_1']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
plt_2d_00Display = Show(plt_2d_00, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
plt_2d_00Display.Representation = 'Outline'
plt_2d_00Display.ColorArrayName = [None, '']
plt_2d_00Display.SelectTCoordArray = 'None'
plt_2d_00Display.SelectNormalArray = 'None'
plt_2d_00Display.SelectTangentArray = 'None'
plt_2d_00Display.OSPRayScaleFunction = 'PiecewiseFunction'
plt_2d_00Display.SelectOrientationVectors = 'None'
plt_2d_00Display.ScaleFactor = 0.1
plt_2d_00Display.SelectScaleArray = 'None'
plt_2d_00Display.GlyphType = 'Arrow'
plt_2d_00Display.GlyphTableIndexArray = 'None'
plt_2d_00Display.GaussianRadius = 0.005
plt_2d_00Display.SetScaleArray = [None, '']
plt_2d_00Display.ScaleTransferFunction = 'PiecewiseFunction'
plt_2d_00Display.OpacityArray = [None, '']
plt_2d_00Display.OpacityTransferFunction = 'PiecewiseFunction'
plt_2d_00Display.DataAxesGrid = 'GridAxesRepresentation'
plt_2d_00Display.PolarAxes = 'PolarAxesRepresentation'
plt_2d_00Display.ScalarOpacityUnitDistance = 0.17184120462482527

# reset view to fit data
renderView1.ResetCamera(False)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=plt_2d_00)
calculator1.AttributeType = 'Cell Data'
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = 'h+b'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
calculator1Display.Representation = 'Outline'
calculator1Display.ColorArrayName = ['CELLS', '']
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'None'
calculator1Display.ScaleFactor = 0.1
calculator1Display.SelectScaleArray = 'Result'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'Result'
calculator1Display.GaussianRadius = 0.005
calculator1Display.SetScaleArray = [None, '']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = [None, '']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityUnitDistance = 0.17184120462482527

# hide data in view
Hide(plt_2d_00, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=calculator1)
threshold1.Scalars = ['CELLS', 'Result']
threshold1.LowerThreshold = 0.5
threshold1.UpperThreshold = 1.876953125

# Properties modified on threshold1
threshold1.Scalars = ['CELLS', 'h']
threshold1.UpperThreshold = 1e-06
threshold1.ThresholdMethod = 'Above Upper Threshold'

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'
threshold1Display.ColorArrayName = ['CELLS', 'Result']
threshold1Display.LookupTable = resultLUT
threshold1Display.SelectTCoordArray = 'None'
threshold1Display.SelectNormalArray = 'None'
threshold1Display.SelectTangentArray = 'None'
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.ScaleFactor = 0.1
threshold1Display.SelectScaleArray = 'Result'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'Result'
threshold1Display.GaussianRadius = 0.005
threshold1Display.SetScaleArray = [None, '']
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityArray = [None, '']
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.ScalarOpacityFunction = resultPWF
threshold1Display.ScalarOpacityUnitDistance = 0.19057503127657266
threshold1Display.OpacityArrayName = ['CELLS', 'Result']

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(calculator1, renderView1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1399, 707)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.5, 0.5, 3.8460652149512318]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.5]
renderView1.CameraParallelScale = 0.8660254037844386

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).