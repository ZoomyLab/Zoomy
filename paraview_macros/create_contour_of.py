# trace generated using paraview version 5.10.0-451-g61c424a2e6
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
mono = FindSource('mono')

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=mono)
threshold1.Scalars = ['POINTS', 'p']
threshold1.LowerThreshold = -1.1787300109863281
threshold1.UpperThreshold = 950.616943359375

# Properties modified on threshold1
threshold1.Scalars = ['POINTS', 'alpha.water']
threshold1.UpperThreshold = 0.5
threshold1.ThresholdMethod = 'Above Upper Threshold'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'
threshold1Display.ColorArrayName = ['POINTS', 'p']
threshold1Display.LookupTable = pLUT
threshold1Display.SelectTCoordArray = 'None'
threshold1Display.SelectNormalArray = 'None'
threshold1Display.SelectTangentArray = 'None'
threshold1Display.OSPRayScaleArray = 'p'
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.SelectOrientationVectors = 'U'
threshold1Display.ScaleFactor = 0.1
threshold1Display.SelectScaleArray = 'p'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'p'
threshold1Display.GaussianRadius = 0.005
threshold1Display.SetScaleArray = ['POINTS', 'p']
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityArray = ['POINTS', 'p']
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.ScalarOpacityFunction = pPWF
threshold1Display.ScalarOpacityUnitDistance = 0.08914941769641316
threshold1Display.OpacityArrayName = ['POINTS', 'p']
threshold1Display.SelectInputVectors = ['POINTS', 'U']
threshold1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [0.6544992327690125, 0.0, 0.5, 0.0, 950.616943359375, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [0.6544992327690125, 0.0, 0.5, 0.0, 950.616943359375, 1.0, 0.5, 0.0]

# hide data in view
Hide(mono, renderView1)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=threshold1)
resampleToImage1.SamplingBounds = [0.0, 1.0, 0.0, 0.09666666388511658, 0.0, 1.0]

# Properties modified on resampleToImage1
resampleToImage1.SamplingDimensions = [1, 100, 2000]

# show data in view
resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
resampleToImage1Display.Representation = 'Slice'
resampleToImage1Display.ColorArrayName = ['POINTS', 'p']
resampleToImage1Display.LookupTable = pLUT
resampleToImage1Display.SelectTCoordArray = 'None'
resampleToImage1Display.SelectNormalArray = 'None'
resampleToImage1Display.SelectTangentArray = 'None'
resampleToImage1Display.OSPRayScaleArray = 'p'
resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
resampleToImage1Display.SelectOrientationVectors = 'U'
resampleToImage1Display.ScaleFactor = 0.09999989999999999
resampleToImage1Display.SelectScaleArray = 'p'
resampleToImage1Display.GlyphType = 'Arrow'
resampleToImage1Display.GlyphTableIndexArray = 'p'
resampleToImage1Display.GaussianRadius = 0.004999994999999999
resampleToImage1Display.SetScaleArray = ['POINTS', 'p']
resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.OpacityArray = ['POINTS', 'p']
resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
resampleToImage1Display.ScalarOpacityUnitDistance = 0.01723997378730899
resampleToImage1Display.ScalarOpacityFunction = pPWF
resampleToImage1Display.OpacityArrayName = ['POINTS', 'p']
resampleToImage1Display.IsosurfaceValues = [475.30816650390625]
resampleToImage1Display.SliceFunction = 'Plane'
resampleToImage1Display.Slice = 999
resampleToImage1Display.SelectInputVectors = ['POINTS', 'U']
resampleToImage1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 950.6163330078125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 950.6163330078125, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
resampleToImage1Display.SliceFunction.Origin = [5.000000000143778e-07, 0.04833333194255829, 0.5]

# hide data in view
Hide(threshold1, renderView1)

# show color bar/color legend
resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=resampleToImage1)
contour1.ContourBy = ['POINTS', 'p']
contour1.Isosurfaces = [475.30816650390625]
contour1.PointMergeMethod = 'Uniform Binning'

# set active source
SetActiveSource(resampleToImage1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=resampleToImage1Display.SliceFunction)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=resampleToImage1Display)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=resampleToImage1Display.SliceFunction)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=resampleToImage1Display)

# destroy contour1
Delete(contour1)
del contour1

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=resampleToImage1)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.ResultArrayName = 'U_z'
calculator1.Function = 'U_Z'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'UniformGridRepresentation')

# get color transfer function/color map for 'U_z'
u_zLUT = GetColorTransferFunction('U_z')

# get opacity transfer function/opacity map for 'U_z'
u_zPWF = GetOpacityTransferFunction('U_z')

# trace defaults for the display properties.
calculator1Display.Representation = 'Slice'
calculator1Display.ColorArrayName = ['POINTS', 'U_z']
calculator1Display.LookupTable = u_zLUT
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.OSPRayScaleArray = 'U_z'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'U'
calculator1Display.ScaleFactor = 0.09999989999999999
calculator1Display.SelectScaleArray = 'U_z'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'U_z'
calculator1Display.GaussianRadius = 0.004999994999999999
calculator1Display.SetScaleArray = ['POINTS', 'U_z']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'U_z']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityUnitDistance = 0.01723997378730899
calculator1Display.ScalarOpacityFunction = u_zPWF
calculator1Display.OpacityArrayName = ['POINTS', 'U_z']
calculator1Display.IsosurfaceValues = [0.06552600290160626]
calculator1Display.SliceFunction = 'Plane'
calculator1Display.Slice = 999
calculator1Display.SelectInputVectors = ['POINTS', 'U']
calculator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [-0.0024496132973581553, 0.0, 0.5, 0.0, 0.13350161910057068, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [-0.0024496132973581553, 0.0, 0.5, 0.0, 0.13350161910057068, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
calculator1Display.SliceFunction.Origin = [5.000000000143778e-07, 0.04833333194255829, 0.5]

# hide data in view
Hide(resampleToImage1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=calculator1)
calculator2.Function = ''

# set active source
SetActiveSource(calculator1)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=calculator1Display.SliceFunction)

# toggle 3D widget visibility (only when running from the GUI)
Show3DWidgets(proxy=calculator1Display)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=calculator1Display.SliceFunction)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=calculator1Display)

# destroy calculator2
Delete(calculator2)
del calculator2

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=calculator1)
contour1.ContourBy = ['POINTS', 'U_z']
contour1.Isosurfaces = [0.06552600290160626]
contour1.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour1
contour1.Isosurfaces = [0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31]

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = [None, '']
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'None'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = -2.0000000000000002e+298
contour1Display.SelectScaleArray = 'None'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'None'
contour1Display.GaussianRadius = -1e+297
contour1Display.SetScaleArray = [None, '']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = [None, '']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'
contour1Display.SelectInputVectors = ['FIELD', 'CasePath']
contour1Display.WriteLog = ''

# update the view to ensure updated data information
renderView1.Update()

# reset view to fit data
renderView1.ResetCamera(False)

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToLast()

animationScene1.GoToFirst()

animationScene1.Play()

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2880, 1314)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-1.9408542635388253, 0.04833333194255829, 0.5]
renderView1.CameraFocalPoint = [5.000000000143778e-07, 0.04833333194255829, 0.5]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = 0.5023301765817975

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).