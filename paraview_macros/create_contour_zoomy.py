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
cellDatatoPointData1.CellDataArraytoprocess = ['0', '1', '2', '3', '4', '5', 'aux_0']

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=cellDatatoPointData1)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# set active source
SetActiveSource(programmableFilter1)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
programmableFilter1Display = Show(programmableFilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
programmableFilter1Display.Representation = 'Surface'
programmableFilter1Display.ColorArrayName = [None, '']
programmableFilter1Display.SelectTCoordArray = 'None'
programmableFilter1Display.SelectNormalArray = 'None'
programmableFilter1Display.SelectTangentArray = 'None'
programmableFilter1Display.OSPRayScaleArray = '0'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'None'
programmableFilter1Display.ScaleFactor = 0.05
programmableFilter1Display.SelectScaleArray = '0'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = '0'
programmableFilter1Display.GaussianRadius = 0.0025
programmableFilter1Display.SetScaleArray = ['POINTS', '0']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = ['POINTS', '0']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 0.023521471297559562
programmableFilter1Display.OpacityArrayName = ['POINTS', '0']
programmableFilter1Display.SelectInputVectors = ['POINTS', '0']
programmableFilter1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
programmableFilter1Display.ScaleTransferFunction.Points = [6.609198023337276e-22, 0.0, 0.5, 0.0, 8.246431783325655e-22, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
programmableFilter1Display.OpacityTransferFunction.Points = [6.609198023337276e-22, 0.0, 0.5, 0.0, 8.246431783325655e-22, 1.0, 0.5, 0.0]

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=programmableFilter1)
calculator2.Function = ''

# Properties modified on calculator2
calculator2.ResultArrayName = 'U_z'
calculator2.Function = '"2"'

# show data in view
calculator2Display = Show(calculator2, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'U_z'
u_zLUT = GetColorTransferFunction('U_z')

# get opacity transfer function/opacity map for 'U_z'
u_zPWF = GetOpacityTransferFunction('U_z')

# trace defaults for the display properties.
calculator2Display.Representation = 'Surface'
calculator2Display.ColorArrayName = ['POINTS', 'U_z']
calculator2Display.LookupTable = u_zLUT
calculator2Display.SelectTCoordArray = 'None'
calculator2Display.SelectNormalArray = 'None'
calculator2Display.SelectTangentArray = 'None'
calculator2Display.OSPRayScaleArray = 'U_z'
calculator2Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator2Display.SelectOrientationVectors = 'None'
calculator2Display.ScaleFactor = 0.05
calculator2Display.SelectScaleArray = 'U_z'
calculator2Display.GlyphType = 'Arrow'
calculator2Display.GlyphTableIndexArray = 'U_z'
calculator2Display.GaussianRadius = 0.0025
calculator2Display.SetScaleArray = ['POINTS', 'U_z']
calculator2Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator2Display.OpacityArray = ['POINTS', 'U_z']
calculator2Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator2Display.DataAxesGrid = 'GridAxesRepresentation'
calculator2Display.PolarAxes = 'PolarAxesRepresentation'
calculator2Display.ScalarOpacityFunction = u_zPWF
calculator2Display.ScalarOpacityUnitDistance = 0.023521471297559562
calculator2Display.OpacityArrayName = ['POINTS', 'U_z']
calculator2Display.SelectInputVectors = ['POINTS', '0']
calculator2Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator2Display.ScaleTransferFunction.Points = [0.010484504953816446, 0.0, 0.5, 0.0, 0.24195427330790936, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator2Display.OpacityTransferFunction.Points = [0.010484504953816446, 0.0, 0.5, 0.0, 0.24195427330790936, 1.0, 0.5, 0.0]

# hide data in view
Hide(programmableFilter1, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# find source
resampleToImage1 = FindSource('ResampleToImage1')

# find source
fl1foam = FindSource('fl1.foam')

# find source
cellDatatoPointData1_1 = FindSource('CellDatatoPointData1')

# find source
resampleToImage2 = FindSource('ResampleToImage2')

# find source
calculator1 = FindSource('Calculator1')

# find source
threshold1 = FindSource('Threshold1')

# find source
contour2 = FindSource('Contour2')

# find source
threshold2 = FindSource('Threshold2')

# find source
calculator3 = FindSource('Calculator3')

# find source
contour3 = FindSource('Contour3')

# find source
mono = FindSource('mono')

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=calculator2)
contour1.ContourBy = ['POINTS', 'U_z']
contour1.Isosurfaces = [0.1262193891308629]
contour1.PointMergeMethod = 'Uniform Binning'

# Properties modified on contour1
contour1.Isosurfaces = [0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31]

# show data in view
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', 'U_z']
contour1Display.LookupTable = u_zLUT
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'None'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleArray = 'U_z'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.0173282596717526
contour1Display.SelectScaleArray = 'U_z'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'U_z'
contour1Display.GaussianRadius = 0.00086641298358763
contour1Display.SetScaleArray = ['POINTS', 'U_z']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'U_z']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'
contour1Display.SelectInputVectors = ['POINTS', '0']
contour1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.19, 0.0, 0.5, 0.0, 0.23, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [0.19, 0.0, 0.5, 0.0, 0.23, 1.0, 0.5, 0.0]

# hide data in view
Hide(calculator2, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(contour1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(u_zLUT, renderView1)

# Properties modified on contour1Display
contour1Display.LineWidth = 5.0

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2176, 1316)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-1.9385944339767136, -0.12168665988109592, 0.5363822490047493]
renderView1.CameraFocalPoint = [2.499999993688107e-07, -0.12168665988109592, 0.5363822490047493]
renderView1.CameraViewUp = [0.0, 1.0, -6.661338147750939e-16]
renderView1.CameraParallelScale = 0.2765685090736492

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).