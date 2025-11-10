import '@kitware/vtk.js/favicon';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';

// Create renderer
const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
  rootContainer: document.querySelector('#container'),
  containerStyle: { height: '100%', width: '100%' },
});
const renderer = fullScreenRenderer.getRenderer();
const renderWindow = fullScreenRenderer.getRenderWindow();

// Load VTP file
fetch('/assets/data.vtkhdf')
  .then((res) => res.arrayBuffer())
  .then((buffer) => {
    const reader = vtkXMLPolyDataReader.newInstance();
    reader.parseAsArrayBuffer(buffer);

    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(reader.getOutputPort());

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);

    renderer.addActor(actor);
    renderer.resetCamera();
    renderWindow.render();
  });

