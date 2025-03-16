import numpy as np
import os
import meshio
import matplotlib.pyplot as plt
import panel as pn
from mpl_toolkits.mplot3d import Axes3D

def load_gmsh(path, depr=True):
    if depr:
        (n_cells, plot) = _load_and_plot_gmsh(path)
        return [plot], [n_cells]


    plots = []
    cells = []
    for file in os.listdir(path):
        if file.lower().endswith('.msh'):
            full_path = os.path.join(path, file)
            
            if os.path.isfile(full_path):
                try:
                    (n_cells, plot) = _load_and_plot_gmsh(full_path)
                    plots.append(plot)
                    cells.append(n_cells)

                except Exception as e:
                    print(f"Failed to load {full_path}: {e}")
    order = list(np.argsort(np.array(cells, dtype=int)))
    oplots = [plots[o] for o in order]
    ocells = [cells[o] for o in order]
    return oplots, ocells

def _load_and_plot_gmsh(file_path):
    
    mesh = meshio.read(file_path)
    
    # Extract points and cells
    points = mesh.points
    lines = mesh.cells_dict.get("line", [])
    triangles = mesh.cells_dict.get("triangle", [])
    quads = mesh.cells_dict.get("quad", [])
    tetrahedrons = mesh.cells_dict.get("tetra", [])
    hexahedrons = mesh.cells_dict.get("hexahedron", [])

    # Determine if the mesh is 2D or 3D
    # is_3d = points.shape[1] == 3
    is_3d = len(tetrahedrons) > 0 or len(hexahedrons) > 0
    is_2d = len(quads) > 0 or len(triangles) > 0

    # Create a plot
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tetrahedrons
        for cell in tetrahedrons:
            tetra = points[cell]
            for i in range(4):
                for j in range(i + 1, 4):
                    ax.plot([tetra[i, 0], tetra[j, 0]], [tetra[i, 1], tetra[j, 1]], [tetra[i, 2], tetra[j, 2]], 'k-')
        
        # Plot hexahedrons
        for cell in hexahedrons:
            hexa = points[cell]
            for i in range(8):
                for j in range(i + 1, 8):
                    ax.plot([hexa[i, 0], hexa[j, 0]], [hexa[i, 1], hexa[j, 1]], [hexa[i, 2], hexa[j, 2]], 'k-')
        
        # Set the viewpoint to fit the entire mesh
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_zlim(points[:, 2].min(), points[:, 2].max())
        ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth for a better view
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        cells = len(hexahedrons) + len(tetrahedrons)
    elif is_2d:
        fig, ax = plt.subplots()
        
        # Plot triangles
        for cell in triangles:
            triangle = points[cell]
            polygon = plt.Polygon(triangle[:, :2], edgecolor='k', facecolor='none')
            ax.add_patch(polygon)
        
        # Plot quads
        for cell in quads:
            quad = points[cell]
            polygon = plt.Polygon(quad[:, :2], edgecolor='k', facecolor='none')
            ax.add_patch(polygon)


        # Adjust the viewpoint to fit the entire mesh
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cells = len(triangles) + len(quads)
    else:
        fig, ax = plt.subplots()
        
        # Plot triangles
        for cell in lines:
            line = points[cell]
            ax.plot(line[:, 0], line[:, 1], color='k')
            #polygon = plt.Polygon(line[:, :2], edgecolor='k', facecolor='none')
            #ax.add_patch(polygon)
        
        # Adjust the viewpoint to fit the entire mesh
        #ax.set_xlim(points[:, 0].min(), points[:, 0].max()) ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        #ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cells = len(lines)


    return (cells, pn.pane.Matplotlib(fig, width=300))
