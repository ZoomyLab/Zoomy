import meshio
import matplotlib.pyplot as plt
import panel as pn

def load_and_plot_gmsh(file_path):
    # Load the Gmsh file
    mesh = meshio.read(file_path)
    
    # Extract points and cells
    points = mesh.points
    triangles = mesh.cells_dict.get("triangle", [])
    quads = mesh.cells_dict.get("quad", [])

    # Create a plot
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

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return pn.pane.Matplotlib(fig, width=300)