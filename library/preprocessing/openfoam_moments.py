import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def load_file(filename):
    filename = 'channelflow_coarse_103.vtm'
    reader = pv.get_reader(filename)
    # get(0) gets us the internal (not boundary) data. The boundary data is non-existant anyways in our case
    vtkfile = reader.read().get(0)
    return vtkfile

def get_fields(vtkfile, fieldnames):
    fields = []
    for fieldname in fieldnames:
        field = vtkfile.point_data[fieldname]
        fields.append(np.array(field))
    return fields
    # return np.array(fields)

def get_coordinates(vtkfile):
    return np.array(vtkfile.points)

def get_time(vtkfile):
    return vtkfile.field_data['TimeValue'][0]

def sort_data(coordinates, fields):
    # Sort by z, y, x
    order = np.lexsort(
        (coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    )  
    return order

def apply_order(coordinates, fields, order):
    coordinates_copy = np.array(coordinates)
    fields_copy = deepcopy(fields)
    for d in range(coordinates.shape[1]):
        coordinates[:,d] = coordinates_copy[order, d]
    for field, field_copy in zip(fields, fields_copy):
        field[:] = field_copy[order]
    return coordinates, fields

def get_number_of_layers_and_elements_in_plane(coordinates):
    layers = len(np.unique(coordinates[:, 2]))
    n_elements_plane = int(coordinates[:,0].size / layers)
    return layers, n_elements_plane

def get_layer(data, layer, n_elements_plane):
    return data[layer*n_elements_plane:(layer+1)*n_elements_plane]

def compute_height(alpha, n_layers, n_elements_per_layer, total_height=1.0, threshold=0.5):
    height = np.zeros(n_elements_per_layer)
    active = np.ones(n_elements_per_layer, dtype=bool)
    dh = total_height / n_layers
    for i in range(n_layers):
        alpha_layer = get_layer(alpha, i, n_elements_per_layer)
        height += np.where(np.logical_and((alpha_layer >= threshold), active), dh, 0)
    return height

def extract_faces(mesh):
    faces = []
    i, offset = 0, 0
    cc = mesh.cells # fetch up front
    while i < mesh.n_cells:
        nn = cc[offset]
        faces.append(cc[offset+1:offset+1+nn])
        offset += nn + 1
        i += 1
    return np.array(faces)

def sort_faces(faces, order):
    for face in faces:
        for i, point in enumerate(face):
            face[i] = order[point]
    faces = faces[order]
    return faces

def add_noslip_layer(U, n_elements_per_layer):
    Unew = np.zeros((U.shape[0] + n_elements_per_layer, U.shape[1]))
    for d in range(U.shape[1]):
        Unew[n_elements_per_layer:, d] = U[:,d]
    return Unew

def extract_velocity_column_at_coordinate(U, n_elements_per_layer, n_layer, coordinate):
    u = np.array([get_layer(U[:, 0],i, n_elements_per_layer)[coordinate] for i in range(n_layer+1)])
    v = np.array([get_layer(U[:, 1],i, n_elements_per_layer)[coordinate] for i in range(n_layer+1)])
    w = np.array([get_layer(U[:, 2],i, n_elements_per_layer)[coordinate] for i in range(n_layer+1)])
    return u, v, w

def shift_integration_interval(xi):
    return (xi + 1) / 2


def plot_basis(basis_generator):
    fig, ax = plt.subplots()
    basis = [basis_generator(n) for n in range(0, 8)]
    X = np.linspace(0,1,100)
    for i in range(8):
        ax.plot(basis[i](X), X)
    return fig, ax

def moment_projection(field, n_layers, basis, integration_order = None):
    if integration_order is None:
        integration_order = max(len(basis), n_layers)
    xi, wi = np.polynomial.legendre.leggauss(integration_order)
    xi = shift_integration_interval(xi)
    dz = 1/(n_layers)
    xp = np.arange(dz/2, 1-dz/2, dz)
    xp = np.insert(xp, 0,0)
    fp = field
    field_xi = np.interp(xi, xp, fp)
    basis_xi = [basis[i](xi) for i in range(len(basis))]
    projections = np.zeros(len(basis))
    for i in range(len(basis)):
        projections[i] = np.sum(field_xi * basis_xi[i] * wi)
    return projections


def convert_openfoam_to_moments_single(filename, n_levels):
    vtkfile = load_file(filename)
    coordinates = get_coordinates(vtkfile)  
    n_layer, n_elements_per_layer = get_number_of_layers_and_elements_in_plane(coordinates)
    fields = get_fields(vtkfile, ['alpha.water', 'U'])
    sort_order = sort_data(coordinates, fields)    
    coordinates, fields = apply_order(coordinates, fields, sort_order)
    fields[1] = add_noslip_layer(fields[1], n_elements_per_layer)
    time = get_time(vtkfile)
    Q, basis = compute_shallow_moment_projection(fields, coordinates, n_levels)
    return coordinates, Q, time, basis

def convert_openforam_to_moments(filepath, n_levels):
    hdffile = h5py.File(os.path.join(filepath/'openfoam_moments.hdf5', 'w'))
    for i_file, file in enumerate(os.listdir(filepath)):
        if file.endswith(".vtm"):
            coordinates, Q, time, basis = convert_openfoam_to_moments_single(file, n_levels)
            if i_file == 0:
                return
                # TODO save mesh
            #TODO same time
            #TODO save Q

def sort_openfoam_hdf5_file_by_mesh_coordinates(filepath_mesh, filepath_openfoam):
    #TODO sort 
    return 0

    
def plot_contour(coordinates, field):
    fig, ax = plt.subplots()
    n_layer, n_elements_per_layer = get_number_of_layers_and_elements_in_plane(coordinates)
    X = get_layer(coordinates[:,0], 0, n_elements_per_layer)
    Y = get_layer(coordinates[:,1], 0, n_elements_per_layer)
    # Z = get_layer(fields[1][:, 0], 1, n_elements_per_layer)
    # Z = get_layer(fields[0], 12, n_elements_per_layer)
    # Z = compute_height(fields[0], n_layer, n_elements_per_layer)
    Z = field
    colorbar = ax.tricontourf(X, Y, Z)
    ax.set_aspect('equal')

    circle = plt.Circle((0.5, 1), radius=0.2, fc="silver", zorder=10, edgecolor="k")
    plt.gca().add_patch(circle)
    fig.colorbar(colorbar)
    return fig, ax


def basis_legendre(i):
    def f(x):
        basis = np.polynomial.legendre.Legendre.basis(i, domain=[0, 1], window=[-1, 1])
        return basis(x) * basis(0)
    return f


def compute_shallow_moment_projection(fields, coordinates, n_levels, basis_generator=basis_legendre):
    n_layers, n_elements_per_layer = get_number_of_layers_and_elements_in_plane(coordinates)
    height = compute_height(fields[0], n_layers, n_elements_per_layer)
    basis = [basis_generator(n) for n in range(0, n_levels+1)]
    alphas = np.zeros((n_elements_per_layer, n_levels+1))
    betas = np.zeros((n_elements_per_layer, n_levels+1))
    for i in range(n_elements_per_layer):
        u, v, w = extract_velocity_column_at_coordinate(fields[1], n_elements_per_layer, n_layers, i)
        alphas[i] = moment_projection(u, n_layers+1, basis )
        betas[i] = moment_projection(v, n_layers+1, basis )
    return np.concatenate((height.reshape((n_elements_per_layer, 1)), alphas, betas), axis=1).T, basis


def plot_data_vs_moments(fields, coordinates, Q, basis, coordinate_index):
    n_layers, n_elements_per_layer = get_number_of_layers_and_elements_in_plane(coordinates)
    n_levels = int((Q.shape[0]-1)/2)-1
    X = np.linspace(0,1,100)
    U = np.array([basis[i](X) * Q[1+i, coordinate_index] for i in range(len(basis))]).sum(axis=0)
    V = np.array([basis[i](X) * Q[1+n_levels+1+i, coordinate_index] for i in range(len(basis))]).sum(axis=0)
    dz = 1/(n_layers)
    x = np.linspace(dz/2, 1-dz/2, n_layers)
    x = np.insert(x, 0, 0)
    u, v, w = extract_velocity_column_at_coordinate(fields[1], n_elements_per_layer, n_layers, coordinate_index)
    fig, ax = plt.subplots(2)
    ax[0].plot(U, X)
    ax[0].plot(u, x)
    ax[1].plot(V, X)
    ax[1].plot(v, x)
    return fig, ax

def load_openfoam_file(filepath):
    vtkfile = load_file(filepath)
    coordinates = get_coordinates(vtkfile)  
    n_layer, n_elements_per_layer = get_number_of_layers_and_elements_in_plane(coordinates)
    fields = get_fields(vtkfile, ['alpha.water', 'U'])
    sort_order = sort_data(coordinates, fields)    
    coordinates, fields = apply_order(coordinates, fields, sort_order)
    fields[1] = add_noslip_layer(fields[1], n_elements_per_layer)
    time = get_time(vtkfile)
    return coordinates, fields, time


def test_load():
    coordinates, fields, time = load_openfoam_file('channelflow_coarse_131.vtm')

def test_moment_projection():
    coordinates, fields, time = load_openfoam_file('channelflow_coarse_131.vtm')
    Q, basis = compute_shallow_moment_projection(fields, coordinates, 3)

def test_convert_openfoam_single():
    X, Q, t, basis = convert_openfoam_to_moments_single('channelflow_coarse_131.vtm', 3)

def test_plots():
    coordinates, fields, time = load_openfoam_file('channelflow_coarse_131.vtm')
    X, Q, t, basis = convert_openfoam_to_moments_single('channelflow_coarse_131.vtm', 3)
    fig, ax = plot_data_vs_moments(fields, coordinates, Q, basis, 0)
    plt.show()
    fig, ax = plot_basis(basis_legendre)
    plt.show()
    fig, ax = plot_contour(coordinates, Q[0])
    plt.show()


if __name__ == '__main__':

    test_load()
    test_moment_projection()
    test_convert_openfoam_single()
    test_plots()

