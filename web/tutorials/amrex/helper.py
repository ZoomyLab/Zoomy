import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show

def create_artificial_raster(func, bounds, dx, path):
    """
    Create an artificial raster dataset from a lambda function.

    Parameters
    ----------
    func : callable
        Function of the form f(x, y) returning a scalar.
    bounds : tuple
        (x0, x1, y0, y1) domain extents.
    dx : float
        Grid spacing in both x and y directions.
    path : str
        Output GeoTIFF file path.
    """
    x0, x1, y0, y1 = bounds
    
    # Compute grid size
    nx = int(np.ceil((x1 - x0) / dx))
    ny = int(np.ceil((y1 - y0) / dx))

    # Grid coordinates (center of cells)
    xs = x0 + (np.arange(nx) + 0.5) * dx
    ys = y1 - (np.arange(ny) + 0.5) * dx  # y decreases downward in rasters
    
    X, Y = np.meshgrid(xs, ys)

    # Evaluate function
    data = func(X, Y).astype(np.float32)

    # Define affine transform
    transform = from_origin(x0, y1, dx, dx)

    # Write raster
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        
def show_raster(path, cmap="viridis"):

    with rasterio.open(path) as dem:
        elevation = dem.read(1, masked=True)  # handles nodata
        show(elevation, cmap="terrain")
        


def transform_tiff(file_path, tilt=False, scale=1., zoom=[None, None]):
    # Read the GeoTIFF file
    with rasterio.open(file_path) as src:
        elevation = src.read(1)
        Ny, Nx = elevation.shape
        dx = src.transform[0]  # Pixel size in x direction
        dy = -src.transform[4]  # Pixel size in y direction (negative because y decreases upwards)

    # Create meshgrid for coordinates
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y)

    X = X.T
    Y = Y.T
    Z = elevation

    # Flatten for fitting
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()

    # Stack design matrix for 2D polynomial fit: [1, x, y]
    A = np.stack([np.ones_like(x_flat), x_flat, y_flat], axis=1)

    # Least squares fit: solve A @ [a0, a1, a2] = z
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    a0, a1, a2 = coeffs  # plane: z = a0 + a1*x + a2*y

    # Compute slope magnitude and inclination angle (in radians and degrees)
    slope_x = -a1  # Slope in x direction (downward)
    alpha_rad = -np.arctan(slope_x)
    alpha_deg = np.degrees(alpha_rad)

    slope_y = a2  # Slope in y direction (upward)
    theta_rad = np.arctan(slope_y)
    theta_deg = np.degrees(theta_rad)

    print(f"Inclination angle: alpha: {alpha_deg:.2f} degrees ; theta: {theta_deg:.2f} degrees")

    # Creating adjusted elevation data based on offsets calculated from the fitted plane.
    if tilt:
        adjusted_elevation_data = (elevation - (a0 + a1 * X).T)*scale
    else:
        adjusted_elevation_data = (elevation)*scale
        
    
    if zoom[0] is not None:
        adjusted_elevation_data = adjusted_elevation_data[zoom[0][0]:zoom[0][1], :]
        Ny = zoom[0][1] - zoom[0][0]
    if zoom[1] is not None:
        adjusted_elevation_data = adjusted_elevation_data[:, zoom[1][0]:zoom[1][1]]
        Nx = zoom[1][1] - zoom[1][0]

    
    # Write adjusted data back to new GeoTIFF file
    new_file_path = 'adjusted_' + file_path.split('/')[-1]
    
    with rasterio.open(new_file_path,
                        'w',
                        driver='GTiff',
                        height=Ny,
                        width=Nx,
                        count=1,
                        dtype=elevation.dtype,
                        crs=src.crs,
                        transform=from_origin(src.transform[2], src.transform[5], dx, dy)) as dst:
        dst.write(adjusted_elevation_data.astype(elevation.dtype), 1)

    return new_file_path, alpha_deg