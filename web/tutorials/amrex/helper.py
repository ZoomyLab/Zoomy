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
        
