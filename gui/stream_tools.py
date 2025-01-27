import numpy as np


def coords_to_pixels(x_coords, y_coords, xlim, ylim, img_shape):
    # Calculate scale factors
    x_scale = img_shape[1] / (xlim[1] - xlim[0])
    y_scale = img_shape[0] / (ylim[1] - ylim[0])

    # Convert coordinates to pixel indices
    x_pixels = np.clip((np.array(x_coords) - xlim[0]) * x_scale, 0, img_shape[1]-1).astype(np.uint8)
    y_pixels = np.clip((np.array(y_coords) - ylim[0]) * y_scale, 0, img_shape[0]-1).astype(np.uint8)

    return x_pixels, y_pixels


def segment_to_pixel(segment,  width=3):
    line_x = []
    line_y = []
    for i in range(1, segment.shape[0]):
        x0 = segment[i-1, 0]
        y0 = segment[i-1, 1]
        x1 = segment[i, 0]
        y1 = segment[i, 1]
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        line_x += list(np.linspace(x0+0.5, x1+0.5, int(400*d+1), dtype=float))
        line_y += list(np.linspace(y0+0.5, y1+0.5, int(400*d+1), dtype=float))
    line_x = np.array(line_x).flatten()
    line_y = np.array(line_y).flatten()

    line_x_t = np.abs(1-line_y)
    line_y_t = line_x

    idx_x = np.array(line_x_t * 400, dtype=int)
    idx_y = np.array(line_y_t * 400, dtype=int)
    idx_x = np.clip(idx_x, a_min=0,  a_max=399)
    idx_y = np.clip(idx_y, a_min=0,  a_max=399)

    raster = np.zeros((400+2*width, 400+2*width), dtype=np.uint8)
    offset = int(width)

    for wx in range(1, width+1):
        for wy in range(1, width+1):
            ox =offset + idx_x - width + wx 
            oy = offset + idx_y - width + wy
            raster[ox, oy] += 1
    raster = raster[offset:-offset, offset:-offset]
    return np.array(raster, dtype=np.uint8)


def rasterize(raster, data):
    if 'xs' in data and 'ys' in data:
        for xs, ys in zip(data['xs'], data['ys']):
            segment = np.zeros((np.array(xs).shape[0], 2))
            segment[:, 0] = xs
            segment[:, 1] = ys
            raster += segment_to_pixel(segment)

        raster = np.array(np.where(raster > 0, 255, 0), dtype=np.uint8)
    return raster

