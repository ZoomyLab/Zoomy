import numpy as np

import apps.game.stream.parameters as param

def segment_to_pixel(segment,  width=3):
    line_x = []
    line_y = []
    for i in range(1, segment.shape[0]):
        x0 = segment[i-1, 0]
        y0 = segment[i-1, 1]
        x1 = segment[i, 0]
        y1 = segment[i, 1]
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        N = max(param.Nx+2*param.n_ghosts + 2*width, param.Ny+2*param.n_ghosts +2* width)
        line_x += list(np.linspace(x0, x1, int(N*d+1), dtype=float))
        line_y += list(np.linspace(y0, y1, int(N*d+1), dtype=float))
    line_x = np.array(line_x).flatten()
    line_y = np.array(line_y).flatten()

    line_x_t = line_x
    line_y_t = line_y

    idx_x = np.array(line_x_t * (param.Ny + param.n_ghosts + width), dtype=int)
    idx_y = np.array(line_y_t * (param.Nx + param.n_ghosts + width), dtype=int)
    idx_x = np.clip(idx_x, a_min=0,  a_max=param.Ny-1 + 2*param.n_ghosts + 2*width)
    idx_y = np.clip(idx_y, a_min=0,  a_max=param.Nx-1 + 2*param.n_ghosts + 2*width)

    raster = np.zeros((param.Nx+2*param.n_ghosts+2*width, param.Ny+2*param.n_ghosts+2*width), dtype=np.uint8)
    offset = int(width)

    for wx in range(1, width+1):
        for wy in range(1, width+1):
            ox =offset + idx_x - width + wx 
            oy = offset + idx_y - width + wy
            raster[oy, ox] += 1
    raster = raster[offset:-offset, offset:-offset]
    return np.array(raster, dtype=np.uint8)


def rasterize_segments(raster, data):
    if 'xs' in data and 'ys' in data:
        for xs, ys in zip(data['xs'], data['ys']):
            segment = np.zeros((np.array(xs).shape[0], 2))
            segment[:, 0] = xs
            segment[:, 1] = ys
            raster += segment_to_pixel(segment)

        raster = np.array(np.where(raster > 0, 255, 0), dtype=np.uint8)
    return raster

def rasterize_boxes(raster, data):
    num_rects = len(data['x'])
    for i in range(num_rects):
        x_center = data['x'][i]
        y_center = data['y'][i]
        width = data['width'][i]
        height = data['height'][i]
        
        # Calculate corner coordinates
        left = x_center - width / 2
        right = x_center + width / 2
        bottom = y_center - height / 2
        top = y_center + height / 2
    
        x0 =  int(left * param.Ny)
        x1 =  int(right * param.Ny)
        y0 =  int(bottom * param.Nx)
        y1 =  int(top * param.Nx)

        for v in [x0, x1]:
            if v == param.Ny:
                v = param.Ny-1

        for v in [y0, y1]:
            if v == param.Nx:
                v = param.Nx-1

        #raster[x0:x1, y0:y1] = 1.
        raster[y0:y1, x0:x1] = 1.

    raster = np.array(np.where(raster > 0, 255, 0), dtype=np.uint8)

    return raster

