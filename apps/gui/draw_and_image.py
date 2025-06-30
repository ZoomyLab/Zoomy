import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn
from bokeh.models import FreehandDrawTool

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

# Create an empty Path element (used for freehand drawing)
path = hv.Path([]).opts(line_width=2)

# Global variable to store renderer reference for easy access
renderer_ref = {}

# Add FreehandDrawTool to plot via hooks and store renderer reference
def add_freehand_tool(plot, element):
    tool = FreehandDrawTool(renderers=[plot.handles['glyph_renderer']])
    plot.state.add_tools(tool)
    plot.state.toolbar.active_drag = tool
    
    # Store renderer reference for later use
    renderer_ref['renderer'] = plot.handles['glyph_renderer']

dmap = hv.DynamicMap(lambda: path).opts(
    hooks=[add_freehand_tool],
    width=400,
    height=400,
)

def coords_to_pixels(x_coords, y_coords, xlim, ylim, img_shape):
    # Calculate scale factors
    x_scale = img_shape[1] / (xlim[1] - xlim[0])
    y_scale = img_shape[0] / (ylim[1] - ylim[0])

    # Convert coordinates to pixel indices
    x_pixels = np.clip((np.array(x_coords) - xlim[0]) * x_scale, 0, img_shape[1]-1).astype(int)
    y_pixels = np.clip((np.array(y_coords) - ylim[0]) * y_scale, 0, img_shape[0]-1).astype(int)

    return x_pixels, y_pixels


def segment_to_pixel(segment, width=3):
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

    raster = np.zeros((400, 400))
    raster[idx_x, idx_y] += 1.
    for w in range(1, width+1):
        for ww in range(1, width+1):
            raster[idx_x+w, idx_y] += 1.
            raster[idx_x, idx_y+w] += 1.
            raster[idx_x+w, idx_y+ww] += 1.
            raster[idx_x+ww, idx_y+w] += 1.
            raster[idx_x-w, idx_y] += 1.
            raster[idx_x, idx_y-w] += 1.
            raster[idx_x-w, idx_y-w] += 1.
            raster[idx_x-ww, idx_y-w] += 1.
            raster[idx_x-w, idx_y-ww] += 1.
    return raster

# Function to print current path data directly from the renderer's ColumnDataSource
def rasterize():
    raster = np.zeros((400, 400))
    if 'renderer' in renderer_ref:
        cds = renderer_ref['renderer'].data_source  # Get ColumnDataSource
        
        
        if 'xs' in cds.data and 'ys' in cds.data:
            for xs, ys in zip(cds.data['xs'], cds.data['ys']):
                #print("New Segment:")
                #print(np.array(xs).shape)
                segment = np.zeros((np.array(xs).shape[0], 2))
                segment[:, 0] = xs
                segment[:, 1] = ys
                raster += segment_to_pixel(segment)

        raster = np.where(raster > 0, 255., 0.)

                #for x, y in zip(xs, ys):
                #    print(f"({x}, {y})")
                #x, y = coords_to_pixels(xs, ys, (-0.5, 0.5), (-0.5, 0.5), (400, 400))
    #print(np.sum(raster.nonzero()))
    return raster

# Layout with clear button functionality
def clear_canvas(event):
    if 'renderer' in renderer_ref:
        renderer_ref['renderer'].data_source.data.update(xs=[], ys=[])

reset_button = pn.widgets.Button(name='Clear Canvas')
reset_button.on_click(clear_canvas)

print_button = pn.widgets.Button(name='Print Data')
print_button.on_click(lambda event: rasterize())


# Function to generate random image data
def random_image():
    #return np.random.rand(400, 400)
    return rasterize()

# Function to update the image plot using a stream parameter
def update_image(next):
    return hv.Image(random_image()).opts(alpha=0.3, tools=['reset'])

# Create a stream that triggers updates every second with an initial value for 'next'
stream = streams.Stream.define('Next', next=0)()

# Create a DynamicMap with the update function and stream
dmap_image = hv.DynamicMap(update_image, streams=[stream]).opts(tools=['reset'])


# Use Panel's state management for periodic callbacks (Jupyter-friendly)
def periodic_callback():
    current_next = stream.contents['next']
    stream.event(next=current_next + 1)

pn.state.add_periodic_callback(periodic_callback, period=1000)  # Period in milliseconds

#dmap_final = (dmap_image * dmap).opts(tools=['reset'])

layout = pn.Column(reset_button, print_button, dmap_image, sizing_mode='stretch_both')

# If you're using this in a standalone script, make sure it's structured so it can be served.
if __name__.startswith("bokeh"):
    # This block ensures compatibility when running via 'panel serve'
    layout.servable()
