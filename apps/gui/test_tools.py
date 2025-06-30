import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

# Create an empty Path object for drawing
path = hv.Path([])

# Freehand drawing tool setup
draw_stream = streams.FreehandDraw(source=path, num_objects=1)

def on_draw(data):
    print("Finished")
    if 'data' in data:
        new_segment = data['data']
        print("New Segment:", new_segment)

# Attach subscriber (callback) to draw_stream updates
draw_stream.add_subscriber(on_draw)

# Function to process or visualize based on current path state
def process_path(data):
    if 'element' in data:
        return hv.Path(data['element']).opts(line_color='blue', line_dash='dashed')
    else:
        return hv.Path([])  # Return an empty path if no element is present

# Create a DynamicMap using process_path function, updating based on draw_stream's element (the drawn path)
dynamic_map = hv.DynamicMap(process_path, streams=[draw_stream])

# Combine original path display with dynamic visualization from DynamicMap,
layout = pn.Column(
    (path.opts(line_color='red', tools=[], active_tools=['freehand_draw']) * dynamic_map)
)

if __name__.startswith("bokeh"):
    layout.servable()
