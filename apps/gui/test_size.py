import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

# Create a 400x400 image with random data
image_data = np.random.rand(400, 400)

# Function to create an hv.Image using the global variable (called once)
def create_image():
    return hv.Image(image_data).opts(
        width=400,  # Logical width of the data
        height=400,  # Logical height of the data
        tools=['reset'],  
    )

# Create a DynamicMap that references the static hv.Image object
dmap = hv.DynamicMap(create_image)

# Use Panel's layout capabilities to scale up display size
layout = pn.Column(dmap.opts(width=800, height=800), sizing_mode='stretch_both')

if __name__.startswith("bokeh"):
    # This block ensures compatibility when running via 'panel serve'
    layout.servable()
