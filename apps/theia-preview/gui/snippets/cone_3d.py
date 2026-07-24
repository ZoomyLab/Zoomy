import plotly.graph_objects as go
import numpy as np

fig = go.Figure(
    data=go.Cone(
        x=[1, 2],
        y=[1, 2],
        z=[1, 2],
        u=[1, 1],
        v=[1, 0],
        w=[0, 1],
        colorscale="Viridis",
        sizemode="scaled",
        sizeref=2,
    )
)

fig.update_layout(scene=dict(aspectmode="data"), title="3D Vector Field")
display(fig)
