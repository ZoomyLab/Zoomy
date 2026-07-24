import plotly.graph_objects as go
import numpy as np

# Use standard lists to rule out NumPy-to-JSON issues
x = list(range(10))
y = [i**2 for i in x]

fig = go.Figure(data=go.Scatter(x=x, y=y))
fig.update_layout(title="Test Plot")
display(fig)
