import plotly.graph_objects as go

# 1. Define Vertices (Nodes)
# A simple pyramid shape: 4 base corners + 1 peak
# Node 0: (0,0)  Node 1: (10,0)
# Node 3: (0,10) Node 2: (10,10)
# Node 4: (5,5)  <-- Peak at height 5
x = [0,  10, 10, 0,  5]
y = [0,  0,  10, 10, 5]
z = [0,  0,  0,  0,  5]

# 2. Define Connectivity (Triangles)
# We map the indices (0-4) to form triangles.
# 4 Triangles connecting the base corners to the peak (index 4)
i = [0, 1, 2, 3]  # First vertex of each triangle
j = [1, 2, 3, 0]  # Second vertex
k = [4, 4, 4, 4]  # Third vertex (all connect to peak)

# 3. Define a Field (Scalar Data)
# Let's just color it by height (Z)
# Pure Python list, no NumPy
intensity = [0.0, 0.0, 0.0, 0.0, 10.0] 

fig = go.Figure(data=[
    go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensity,
        colorscale='Viridis',
        name='Simple Pyramid',
        showscale=True
    )
])

fig.update_layout(
    title="Hand-Constructed Mesh (No SciPy, No NumPy)",
    scene=dict(aspectmode='data')
)
display(fig)
