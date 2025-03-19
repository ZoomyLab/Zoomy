import jax.numpy as np
import matplotlib.pyplot as plt

from parameters import *

def init_streamline(streamlines, px, py, N, M):
    streamline_x = [int(px * N)]
    streamline_y = [int(py * M)]
    streamlines.append([streamline_x, streamline_y])


def integrate_streamlines(streamlines, u, v, dt, X, Y):
    finished = False
    for streamline  in streamlines:
        streamline_x = streamline[0]
        streamline_y = streamline[1]
        x = streamline_x[-1]
        y = streamline_y[-1]
        # find index of x and y in the grid
        # ix = np.argmin(np.abs(X - x))
        # iy = np.argmin(np.abs(Y - y))
        ix = int(np.rint(x))
        iy = int(np.rint(y))
        dx = u[iy, ix] * dt
        dy = v[iy, ix] * dt
        streamline_x.append(streamline_x[-1] + dx)
        streamline_y.append(streamline_y[-1] + dy)
        finished = finished or streamline_y[-1] <= 0
    return streamlines, finished

def plot_streamlines(streamlines, ax):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, streamline in enumerate(streamlines):
        streamline_x = streamline[0]
        streamline_y = streamline[1]
        streamline_fig = ax.plot(streamline_x, streamline_y, colors[i])

    for i, streamline in enumerate(streamlines):
        streamline_x = streamline[0]
        streamline_y = streamline[1]
        streamline_fig = ax.plot(streamline_x[0], streamline_y[0], colors[i], marker='o', markersize=5)
    if len(streamlines) > 0:
        return streamline_fig
    return None

def plot_streamlines_3d(Q, streamlines, ax):
    colors = ['r', 'g', 'y', 'c']
    for i, streamline in enumerate(streamlines):
        streamline_x = streamline[0]
        streamline_y = streamline[1]
        ix = np.array(np.rint(np.array(streamline_x)), dtype=int)
        iy = np.array(np.rint(np.array(streamline_y)), dtype=int)
        streamline_fig = ax.plot(streamline_x, streamline_y, (Q[0, iy, ix] + Q[3, iy, ix])*1.1,  colors[i], linewidth=5, alpha=1)
    if len(streamlines) > 0:
        return streamline_fig
    return None
