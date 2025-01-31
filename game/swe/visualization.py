import numpy as nnp
import matplotlib.pyplot as plt

from parameters import *
from streamline import *


def setup():
    fig, ax = plt.subplot_mosaic([['a', 'b'], ['a', 'c']], layout='constrained', gridspec_kw={'width_ratios': [1, 2]})
    ax['b'].remove()
    ax['c'].remove()
    ax['b'] = fig.add_subplot(1, 2, 2, projection='3d')
    return fig, ax


def update(streamlines, fig, ax, hdisplay, Q, time, images=None):
    ax['a'].cla()
    ax['b'].cla()

    z = Q[3]

    plt.title(f'Time {time:10.1f}')
    image = ax['a'].imshow(z, origin='lower', cmap='Greys')
    ax['a'].contour(xv, yv, z, cmap='summer', levels=20)
    # ax['a'].imshow(Q[0] > 0, origin='lower', cmap='Blues', alpha=0.5)
    ax['a'].imshow(Q[0], origin='lower', cmap='Blues', alpha=0.5)
    # plot_streamlines(streamlines, ax['a'])

    # colors = nnp.zeros((Q[0].shape[0], Q[0].shape[1], 4))
    # colors[Q[0] > wet_tol] = [0, 0, 1, 0.5]  # Blue where Q[0] > 0
    # colors[Q[0] <= wet_tol] = [0, 0, 1, 0]  # Fully transparent where Q[0] <= 0
    # colors_b = nnp.zeros((Q[0].shape[0], Q[0].shape[1], 4))
    # colors_b [:,:,3] = 0.8
    # ax['b'].plot_surface(xv, yv,Q[3], facecolors=colors_b, shade=False)
    # ax['b'].plot_surface(xv, yv, Q[0] + Q[3] , facecolors=colors, edgecolor='none', shade=False)
    # ax[3].view_init(elev=45, azim=-90)
    

    fig.canvas.draw()
    if images is not None:
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image_array)
    fig.canvas.flush_events()
    if hdisplay is not None:
        hdisplay.update(fig)
    return image
