import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

from parameters import *
from streamline import *
import visualization as visu
from solver import *


streamlines = []
# init_streamline(streamlines, 0.2, 0.95, N, M)
# init_streamline(streamlines, 0.4, 0.95, N, M)
# init_streamline(streamlines, 0.6, 0.95, N, M)
# init_streamline(streamlines, 0.8, 0.95, N, M)


plt.ion()
fig, ax = visu.setup()
images = []
# ax['a'].set_title(f'Who is first? \n Game starts in {5} seconds')
devices = jax.devices()
print(f'found jax devices: {devices}')

image = simulate(fig, ax, streamlines, images=images)
# imageio.mimsave('output.gif', images, fps=1)
