import pyswashes
import numpy as np
from matplotlib import pyplot as plt
#s = pyswashes.TwoDimensional(2, 1, 1, 50, 50)
s = pyswashes.OneDimensional(3, 1, 1, 50)


print(s.dom_params)
#attrs = ['x','depth','u','gd_elev','q','head','Froude','crit_head']
#attrs = ['depth','gd_elev','head']

#atts = ['x','y','depth','u','v','head','gd_elev','U','Froude','qx','qy','q']
#for a in attrs:
#    print(a, s.np_array(a))
#    print('====================')


x = np.linspace(0, s.dom_params['length'], int(s.dom_params['ncellx']))
b = s.np_array('gd_elev')
h = s.np_array('depth')
u = s.np_array('u')
#v = s.np_array('v')

# Plot
fig, ax = plt.subplots()
ax.plot(x, h+b, label='Water height h(x)+b(x)')
ax.plot(x, u, label='u(x)')
#ax.plot(x, v, label='v(x)')
ax.plot(x, b, label='bottom b(x)')
plt.title(f'Short channel')
ax.set_xlabel('x')
ax.grid()
plt.legend()
plt.tight_layout()
plt.show()

