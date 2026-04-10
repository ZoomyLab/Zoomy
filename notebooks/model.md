**SME** (mass, x_momentum)

**Assumptions:** hydrostatic, material=Newtonian

**mass:**
$$
\frac{d}{d t} H + \frac{\partial}{\partial x} \int\limits_{b}^{H + b} u\, dz = 0
$$

**x_momentum:**
$$
2 \nu \left(\frac{d}{d x} H + \frac{d}{d x} b\right) \left. \frac{d}{d x} u \right|_{\substack{ z=H + b }} - 2 \nu \frac{d}{d x} b \left. \frac{d}{d x} u \right|_{\substack{ z=b }} - u^{2} \left(\frac{d}{d x} H + \frac{d}{d x} b\right) + u^{2} \frac{d}{d x} b - u \left(u \frac{d}{d x} b + \frac{d}{d t} b\right) - u \left(\frac{d}{d t} H + \frac{d}{d t} b\right) + u \left(u \left(\frac{d}{d x} H + \frac{d}{d x} b\right) + \frac{d}{d t} H + \frac{d}{d t} b\right) + u \frac{d}{d t} b + \frac{\partial}{\partial t} \int\limits_{b}^{H + b} u\, dz + \frac{\partial}{\partial x} \int\limits_{b}^{H + b} u^{2}\, dz + \frac{\partial}{\partial x} \int\limits_{b}^{H + b} \left(- 2 \nu \frac{d}{d x} u\right)\, dz + \frac{\partial}{\partial x} \int\limits_{b}^{H + b} \frac{H g \rho + b g \rho - g \rho z + p_{atm}}{\rho}\, dz + \frac{\left(H g \rho + p_{atm}\right) \frac{d}{d x} b}{\rho} + \frac{\nu \rho \frac{d}{d b} u + \nu \rho \left. \frac{d}{d x} w \right|_{\substack{ z=b }}}{\rho} - \frac{\nu \rho \left. \frac{d}{d z} u \right|_{\substack{ z=H + b }} + \nu \rho \left. \frac{d}{d x} w \right|_{\substack{ z=H + b }}}{\rho} - \frac{\left(\frac{d}{d x} H + \frac{d}{d x} b\right) \left(H g \rho + b g \rho - g \rho \left(H + b\right) + p_{atm}\right)}{\rho} = 0
$$
