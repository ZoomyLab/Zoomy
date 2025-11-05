import numpy as np
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Matrix, sqrt
from sympy.abc import x

from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify


from zoomy_core.model.basemodel import (
    register_sympy_attribute,
    eigenvalue_dict_to_matrix,
)
from zoomy_core.model.basemodel import Model
import zoomy_core.model.initial_conditions as IC

class VAMHyperbolic(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=6,
        aux_variables=['hw2', 'hp0', 'hp1', 'dbdx', 'dhdx', 'dhp0dx', 'dhp1dx'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )
        
    def flux(self):
        fx = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        
        fx[0] = hu0
        fx[1] = hu0 * u0 + 1/3 * hu1 * u1 * u1
        fx[2] = hu0 * w0 + 1/3 * hu1 * w1
        fx[3] = 2*hu0 * u1
        fx[4] = hu0 * w1 + u1 * (hw0 + 2/5*hw2)
        
        return [fx]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])

        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 / h

        nc[1, 0] = param.g * h
        nc[3, 2] = u0
        nc[4, 2] = - 1/5 * w2 + w0
        nc[1, 5] = param.g * h
        return [-nc]
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h

        ev[0] = u0
        ev[1] = u0 + 1/sqrt(3) * u1
        ev[2] = u0 - 1/sqrt(3) * u1
        ev[3] = u0 + sqrt(param.g * h + u1**2)
        ev[4] = u0 - sqrt(param.g * h + u1**2)
        ev[5] = 0
        
        return ev

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        b = self.variables[5]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 /h  


        p0 = self.aux_variables.hp0/h
        p1 = self.aux_variables.hp1/h
        dbdx = self.aux_variables.dbdx
        dhdx = self.aux_variables.dhdx
        dhp0dx = self.aux_variables.dhp0dx
        dhp1dx = self.aux_variables.dhp1dx

        R[0] = 0.
        R[1] = dhp0dx + 2 * p1 * dbdx 
        R[2] = -2*p1
        R[3] = dhp1dx - (3*p0 - p1)*dhdx  -6*(p0-p1)*dbdx
        R[4] = 6*(p0-p1)
        R[5] = 0.
        return R
        

class VAMPoisson(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=['hp0', 'hp1'],
        aux_variables=['h', 'hu0', 'hu1', 'hw0', 'hw1' ,'b', 'hw2', 'dbdx', 'ddbdxx', 'dhdx', 'ddhdxx', 'du0dx', 'du1dx', 'dhp0dx', 'ddhp0dxx', 'dhp1dx', 'ddhp1dxx', 'dt', 'd4hp0dx4', 'd4hp1dx4'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])

        h = self.aux_variables.h
        #p0 = self.variables.p0/h
        #p1 = self.variables.p1/h
        dt = self.aux_variables.dt

        dbdx   = self.aux_variables.dbdx
        ddbdxx = self.aux_variables.ddbdxx
        dhdx   = self.aux_variables.dhdx
        ddhdxx = self.aux_variables.ddhdxx
        #dp0dx   = self.aux_variables.dp0dx
        #ddp0dxx = self.aux_variables.ddp0dxx
        #dp1dx   = self.aux_variables.dp1dx
        #ddp1dxx   = self.aux_variables.ddp1dxx
        du0dx = self.aux_variables.du0dx
        du1dx = self.aux_variables.du1dx

        u0 = self.aux_variables.hu0/h
        u1 = self.aux_variables.hu1/h
        w0 = self.aux_variables.hw0/h
        w1 = self.aux_variables.hw1/h
        
        hp0 = self.variables.hp0
        hp1 = self.variables.hp1
        dhp0dx   = self.aux_variables.dhp0dx
        ddhp0dxx = self.aux_variables.ddhp0dxx
        dhp1dx   = self.aux_variables.dhp1dx
        ddhp1dxx   = self.aux_variables.ddhp1dxx
        d4hp0dx4   = self.aux_variables.d4hp0dx4
        d4hp1dx4   = self.aux_variables.d4hp1dx4


        

        delta = 0.
        #I1 = 0.666666666666667*dt*dp0dx - 2*(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dbdx/h + 2*(-dt*(-(3*p0 - p1)*dhdx - (6*p0 - 6*p1)*dbdx + h*dp0dx + p1*dhdx) + h*u1)/h + 0.333333333333333*(2*dt*p1 + h*u0)*dhdx/h + (-(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dhdx/h**2 + (-dt*(h*du1dx + p0*ddhdxx + 2*p1*ddbdxx + 2*dbdx*dp0dx + 2*dhdx*ddp0dxx) + h*dhdx + dp1dx*dhdx)/h)*h + 0.333333333333333*h*du0dx + 0.333333333333333*u0*dhdx + delta * ddp0dxx
        #I2 = -2*(-dt*(6*p0 - 6*p1) + h*w0)/h + 2*(2*dt*p1 + h*u0)*dbdx/h + (2*dt*p1 + h*u0)*dhdx/h + (-(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dhdx/h**2 + (-dt*(h*du1dx + p0*ddhdxx + 2*p1*ddbdxx + 2*dbdx*dp0dx + 2*dhdx*ddp0dxx) + h*dhdx + dp1dx*dhdx)/h)*h + delta * ddp1dxx
        I1 = 0.666666666666667*dt*dhp1dx/h - 0.666666666666667*dt*hp1*dhdx/h**2 - 2*(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dbdx/h + 2*(-dt*(-(3*hp0/h - hp1/h)*dhdx - (6*hp0/h - 6*hp1/h)*dbdx + dhp1dx) + h*w0)/h + 0.333333333333333*(2*dt*hp1/h + h*u1)*dhdx/h + (-(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dhdx/h**2 + (-dt*(ddhp0dxx + 2*hp1*ddbdxx/h + 2*dbdx*dhp1dx/h - 2*hp1*dbdx*dhdx/h**2) + h*du0dx + u0*dhdx)/h)*h + 0.333333333333333*h*du1dx + 0.333333333333333*u1*dhdx + delta *ddhp0dxx
        I2 = -2*(-dt*(6*hp0/h - 6*hp1/h) + h*w1)/h + 2*(2*dt*hp1/h + h*u1)*dbdx/h + (2*dt*hp1/h + h*u1)*dhdx/h + (-(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dhdx/h**2 + (-dt*(ddhp0dxx + 2*hp1*ddbdxx/h + 2*dbdx*dhp1dx/h - 2*hp1*dbdx*dhdx/h**2) + h*du0dx + u0*dhdx)/h)*h + delta *ddhp1dxx
        R[0] = I1 +I2
        R[1] = -I2 + I1

        return R
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        return ev


class VAMPoissonFull(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=8,
        aux_variables=['dhdt', 'dhu0dt', 'dhu1dt', 'dhw0dt', 'dhw1dt', 'dhdx', 'dhu0dx', 'dhu1dx', 'dhw0dx', 'dhw1dx', 'dhp0dx', 'dhp1dx', 'dbdx', 'hw2', 'ddp0dxx', 'ddp1dxx', 'du0dx', 'du1dx'],
        parameters={},
        _default_parameters={"g": 1},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        b = self.variables[5]
        p0 = self.variables[6]/h
        p1 = self.variables[7]/h
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 /h  


        dhdt   = self.aux_variables.dhdt   
        dhu0dt = self.aux_variables.dhu0dt 
        dhu1dt = self.aux_variables.dhu1dt 
        dhw0dt = self.aux_variables.dhw0dt 
        dhw1dt = self.aux_variables.dhw1dt 
        dhdx   = self.aux_variables.dhdx   
        dhu0dx = self.aux_variables.dhu0dx 
        dhu1dx = self.aux_variables.dhu1dx 
        dhw0dx = self.aux_variables.dhw0dx 
        dhw1dx = self.aux_variables.dhw1dx 
        dhp0dx = self.aux_variables.dhp0dx 
        dhp1dx = self.aux_variables.dhp1dx 
        dbdx   = self.aux_variables.dbdx   
        du0dx  = self.aux_variables.du0dx
        ddp0dxx = self.aux_variables.ddp0dxx
        ddp1dxx = self.aux_variables.ddp1dxx

        R[0] = dhdt 
        R[1] = dhu0dt + dhp0dx + 2 * p1 * dbdx 
        R[2] = dhu1dt -2*p1
        R[3] = dhw0dt + dhp1dx - (3*p0 - p1)*dhdx  -6*(p0-p1)*dbdx
        R[4] = 6*(p0-p1)
        R[5] = 0.
        I1 = h*du0dx + 1/3 * dhu1dx + 1/3 * u1 * dhdx + 2*(w0 - u0 * dbdx)
        I2 = h * du0dx + u1*dhdx + 2*(u1 * dbdx - w1)
        R[6] = I1 + I2
        R[7] = I1 - I2
        
        delta = 0.0
        R[6] +=  delta * (ddp0dxx )
        R[7] +=  delta * (ddp1dxx)

        return R
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        return ev

class VAMFullImplicit(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=8,
        aux_variables=[ 'hw2', 'dhdt', 'dhu0dt', 'dhu1dt', 'dhw0dt', 'dhw1dt', 'dhdx', 'dhu0dx', 'dhu1dx', 'dhw0dx', 'dhw1dx', 'dhp0dx', 'dhp1dx', 'dbdx','ddhp0dxx', 'ddhp1dxx', 'du0dx', 'du1dx'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        b = self.variables[5]
        p0 = self.variables[6]/h
        p1 = self.variables[7]/h
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 /h  


        dhdt   = self.aux_variables.dhdt   
        dhu0dt = self.aux_variables.dhu0dt 
        dhu1dt = self.aux_variables.dhu1dt 
        dhw0dt = self.aux_variables.dhw0dt 
        dhw1dt = self.aux_variables.dhw1dt 
        dhdx   = self.aux_variables.dhdx   
        dhu0dx = self.aux_variables.dhu0dx 
        dhu1dx = self.aux_variables.dhu1dx 
        dhw0dx = self.aux_variables.dhw0dx 
        dhw1dx = self.aux_variables.dhw1dx 
        dhp0dx = self.aux_variables.dhp0dx 
        dhp1dx = self.aux_variables.dhp1dx 
        dbdx   = self.aux_variables.dbdx   
        du0dx  = self.aux_variables.du0dx
        ddhp0dxx = self.aux_variables.ddhp0dxx
        ddhp1dxx = self.aux_variables.ddhp1dxx

        R[0] = dhdt 
        R[1] = dhu0dt + dhp0dx + 2 * p1 * dbdx 
        R[2] = dhu1dt -2*p1
        R[3] = dhw0dt + dhp1dx - (3*p0 - p1)*dhdx  -6*(p0-p1)*dbdx
        R[4] = 6*(p0-p1)
        R[5] = 0.
        I1 = h*du0dx + 1/3 * dhu1dx + 1/3 * u1 * dhdx + 2*(w0 - u0 * dbdx)
        I2 = h * du0dx + u1*dhdx + 2*(u1 * dbdx - w1)
        R[6] = I1 
        R[7] = I2
        
        delta = 0.1
        R[6] +=  delta * (ddhp0dxx )
        R[7] +=  delta * (ddhp1dxx)

        return R

    def flux(self):
        fx = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        
        fx[0] = hu0
        fx[1] = hu0 * u0 + 1/3 * hu1 * u1 * u1
        fx[2] = hu0 * w0 + 1/3 * hu1 * w1
        fx[3] = 2*hu0 * u1
        fx[4] = hu0 * w1 + u1 * (hw0 + 2/5*hw2)
        
        return [fx]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])

        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 / h

        nc[1, 0] = param.g * h
        nc[3, 2] = u0
        nc[4, 2] = - 1/5 * w2 + w0
        nc[1, 5] = param.g * h
        return [-nc]
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h

        ev[0] = u0
        ev[1] = u0 + 1/sqrt(3) * u1
        ev[2] = u0 - 1/sqrt(3) * u1
        ev[3] = u0 + sqrt(param.g * h + u1**2)
        ev[4] = u0 - sqrt(param.g * h + u1**2)
        ev[5] = 0
        ev[6] = 0
        ev[7] = 0
        
        return ev
    
    
    
