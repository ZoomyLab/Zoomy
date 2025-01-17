import numpy as np
import os
import re
import logging
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
import matplotlib.pyplot as plt
import inspect
import pytest
from copy import deepcopy
from scipy import interpolate
from scipy.optimize import least_squares as lsq
import sympy
from sympy import Symbol, Matrix, lambdify
# from sympy import *
from sympy import zeros, ones
from sympy import bspline_basis, bspline_basis_set
from sympy.abc import x
import h5py

from library.model.models.base import register_sympy_attribute, eigenvalue_dict_to_matrix
from library.model.models.base import Model
import library.model.initial_conditions as IC


# from library.solver.baseclass import BaseYaml
from library.model import *
# from library.solver.boundary_conditions import *
# import library.solver.initial_condition as initial_condition
# import library.solver.smm_model as smm
# import library.solver.smm_model_hyperbolic as smmh
# import library.solver.smm_model_exner as smm_exner
# import library.solver.smm_model_exner_hyperbolic as smm_exner_hyper

# from library.solver.baseclass import BaseYaml  # nopep8

# main_dir = os.getenv("SMPYTHON")
# eps = 10 ** (-10)

from attr import define
from sympy import integrate, diff

from sympy import legendre
from sympy import lambdify

class Legendre_shifted:
    def basis_definition(self):
        x = Symbol('x')
        b = lambda k, x: legendre(k, 2*x-1) * (-1)**(k)
        # b = lambda k, x: legendre(k, 2*x-1)
        return [b(k, x) for k in range(self.order+1)]

    def __init__(self, order = 1, **kwargs):
        self.order = order
        self.basis = self.basis_definition(**kwargs)


    def get(self, k):
        return self.basis[k]
    
    def eval(self, k, z):
        return self.get(k).subs(x, z)

    # TODO vectorization not yet working properly without loop
    def get_lambda(self, k):
        # s = str(self.get(k)).replace('x', 'z')
        # s = re.sub(r'([^+-]+)', r'np.ones_like(z)*(\1)', s)
        # lam = lambda z: eval(s)
        f = lambdify(x, self.get(k)) 
        def lam(z):
            if type(z) == int or type(z) == float:
                return f(z)
            elif type(z) == list or type(z) == np.ndarray:
                return np.array([f(xi) for xi in z])
            else: 
                assert False
        return lam

    def plot(self):
        fig, ax = plt.subplots()
        X = np.linspace(0,1,1000)
        for i in range(len(self.basis)):
            # print(self.get(i))
            f = lambdify(x, self.get(i)) 
            y = np.array([f(xi) for xi in X])
            ax.plot(X, y, label=f"basis {i}")
        plt.legend()
        plt.show()

    def reconstruct_velocity_profile(self, alpha, Z=np.linspace(0,1,100)):
        u = np.zeros_like(Z)
        for i in range(len(self.basis)):
            b = lambdify(x, self.get(i)) 
            u[:] += alpha[i] * b(Z)
        return u

    def reconstruct_alpha(self, velocities, z):
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        for i in range(n_basis):
            b = lambdify(x, self.get(i)) 
            nom = np.trapz(velocities * b(z), z) 
            if type(b(z)) == int:
                den = b(z)**2
            else:
                den = np.trapz((b(z) * b(z)).reshape(z.shape), z)
            res = nom/den
            alpha[i] = res

        #mean = np.mean(velocities)
        #print('----------------------------')
        #U = np.zeros_like(z)
        #for i in range(n_basis):
        #    print(i)
        #    b = lambdify(x, self.get(i)) 
        #    U += alpha[i] * b(z)
        #    print(np.sum((U - velocities))**2)
        #print(mean)
        #print(alpha)
        #print(velocities)
        #print(U)
        #print('----------------------------')
        return alpha

    def get_diff_basis(self):
        db = [diff(b, x) for i, b in enumerate(self.basis)]
        self.basis = db


class Spline(Legendre_shifted):
    def basis_definition(self, degree=1, knots = [0, 0, 0.001, 1, 1]):
        x = Symbol('x')
        basis = bspline_basis_set(degree, knots, x)
        return basis

class OrthogonalSplineWithConstant(Legendre_shifted):
    def basis_definition(self, degree=1, knots = [0, 0, 0.5, 1, 1]):
        x = Symbol('x')
        def prod(u, v):
            return integrate(u*v, (x, 0, 1))
        basis = bspline_basis_set(degree, knots, x)
        add_basis = [1]
        # add_basis = [sympy.Piecewise((0, x<0.1), (1, True))]
        basis = add_basis + basis[:-1]
        orth = deepcopy(basis)
        for i in range(1, len(orth)):
            for j in range(0,i):
                orth[i] -= prod(basis[i], orth[j]) / prod(orth[j], orth[j]) * orth[j]
        for i in range(len(orth)):
            orth[i] /= sympy.sqrt(prod(orth[i], orth[i]))

        # str = ""
        # for i in range(len(orth)):
        #     for j in range(len(orth)):
        #         str += f"{(prod(orth[i], orth[j]))} "
        #     str += "\n"
        # print(str)
        return orth
    


class Basis():
    def __init__(self, basis=Legendre_shifted()):
        self.basis = basis
    
    def compute_matrices(self, level):
        self.M = np.empty((level+1, level+1), dtype=float)
        self.A = np.empty((level+1, level+1, level+1), dtype=float)
        self.B = np.empty((level+1, level+1, level+1), dtype=float)
        self.D = np.empty((level+1, level+1), dtype=float)

        for k in range(level+1):
            for i in range(level+1):
                self.M[k, i] = self._M(k, i)
                self.D[k, i] = self._D(k, i)
                for j in range(level+1):
                    self.A[k, i, j] = self._A(k, i, j)
                    self.B[k, i, j] = self._B(k, i, j)

    def enforce_boundary_conditions_lsq(self, rhs=np.zeros(2), dim=1):
        level = len(self.basis.basis)-1
        constraint_bottom = [self.basis.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basis.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1+level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        AtA = A_enforce.T @ A_enforce
        reg = 10**(-6)
        A_enforce_inv = np.array((AtA + reg * np.eye(AtA.shape[0])).inv(), dtype=float)
        def f_1d(Q):
            for i, q in enumerate(Q.T):
                # alpha_enforce = q[I_enforce+1]
                alpha_free = q[I_free+1]
                b = rhs -np.dot(A_free, alpha_free)
                # b = rhs 
                result =  np.dot(A_enforce_inv, A_enforce.T @ b)
                alpha = 1.0
                Q[I_enforce+1, i] = (1-alpha) * Q[I_enforce+1, i] +  (alpha) * result
            return Q
        def f_2d(Q):
            i1 = [[0] + [i+1 for i in range(1+level)]]
            i2 = [[0] + [i+1+1+level for i in range(1+level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q
        if dim==1:
            return f_1d
        elif dim==2:
            return f_2d
        else:
            assert False

    def enforce_boundary_conditions_lsq2(self, rhs=np.zeros(2), dim=1):
        level = len(self.basis.basis)-1
        constraint_bottom = [self.basis.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basis.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1+level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        def obj(alpha0, lam):
            def f(alpha):
                return  np.sum((alpha-alpha0)**2) + lam * np.sum(np.array(np.dot(A, alpha)**2, dtype=float))
            return f
        def f_1d(Q):
            for i, q in enumerate(Q.T):
                h = q[0] 
                alpha = q[1:] / h 
                f = obj(alpha, 0.1)
                result = lsq(f, alpha)
                Q[1:,i] =  h* result.x
            return Q
        def f_2d(Q):
            i1 = [[0] + [i+1 for i in range(1+level)]]
            i2 = [[0] + [i+1+1+level for i in range(1+level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q
        if dim==1:
            return f_1d
        elif dim==2:
            return f_2d
        else:
            assert False
    
    def enforce_boundary_conditions(self, enforced_basis=[-2, -1], rhs=np.zeros(2), dim=1):
        level = len(self.basis.basis)-1
        constraint_bottom = [self.basis.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basis.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
        A = Matrix([constraint_bottom, constraint_top][:len(enforced_basis)])

        # test to only constrain bottom
        # A = Matrix([constraint_bottom])
        # enforced_basis = [-1]
        # rhs=np.zeros(1)

        I = np.linspace(0, level, 1+level, dtype=int)
        I_enforce = I[enforced_basis]
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        A_enforce_inv = np.array(A_enforce.inv(), dtype=float)
        def f_1d(Q):
            for i, q in enumerate(Q.T):
                alpha_enforce = q[I_enforce+1]
                alpha_free = q[I_free+1]
                b = rhs - np.dot(A_free, alpha_free)
                result =  np.dot(A_enforce_inv, b)
                alpha = 1.0
                Q[I_enforce+1, i] = (1-alpha) * Q[I_enforce+1, i] +  (alpha) * result
            return Q
        def f_2d(Q):
            i1 = [[0] + [i+1 for i in range(1+level)]]
            i2 = [[0] + [i+1+1+level for i in range(1+level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q
        if dim==1:
            return f_1d
        elif dim==2:
            return f_2d
        else:
            assert False

        

    """ 
    Compute <phi_k, phi_i>
    """
    def _M(self, k, i):
        return integrate(self.basis.eval(k, x) * self.basis.eval(i, x), (x, 0, 1))

    """ 
    Compute <phi_k, phi_i, phi_j>
    """
    def _A(self, k, i, j):
        return integrate(self.basis.eval(k, x) * self.basis.eval(i, x) * self.basis.eval(j, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, phi_j, int(phi)_j>
    """
    def _B(self, k, i, j):
        return integrate(diff(self.basis.eval(k, x), x) * integrate(self.basis.eval(j, x), x) * self.basis.eval(i, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, (phi')_j>
    """
    def _D(self, k, i):
        return integrate(diff(self.basis.eval(k, x), x) * diff(self.basis.eval(i, x), x), (x, 0, 1))

class BasisNoHOM(Basis):
    def _A(self, k, i, j):
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        if ((i==0 and j==k) or (j==0 and i == k) or (k==0 and i == j)):
            return super()._A(k, i, j)
        return 0

    def _B(self, k, i, j):
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        # if not (i==0 or j==0):
        if ((i==0 and j==k) or (j==0 and i == k) or (k==0 and i == j)):
            return super()._B(k, i, j)
        return 0
        # return super()._B(k, i, j)

class HybridSFFSMM(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 3
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )
   
    def get_alphas(self): 
        Q = self.variables
        h = Q[0]
        # exlude h and P
        ha = Q[1:-1]
        return ha


    def flux(self):
        flux_x = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.get_alphas()
        a = [_ha / h for _ha in ha]
        hu = self.variables[1]
        u = hu / h
        P11 = self.variables[-1]
        p = self.parameters
        # mass malance
        flux_x[0] = ha[0]
        # mean momentum (following SSF)
        flux_x[1] = hu * u + h * P11 + p.g * h**2 / 2
        # flux_x[1] = 0
        # for i in range(self.levels+1):
        #     for j in range(self.levels+1):
        #         flux_x[1] += ha[i] * ha[j] / h * self.basis.A[0, i, j] / self.basis.M[ 0, 0 ]
        # higher order moments (SMM)
        for k in range(1, self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    flux_x[k+1] += h * a[i] * a[j] * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        # P
        flux_x[-1] = 2* P11 * u
        for k in range(1, self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    flux_x[-1] += a[i] * a[j] * a[k] * self.basis.A[k, i, j] 
        return [flux_x]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.get_alphas()
        hu = self.variables[1]
        u = hu / h
        a = [_ha / h for _ha in ha]
        P11 = self.variables[-1]
        p = self.parameters
        um = ha[0]/h

        # mean momentum 
        # nc[1, 0] = - p.g * p.ez * h 
        nc[1, 0] = 0
        # higher order momennts (SMM)
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
        nc[-1, -1] = u
        return [nc]

    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        # return eigenvalue_dict_to_matrix(A.eigenvals())
        evs = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        hu = self.variables[1]
        u = hu/h
        P11 = self.variables[2]
        p = self.parameters

        b = sympy.sqrt(P11)
        a = sympy.sqrt(p.g * h + 3*P11)

        evs[0] = u
        evs[1] = u + a
        evs[2] = u - a

        return evs


    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

class ShallowMomentsAugmentedSSF(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=3,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 2
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )
   
    def get_alphas(self): 
        Q = self.variables
        h = Q[0]
        # exlude u0 and P
        ha = Q[2:-1]
        P = Q[-1]
        sum_aiMii = 0
        N = self.levels
        for i, hai in enumerate(ha):
            sum_aiMii += hai * self.basis.M[i+1, i+1] / h
        
        aN = sympy.sqrt((P - h * sum_aiMii) / (h * self.basis.M[N, N]))
        # now I want to include u0
        ha = Q[1:]
        ha[-1] = aN * h
        return ha


    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.get_alphas()
        P11 = self.variables[-1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux[k+1] += ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        flux[-1] = 2* P11 * ha[0] / h
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.get_alphas()
        P11 = self.variables[-1]
        p = self.parameters
        um = ha[0]/h
        # nc[1, 0] = - p.g * p.ez * h 
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
        for k in range(nc.shape[1]):
            nc[-1, k] = 0
        nc[-1, 1] = - P11
        nc[-1, -1] = + P11
        return [nc]

    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out


class ShallowMoments(Model):
    """
    Shallow Moments 1d

    :gui: 
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=2,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 2
        self.basis = basis
        self.basis.basis = type(self.basis.basis)(order=self.levels)
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux[k+1] +=  ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        um = ha[0]/h
        # nc[1, 0] = - p.g * p.ez * h 
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -=  ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
        return [nc]

    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2:]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def inclined_plane(self):
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        out[1] = h * p.g * (p.ex)
        return out

    def material_wave(self):
        assert "nu" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        return out
        

    def newtonian(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        for k in range(1+self.levels):
            for i in range(1+self.levels):
                #out[1+k] += -p.nu/h * p.eta_bulk * ha[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
                out[1+k] += -p.nu/h *  ha[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
        return out

    def newtonian_boundary_layer(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        phi_0 = [self.basis.basis.eval(i, 0.) for i in range(self.levels+1)]
        dphidx_0 = [(diff(self.basis.basis.eval(i, x), x)).subs(x, 0.) for i in range(self.levels+1)]
        dz_boundary_layer = 0.005
        u_bot = 0
        for i in range(1+self.levels):
            u_bot += ha[i] / h / self.basis.M[i, i] 
        tau_bot = p.nu * (u_bot - 0.)/dz_boundary_layer
        for k in range(1+self.levels):
            out[k+1] = - p.eta * tau_bot / h
        return out

    def steady_state_channel(self):
        assert "eta_ss" in vars(self.parameters)
        moments_ss = np.array([0.21923893, -0.04171894, -0.05129916, -0.04913612, -0.03863209, -0.02533469, -0.0144186, -0.00746847, -0.0031811, -0.00067986, 0.0021782])[:self.levels+1]
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        for i in range(1,self.levels+1):
            out[1+i] =  - p.eta_ss * h * (ha[i]/h - moments_ss[i])
        return out

    def slip(self):
        """
        :gui:
            - requires_parameter: ('lamda', 0.0)
            - requires_parameter: ('rho', 1.0)
        """
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        for k in range(1+self.levels):
            for i in range(1+self.levels):
                out[1+k] += -1./p.lamda/p.rho * ha[i]  / h / self.basis.M[k, k]
        return out

    def no_slip(self):
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        a0 = [ha[i] / h for i in range(self.levels+1)]
        a = [ha[i] / h for i in range(self.levels+1)]
        # for i in range(self.levels+1):
        #     out[i+1] = ha[i]
        phi_0 = np.zeros(self.levels+1)
        ns_iterations = 2
        for k in range(self.levels+1):
            phi_0[k] = self.basis.basis.eval(k, x).subs(x, 0.)
        def f(j, a, a0, basis_0):
            out = 0
            for i in range(self.levels+1):
                # out += -2*p.ns_1*(a[i] - a0[i]) -2*p.ns_2*basis_0[j] * a[i] * basis_0[i]
                out += -2*p.ns_2*basis_0[j] * a[i] * basis_0[i]
            return out
        for i in range(ns_iterations):
            for k in range(1, 1+self.levels):
                out[1+k] += h*(f(k, a, a0, phi_0))
            a = [a[k] + out[k] / h for k in range(self.levels+1)]
        # return sympy.simplify(out)
        return out

    def chezy(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        for i in range(1+self.levels):
            for j in range(1+self.levels):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1+self.levels):
            for l in range(1+self.levels):
                out[1+k] += -1./(p.C**2 * self.basis.M[k,k]) * ha[l] * sqrt / h 
        return out

    def chezy_ssf(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        for i in range(1+self.levels):
            for j in range(1+self.levels):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1+self.levels):
            for l in range(1+self.levels):
                out[1+k] += -(p.Cf * self.basis.M[k,k]) * ha[l] * sqrt / h 
        return out

    def shear_new(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.levels+1)
        phi_0 = np.zeros(self.levels+1)
        for k in range(self.levels+1):
            d_phi_0[k] = diff(self.basis.basis.eval(k, x), x).subs(x, 0.)
            phi_0[k] = self.basis.basis.eval(k, x).subs(x, 0.)
        friction_factor = 0.
        for k in range(self.levels+1):
            friction_factor -= p.beta * d_phi_0[k] * ha[k]/h/h
        k = 0
        # for k in range(1+self.levels):
        out[1+k] += friction_factor * phi_0[k] /(self.basis.M[k,k]) 
        return out

    def shear(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.levels+1)
        phi_0 = np.zeros(self.levels+1)
        for k in range(self.levels+1):
            d_phi_0[k] = diff(self.basis.basis.eval(k, x), x).subs(x, 0.)
            phi_0[k] = self.basis.basis.eval(k, x).subs(x, 0.)
        friction_factor = 0.
        for k in range(self.levels+1):
            friction_factor -= p.beta * d_phi_0[k] * phi_0[k] * ha[k]/h
        for k in range(1+self.levels):
            out[1+k] += friction_factor /(self.basis.M[k,k]) 
        return out

    def shear_crazy(self):
        """
        :gui:
            - requires_parameter: ('beta', 1.0)
        """
        assert "beta" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.levels+1)
        phi_0 = np.zeros(self.levels+1)
        for k in range(self.levels+1):
            d_phi_0[k] = diff(self.basis.basis.eval(k, x), x).subs(x, 0.)
            phi_0[k] = self.basis.basis.eval(k, x).subs(x, 0.)
        for k in range(self.levels+1):
            out[1+k] += -p.beta * d_phi_0[k] * phi_0[k] * ha[k]/h
        return out

    def manning_mean(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        tmp += ha[0] * ha[0] / h / h
        sqrt = sympy.sqrt(tmp)
        k = 0
        l = 0
        out[1+k] += -1.*p.C * (self.basis.M[k,k]) * ha[l] * sqrt / h 
        return out

    def steady_state(self):
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        tmp = 0
        d_phi_0 = np.zeros(self.levels+1)
        phi_0 = np.zeros(self.levels+1)
        for k in range(self.levels+1):
            d_phi_0[k] = diff(self.basis.basis.eval(k, x), x).subs(x, 0.)
            phi_0[k] = self.basis.basis.eval(k, x).subs(x, 0.)
        shear_factor = 0.
        u_bottom = 0.
        u_diff = ha[0]/h
        for k in range(self.levels+1):
            u_diff +=  (phi_0[k] * ha[k]/h) 
        for k in range(self.levels+1):
            shear_factor +=  d_phi_0[k] *  (ha[k]/h - eval(f'p.Q_ss{k+1}')/eval(f'p.Q_ss{0}'))
        # for k in range(1, self.levels+1):
        #     out[1+k] += - shear_factor *  np.abs(u_diff) * p.S *  phi_0[k] /(self.basis.M[k,k]) 
        for k in range(1, self.levels+1):
            # out[1+k] += - p.A * np.abs(u_diff)* (ha[k]/h - eval(f'p.Q_ss{k+1}')/eval(f'p.Q_ss{0}'))
            out[1+k] += - p.A * np.abs(ha[0]/h) * (ha[k]/h - eval(f'p.Q_ss{k+1}')/eval(f'p.Q_ss{0}'))
        return out

class ShallowMomentsSSF(ShallowMoments):

    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        # alpha_erase = self.variables[2:]
        # for alpha_i in alpha_erase:
        #     A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(sympy.simplify(A).eigenvals())

class ShallowMomentsSSFEnergy(ShallowMoments):
    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1] - self.variables[1] / np.diag(self.basis.M)
        ha[1:] -= self.variables[1] * np.diag(self.basis.M)[1:]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux[k+1] += ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]

        flux_hu = flux[1]

        for k in range(1,self.levels+1):
            flux[k+1] += flux_hu / self.basis.M[k, k]
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1] - self.variables[1] / np.diag(self.basis.M)
        ha[1:] -= self.variables[1] * np.diag(self.basis.M)[1:]
        p = self.parameters
        um = ha[0]/h
        # nc[1, 0] = - p.g * p.ez * h 
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
        return [nc]

    def chezy(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1] - self.variables[1] / np.diag(self.basis.M)
        ha[1:] -= self.variables[1] * np.diag(self.basis.M)[1:]
        p = self.parameters
        tmp = 0
        for i in range(1+self.levels):
            for j in range(1+self.levels):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1+self.levels):
            for l in range(1+self.levels):
                out[1+k] += -1./(p.C**2 * self.basis.M[k,k]) * ha[l] * sqrt / h 

        for k in range(1+self.levels):
            out[1+k] += out[1] / self.basis.M[k, k]

        return out



    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        return eigenvalue_dict_to_matrix(sympy.simplify(A).eigenvals())

class ShallowMomentsTurbulenceSimple(ShallowMoments):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=4,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        super().__init__(
            dimension=dimension,
            fields=fields-2,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
            basis = basis,
        )
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = variables.length()
        assert n_fields >= 4

class ShallowMoments2d(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters = {},
        parameters_default={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis()
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = int((self.n_fields - 1)/2)-1
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default = parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        offset = self.levels+1
        flux_x = Matrix([0 for i in range(self.n_fields)])
        flux_y = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+self.levels+1:1+2*(self.levels+1)]
        p = self.parameters
        flux_x[0] = ha[0]
        flux_x[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux_x[k+1] += ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux_x[k+1+offset] += hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]

        flux_y[0] = hb[0]
        flux_y[1+offset] = p.g * p.ez * h * h / 2
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux_y[k+1] += hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        for k in range(self.levels+1):
            for i in range(self.levels+1):
                for j in range(self.levels+1):
                    # TODO avoid devision by zero 
                    flux_y[k+1+offset] += hb[i] * hb[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        offset = self.levels+1
        nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        nc_y = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+offset+self.levels+1]
        p = self.parameters
        um = ha[0]/h
        vm = hb[0]/h
        for k in range(1, self.levels+1):
            nc_x[k+1, k+1] += um
            nc_y[k+1, k+1+offset] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc_x[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
                    nc_y[k+1, i+1+offset] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]

        for k in range(1, self.levels+1):
            nc_x[k+1+offset, k+1] += vm
            nc_y[k+1+offset, k+1+offset] += vm
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc_x[k+1+offset, i+1] -= hb[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
                    nc_y[k+1+offset, i+1+offset] -= hb[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
        return [nc_x, nc_y]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.levels+1
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2:2+self.levels]
        beta_erase = self.variables[2+offset : 2+offset+self.levels]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        for beta_i in beta_erase:        
            A = A.subs(beta_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        assert "dhdy" in vars(self.aux_variables)
        offset = self.levels+1
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        dhdy = self.aux_variables.dhdy
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        out[1+offset] = h * p.g * (p.ey - p.ez * dhdy)
        return out


    def newtonian(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels+1
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+self.levels+1+offset]
        p = self.parameters
        for k in range(1+self.levels):
            for i in range(1+self.levels):
                out[1+k] += -p.nu/h * ha[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
                out[1+k+offset] += -p.nu/h * hb[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
        return out

    def newtonian_boundary_layer(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels+1
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+self.levels+1+offset]
        p = self.parameters
        phi_0 = [self.basis.eval(i, 0.) for i in range(self.levels+1)]
        dphidx_0 = [(diff(self.basis.eval(i, x), x)).subs(x, 0.) for i in range(self.levels+1)]
        for k in range(1+self.levels):
            for i in range(1+self.levels):
                out[1+k] += -p.nu / h * ha[i] / h / self.basis.M[k, k] * phi_0[k] * dphidx_0[i]
                out[1+k+offset] += -p.nu / h  * hb[i]  / h / self.basis.M[k, k]* phi_0[k] * dphidx_0[i]
        return out


    def sindy(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels+1
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+self.levels+1+offset]
        p = self.parameters
        out[1] += p.C1 * sympy.Abs(ha[0]/h) + p.C2 * sympy.Abs(ha[1]/h) + p.C3 * sympy.Abs(ha[0]/h)**(7/3)+ p.C4 * sympy.Abs(ha[1]/h)**(7/3)
        out[2] += p.C5 * sympy.Abs(ha[0]/h) + p.C6 * sympy.Abs(ha[1]/h) + p.C7 * sympy.Abs(ha[0]/h)**(7/3)+ p.C8 * sympy.Abs(ha[1]/h)**(7/3)
        out[3] += p.C1 * sympy.Abs(ha[0]/h) + p.C2 * sympy.Abs(ha[1]/h) + p.C3 * sympy.Abs(ha[0]/h)**(7/3)+ p.C4 * sympy.Abs(ha[1]/h)**(7/3)
        out[4] += p.C5 * sympy.Abs(ha[0]/h) + p.C6 * sympy.Abs(ha[1]/h) + p.C7 * sympy.Abs(ha[0]/h)**(7/3)+ p.C8 * sympy.Abs(ha[1]/h)**(7/3)
        return out

    def slip(self):
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels+1
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+self.levels+1+offset]
        p = self.parameters
        for k in range(1+self.levels):
            for i in range(1+self.levels):
                out[1+k] += -1./p.lamda/p.rho * ha[i]  / h / self.basis.M[k, k]
                out[1+k+offset] += -1./p.lamda/p.rho * hb[i]  / h / self.basis.M[k, k]
        return out

    def chezy(self):
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels+1
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        hb = self.variables[1+offset:1+self.levels+1+offset]
        p = self.parameters
        tmp = 0
        for i in range(1+self.levels):
            for j in range(1+self.levels):
                tmp += ha[i] * ha[j] / h / h + hb[i] * hb[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1+self.levels):
            for l in range(1+self.levels):
                out[1+k] += -1./(p.C**2 * self.basis.M[k,k]) * ha[l] * sqrt / h 
                out[1+k+offset] += -1./(p.C**2 * self.basis.M[k,k]) * hb[l] * sqrt / h 
        return out


def reconstruct_uvw(Q, grad, lvl, phi, psi):
    """
    returns functions u(z), v(z), w(z)
    """
    offset = lvl + 1
    h = Q[0]
    alpha = Q[1 : 1 + offset] / h
    beta = Q[1 + offset : 1 + 2 * offset] / h
    dhalpha_dx = grad[1 : 1 + offset, 0]
    dhbeta_dy = grad[1 + offset : 1 + 2 * offset, 1]

    def u(z):
        u_z = 0
        for i in range(lvl + 1):
            u_z += alpha[i] * phi(z)[i]
        return u_z

    def v(z):
        v_z = 0
        for i in range(lvl + 1):
            v_z += beta[i] * phi(z)[i]
        return v_z

    def w(z):
        basis_0 = psi(0)
        basis_z = psi(z)
        u_z = 0
        v_z = 0
        grad_h = grad[0, :]
        # grad_hb = grad[-1, :]
        grad_hb = np.zeros(grad[0,:].shape)
        result = 0
        for i in range(lvl + 1):
            u_z += alpha[i] * basis_z[i]
            v_z += beta[i] * basis_z[i]
        for i in range(lvl + 1):
            result -= dhalpha_dx[i] * (basis_z[i] - basis_0[i])
            result -= dhbeta_dy[i] * (basis_z[i] - basis_0[i])

        result += u_z * (z * grad_h[0] + grad_hb[0])
        result += v_z * (z * grad_h[1] + grad_hb[1])
        return result
    return u, v, w

def generate_velocity_profiles(
    Q,
    centers,
    model: Model,
    list_of_positions: list[np.ndarray],
):
    def find_closest_element(centers, pos):
        assert centers.shape[1] == np.array(pos).shape[0]
        return np.argmin(np.linalg.norm(centers - pos, axis=1))

    # find the closest element to the given position
    vertices = []
    for pos in list_of_positions:
        vertex = find_closest_element(centers, pos)
        vertices.append(vertex)

    Z = np.linspace(0,1,100)
    list_profiles = []
    list_means = []
    level = int((model.n_fields-1)/model.dimension)-1
    offset = level + 1
    list_h = []
    for vertex in vertices:
        profiles = []
        means = []
        for d in range(model.dimension):
            q = Q[vertex, :]
            h = q[0]
            coefs = q[1+d*offset:1+(d+1)*offset]/h
            profile = model.basis.basis.reconstruct_velocity_profile(coefs, Z=Z)
            mean = coefs[0]
            profiles.append(profile)
            means.append(mean)
        list_profiles.append(profiles)
        list_means.append(means)
        list_h.append(h)
    return list_profiles, list_means, list_of_positions, Z, list_h

            


    
if __name__ == "__main__":
    # basis = Legendre_shifted(1)
    # basis = Spline()
    # basis = OrthogonalSplineWithConstant(degree=2, knots=[0, 0.1, 0.3,0.5, 1,1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.02, 0.04, 0.06, 0.08, 0.1,  1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.1, 1])
    # basis.plot()

    basis = Basis(basis=Legendre_shifted(order=4))
    f = basis.enforce_boundary_conditions()
    q = np.array([[1., 0.1, 0., 0., 0., 0.], [1., 0.1, 0., 0., 3., 0.]])
    print(f(q))


    # basis =Legendre_shifted(order=8)
    # basis.plot()
    # z = np.linspace(0,1,100)
    # f = basis.get_lambda(1)
    # print(f(z), f(1.0))
    # f = basis.get_lambda(1)
    # print(f(z))


    # X = np.linspace(0,1,100)
    # U = basis.reconstruct_velocity_profile([0, 0.0, 0., 0, 1], Z=X)
    # plt.plot(U, X)
    # plt.show()

