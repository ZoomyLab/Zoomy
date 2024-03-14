import numpy as np
import os
import logging
import numpy.polynomial.legendre as L
import numpy.polynomial.chebyshev as C
import matplotlib.pyplot as plt
import inspect
import pytest
from copy import deepcopy
from scipy import interpolate
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

class Legendre_shifted:
    def basis_definition(self):
        x = Symbol('x')
        b = lambda k, x: legendre(k, 2*x-1) * (-1)**(k)
        return [b(k, x) for k in range(self.order+1)]

    def __init__(self, order = 1, **kwargs):
        self.order = order
        self.basis = self.basis_definition(**kwargs)


    def get(self, k):
        return self.basis[k]
    
    def eval(self, k, z):
        return self.get(k).subs(x, z)

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


class Spline(Legendre_shifted):
    def basis_definition(self, degree=1, knots = [0, 0, 0.5, 1, 1]):
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
    
    def enforce_boundary_conditions(self, enforced_basis=[-2, -1], rhs=np.zeros(2)):
        level = len(self.basis.basis)-1
        constraint_bottom = [self.basis.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basis.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
        A = Matrix([constraint_bottom, constraint_top])
        I = np.linspace(0, level, 1+level, dtype=int)
        I_enforce = I[enforced_basis]
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        A_enforce_inv = np.array(A_enforce.inv(), dtype=float)
        def f(Q):
            for i, q in enumerate(Q):
                q_enforce = q[I_enforce+1]
                q_free = q[I_free+1]
                b = rhs - np.dot(A_free, q_free)
                result =  np.dot(A_enforce_inv, b)
                Q[i, I_enforce+1] = result
            return Q
        return f
            
        

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
                    flux[k+1] += ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[ k, k ]
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1:1+self.levels+1]
        p = self.parameters
        um = ha[0]/h
        for k in range(1, self.levels+1):
            nc[k+1, k+1] += um
        for k in range(self.levels+1):
            for i in range(1, self.levels+1):
                for j in range(1, self.levels+1):
                    nc[k+1, i+1] -= ha[j]/h*self.basis.B[k, i, j]/self.basis.M[k, k]
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
        b = 
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
                out[1+k] += -p.nu/h * ha[i]  / h * self.basis.D[i, k]/ self.basis.M[k, k]
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


class ShallowMoments2d(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
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
    basis = OrthogonalSplineWithConstant(degree=2, knots=[0, 0.1, 0.3,0.5, 1,1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.02, 0.04, 0.06, 0.08, 0.1,  1])
    # basis=OrthogonalSplineWithConstant(degree=1, knots=[0,0, 0.1, 1])
    basis.plot()

    # basis = Basis(basis=Legendre_shifted(order=2))
    # f = basis.enforce_boundary_conditions()
    # q = np.array([[1., 0.1, 0., 0.]])
    # print(f(q))


    # X = np.linspace(0,1,100)
    # U = basis.reconstruct_velocity_profile([0, 0.0, 0., 0, 1], Z=X)
    # plt.plot(U, X)
    # plt.show()
