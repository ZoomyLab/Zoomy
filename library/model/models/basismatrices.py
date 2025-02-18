import numpy as np
from library.pysolver.solver import *
from library.model.model import *
from library.model.models.basisfunctions import Legendre_shifted

class Basismatrices():
    def __init__(self, basis=Legendre_shifted()):
        self.basisfunctions = basis
    
    def compute_matrices(self, level):
        self.M = np.empty((level+1, level+1), dtype=float)
        self.A = np.empty((level+1, level+1, level+1), dtype=float)
        self.B = np.empty((level+1, level+1, level+1), dtype=float)
        self.D = np.empty((level+1, level+1), dtype=float)
        self.DD = np.empty((level+1, level+1), dtype=float)
        self.D1 = np.empty((level+1, level+1), dtype=float)
        self.DT = np.empty((level+1, level+1, level+1), dtype=float)

        for k in range(level+1):
            for i in range(level+1):
                self.M[k, i] = self._M(k, i)
                self.D[k, i] = self._D(k, i)
                self.DD[k, i] = self._DD(k, i)
                self.D1[k, i] = self._D1(k, i)
                for j in range(level+1):
                    self.A[k, i, j] = self._A(k, i, j)
                    self.B[k, i, j] = self._B(k, i, j)
                    self.DT[k, i, j] = self._DT(k, i, j)

    def enforce_boundary_conditions_lsq(self, rhs=np.zeros(2), dim=1):
        level = len(self.basisfunctions.basis)-1
        constraint_bottom = [self.basisfunctions.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basisfunctions.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
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
        level = len(self.basisfunctions.basis)-1
        constraint_bottom = [self.basisfunctions.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basisfunctions.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
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
        level = len(self.basisfunctions.basis)-1
        constraint_bottom = [self.basisfunctions.eval(i, 0.) for i in range(level+1)]
        constraint_top = [diff(self.basisfunctions.eval(i, x), x).subs(x, 1.) for i in range(level+1)]
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
        return integrate(self.basisfunctions.eval(k, x) * self.basisfunctions.eval(i, x), (x, 0, 1))

    """ 
    Compute <phi_k, phi_i, phi_j>
    """
    def _A(self, k, i, j):
        return integrate(self.basisfunctions.eval(k, x) * self.basisfunctions.eval(i, x) * self.basisfunctions.eval(j, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, phi_j, int(phi)_j>
    """
    def _B(self, k, i, j):
        return integrate(diff(self.basisfunctions.eval(k, x), x) * integrate(self.basisfunctions.eval(j, x), x) * self.basisfunctions.eval(i, x), (x, 0, 1))

    """ 
    Compute <(phi')_k, (phi')_j>
    """
    def _D(self, k, i):
        return integrate(diff(self.basisfunctions.eval(k, x), x) * diff(self.basisfunctions.eval(i, x), x), (x, 0, 1))

    """ 
    Compute <(phi)_k, (phi')_j>
    """
    def _D1(self, k, i):
        return integrate(self.basisfunctions.eval(k, x) * diff(self.basisfunctions.eval(i, x), x), (x, 0, 1))

    """ 
    Compute <(phi)_k, (phi'')_j>
    """
    def _DD(self, k, i):
        return integrate(self.basisfunctions.eval(k, x) * diff(diff(self.basisfunctions.eval(i, x), x), x), (x, 0, 1))
    """ 

    Compute <(phi')_k, (phi')_j>
    """
    def _DT(self, k, i, j):
        return integrate(diff(self.basisfunctions.eval(k, x), x) * diff(self.basisfunctions.eval(i, x), x) * self.basisfunctions.eval(j, x), (x, 0, 1))

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
