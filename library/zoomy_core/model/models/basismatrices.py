import os
import numpy as np
import sympy
from sympy import integrate, diff, Matrix
from sympy.abc import z
from time import time as get_time

from scipy.optimize import least_squares as lsq

from zoomy_core.model.models.basisfunctions import Legendre_shifted


class Basismatrices:
    def __init__(self, basis=Legendre_shifted(), use_cache=True, cache_path=".cache"):
        self.basisfunctions = basis
        self.use_cache = use_cache
        self.cache_dir = cache_path
        self.cache_subdir = f"basismatrices/{basis.name}/{basis.level}"

    def load_cached_matrices(self):
        main_dir = os.getenv("ZOOMY_DIR")
        path = os.path.join(os.path.join(main_dir, self.cache_dir), self.cache_subdir)
        failed = False
        try:
            self.phib = np.load(os.path.join(path, "phib.npy"))
            self.M = np.load(os.path.join(path, "M.npy"))
            self.A = np.load(os.path.join(path, "A.npy"))
            self.B = np.load(os.path.join(path, "B.npy"))
            self.D = np.load(os.path.join(path, "D.npy"))
            self.Dxi = np.load(os.path.join(path, "Dxi.npy"))
            self.Dxi2 = np.load(os.path.join(path, "Dxi2.npy"))
            self.DD = np.load(os.path.join(path, "DD.npy"))
            self.D1 = np.load(os.path.join(path, "D1.npy"))
            self.DT = np.load(os.path.join(path, "DT.npy"))
        except:
            failed = True
        return failed

    def save_cached_matrices(self):
        main_dir = os.getenv("ZOOMY_DIR")
        path = os.path.join(os.path.join(main_dir, self.cache_dir), self.cache_subdir)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "phib"), self.phib)
        np.save(os.path.join(path, "M"), self.M)
        np.save(os.path.join(path, "A"), self.A)
        np.save(os.path.join(path, "B"), self.B)
        np.save(os.path.join(path, "D"), self.D)
        np.save(os.path.join(path, "Dxi"), self.Dxi)
        np.save(os.path.join(path, "Dxi2"), self.Dxi2)
        np.save(os.path.join(path, "DD"), self.DD)
        np.save(os.path.join(path, "D1"), self.D1)
        np.save(os.path.join(path, "DT"), self.DT)
        


    def _compute_matrices(self, level):
        start = get_time()
        # object is key here, as we need to have a symbolic representation of the fractions.
        self.phib = np.empty((level + 1), dtype=object)
        self.M = np.empty((level + 1, level + 1), dtype=object)
        self.A = np.empty((level + 1, level + 1, level + 1), dtype=object)
        self.B = np.empty((level + 1, level + 1, level + 1), dtype=object)
        self.D = np.empty((level + 1, level + 1), dtype=object)
        self.Dxi = np.empty((level + 1, level + 1), dtype=object)
        self.Dxi2 = np.empty((level + 1, level + 1), dtype=object)

        self.DD = np.empty((level + 1, level + 1), dtype=object)
        self.D1 = np.empty((level + 1, level + 1), dtype=object)
        self.DT = np.empty((level + 1, level + 1, level + 1), dtype=object)

        for k in range(level + 1):
            self.phib[k] = self._phib(k)
            for i in range(level + 1):
                self.M[k, i] = self._M(k, i)
                self.D[k, i] = self._D(k, i)
                self.Dxi[k, i] = self._Dxi(k, i)
                self.Dxi2[k, i] = self._Dxi2(k, i)

                self.DD[k, i] = self._DD(k, i)
                self.D1[k, i] = self._D1(k, i)
                for j in range(level + 1):
                    self.A[k, i, j] = self._A(k, i, j)
                    self.B[k, i, j] = self._B(k, i, j)
                    self.DT[k, i, j] = self._DT(k, i, j)
            

    def compute_matrices(self, level):
        failed = True
        if self.use_cache:
            failed = self.load_cached_matrices()
        if failed or (not self.use_cache):
            self._compute_matrices(level)
            self.save_cached_matrices()

    def enforce_boundary_conditions_lsq(self, rhs=np.zeros(2), dim=1):
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        AtA = A_enforce.T @ A_enforce
        reg = 10 ** (-6)
        A_enforce_inv = np.array((AtA + reg * np.eye(AtA.shape[0])).inv(), dtype=float)

        def f_1d(Q):
            for i, q in enumerate(Q.T):
                # alpha_enforce = q[I_enforce+1]
                alpha_free = q[I_free + 1]
                b = rhs - np.dot(A_free, alpha_free)
                # b = rhs
                result = np.dot(A_enforce_inv, A_enforce.T @ b)
                alpha = 1.0
                Q[I_enforce + 1, i] = (1 - alpha) * Q[I_enforce + 1, i] + (
                    alpha
                ) * result
            return Q

        def f_2d(Q):
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False

    def enforce_boundary_conditions_lsq2(self, rhs=np.zeros(2), dim=1):
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)

        def obj(alpha0, lam):
            def f(alpha):
                return np.sum((alpha - alpha0) ** 2) + lam * np.sum(
                    np.array(np.dot(A, alpha) ** 2, dtype=float)
                )

            return f

        def f_1d(Q):
            for i, q in enumerate(Q.T):
                h = q[0]
                alpha = q[1:] / h
                f = obj(alpha, 0.1)
                result = lsq(f, alpha)
                Q[1:, i] = h * result.z
            return Q

        def f_2d(Q):
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False

    def enforce_boundary_conditions(
        self, enforced_basis=[-2, -1], rhs=np.zeros(2), dim=1
    ):
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top][: len(enforced_basis)])

        # test to only constrain bottom
        # A = Matrix([constraint_bottom])
        # enforced_basis = [-1]
        # rhs=np.zeros(1)

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[enforced_basis]
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        A_enforce_inv = np.array(A_enforce.inv(), dtype=float)

        def f_1d(Q):
            for i, q in enumerate(Q.T):
                alpha_enforce = q[I_enforce + 1]
                alpha_free = q[I_free + 1]
                b = rhs - np.dot(A_free, alpha_free)
                result = np.dot(A_enforce_inv, b)
                alpha = 1.0
                Q[I_enforce + 1, i] = (1 - alpha) * Q[I_enforce + 1, i] + (
                    alpha
                ) * result
            return Q

        def f_2d(Q):
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False
            
    """ 
    Compute phi_k(@xi=0)
    """

    def _phib(self, k):
        return self.basisfunctions.eval(k, self.basisfunctions.bounds()[0])

    """ 
    Compute <phi_k, phi_i>
    """

    def _M(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * self.basisfunctions.eval(k, z) * self.basisfunctions.eval(i, z), (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1])
        )

    """ 
    Compute <phi_k, phi_i, phi_j>
    """

    def _A(self, k, i, j):
        return integrate(
            self.basisfunctions.weight(z) * self.basisfunctions.eval(k, z)
            * self.basisfunctions.eval(i, z)
            * self.basisfunctions.eval(j, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi')_k, phi_j, int(phi)_j>
    """

    def _B(self, k, i, j):
        return integrate(
            self.basisfunctions.weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * integrate(self.basisfunctions.eval(j, z), z)
            * self.basisfunctions.eval(i, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi')_k, (phi')_j>
    """

    def _D(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )
        
    """ 
    Compute <(phi')_k, (phi')_j * xi>
    """
    def _Dxi(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z) * z,
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )
        """ 
    Compute <(phi')_k, (phi')_j * xi**2>
    """
    def _Dxi2(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z) * z * z,
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi)_k, (phi')_j>
    """

    def _D1(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * self.basisfunctions.eval(k, z) * diff(self.basisfunctions.eval(i, z), z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi)_k, (phi'')_j>
    """

    def _DD(self, k, i):
        return integrate(
            self.basisfunctions.weight(z) * self.basisfunctions.eval(k, z)
            * diff(diff(self.basisfunctions.eval(i, z), z), z),
           (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 

    Compute <(phi')_k, (phi')_j>
    """

    def _DT(self, k, i, j):
        return integrate(
            self.basisfunctions.weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z)
            * self.basisfunctions.eval(j, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )


class BasisNoHOM(Basismatrices):
    def _A(self, k, i, j):
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        if (i == 0 and j == k) or (j == 0 and i == k) or (k == 0 and i == j):
            return super()._A(k, i, j)
        return 0

    def _B(self, k, i, j):
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        # if not (i==0 or j==0):
        if (i == 0 and j == k) or (j == 0 and i == k) or (k == 0 and i == j):
            return super()._B(k, i, j)
        return 0
        # return super()._B(k, i, j)
