import jax
import jax.numpy as np
import sympy

x, y = sympy.symbols('x, y')
A = sympy.Matrix([x, y])
print(A)

lA = sympy.lambdify([[x, y]], A, modules='jax')

f = lambda q: lA(q).reshape(q.shape)

q = np.linspace(1,20,20).reshape((2,10))
print(f(q).shape)

f_jit = jax.jit(f)
print(f_jit(q).shape)

def F(f):
    def _F(ql, qr):
        return 0.5*(f(ql) + f(qr)) - (qr-ql)
    return _F

F_jit = jax.jit(F(f_jit))

print(F_jit(q, q).shape)