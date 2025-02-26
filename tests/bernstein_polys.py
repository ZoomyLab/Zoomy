from scipy.interpolate import BPoly
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

x = [0, 1]

N = 8

coef = []
basis = []
for n in range(N):
    c = [[0] for i in range(N)]
    c[n] = [1]
    coef.append(c)
    print(c)
    basis.append(BPoly(c, x))

A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        A[i, j] = integrate.quad(lambda x: basis[i](x)*basis[j](x), 0, 1)[0]

print(A)
Ainv = np.linalg.inv(A)
print(Ainv)
print(np.round(np.matmul(Ainv, A), 2))

# c1 = [[1], [0], [0]]
# c2 = [[0], [1], [0]]
# c3 = [[0], [0], [1]]
# 
# b1 = BPoly(c1, x)
# b2 = BPoly(c2, x)
# b3 = BPoly(c3, x)

#print(integrate.quad(lambda x: b1(x)*b2(x), 0, 1))
#
#fig, ax = plt.subplots()
#ax.plot(xx, b1(xx), label=('b1'))
#ax.plot(xx, b2(xx), label=('b2'))
#ax.plot(xx, b3(xx), label=('b3'))
#plt.show()


