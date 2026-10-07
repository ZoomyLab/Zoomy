from common import *
m = SWE(dimension=2, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in ("left","right","bottom","top")]))
nsm = build_nsm(m); rt = NumpyRuntimeModel.from_system_model(nsm.sm)
p = np.array(list(nsm.sm.parameter_values.values()), float)
for nc in (5, 2):
    Q = np.ones((4, nc)); A = np.ones((1, nc))
    B = np.asarray(rt.nonconservative_matrix(Q, A, p)); print(nc, B.shape, B[2, 0])
print(np.asarray(rt.nonconservative_matrix(np.ones(4), np.ones(1), p)).shape)
m1 = SWE(dimension=1, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in ("left","right")]))
nsm1 = build_nsm(m1); rt1 = NumpyRuntimeModel.from_system_model(nsm1.sm)
print("1d", np.asarray(rt1.nonconservative_matrix(np.ones((3, 5)), np.ones((1, 5)), p)).shape, np.asarray(rt1.nonconservative_matrix(np.ones(3), np.ones(1), p)).shape)
