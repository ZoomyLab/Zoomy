from common import *
nsm = nsm_for("stoker", 1)
print("state", list(nsm.state)); print("aux", list(nsm.aux_state))
print("params", nsm.parameter_values)
print("source", nsm.source)
print("ncm", nsm.nonconservative_matrix)
print("eig", nsm.eigenvalues)
print("dt_max", nsm.dt_max)
