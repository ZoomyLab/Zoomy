def constant(dt=0.1):
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        return dt

    return compute_dt


def adaptive(CFL=0.9):
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        ev_abs_max = compute_max_abs_eigenvalue(Q, Qaux, parameters)
        return (CFL * 2 * min_inradius / ev_abs_max).min()

    return compute_dt
