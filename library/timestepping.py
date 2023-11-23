
def constant(dt=0.1):
    def compute_dt(ev_abs_max=0., min_incircle=0.):
        return dt
    return compute_dt

def adaptive(CFL=0.9):
    def compute_dt(ev_abs_max=0., min_incircle=0.):
        assert ev_abs_max != 0.
        assert min_incircle != 0.
        return CFL * min_inricle / ev_abs_max
    return compute_dt