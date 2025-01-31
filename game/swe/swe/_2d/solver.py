import jax
import jax.numpy as np

import game.swe.tc_simple_2d as tc

def wet_dry_fix(Q):
    h = Q[0]
    hu = Q[1]
    hv = Q[2]
    b = Q[3]
    h = np.where(h > tc.wet_tol, h, 0)
    hu = np.where(h > tc.wet_tol, hu, 0)
    hv = np.where(h > tc.wet_tol, hv, 0)
    Q = np.array([h, hu, hv, b])
    return Q

def step_fvm_conservative(Q):


    # aliases
    dt = tc.dt
    dx = tc.dx

    # cut off unphysical cells with h < 0
    Q = wet_dry_fix(Q)

    Qi, Qn, Qs, Qe, Qw = tc.compute_reconstruction(Q)

    def wall(Qi, Qj, n1, n2):
        Qj = Qj.at[0, :, :].set(np.where(Qj[3,:,:] > 0, Qi[0,:,:], Qj[0,:,:]))
        Qj = Qj.at[1, :, :].set(np.where(Qj[3,:,:] > 0, -n1*Qi[1,:,:], Qj[1,:,:]))
        Qj = Qj.at[2, :, :].set(np.where(Qj[3,:,:] > 0, -n2*Qi[2,:,:], Qj[2,:,:]))
        Qj = Qj.at[0, :, :].set(np.where(Qj[3,:,:] > 0, Qi[0,:,:], Qj[0,:,:]))
        #Qi = Qi.at[0, :, :].set(np.where(Qi[3,:,:] > 0, 0, Qi[0,:,:]))
        #Qi = Qi.at[1, :, :].set(np.where(Qi[3,:,:] > 0, 0, Qi[1,:,:]))
        #Qi = Qi.at[2, :, :].set(np.where(Qi[3,:,:] > 0, 0, Qi[2,:,:]))
        return Qi, Qj

    Qi, Qn = wall(Qi, Qn, 1, 1)
    Qi, Qs = wall(Qi, Qs, 1, 1)
    Qi, Qw = wall(Qi, Qw, 1, 1)
    Qi, Qe = wall(Qi, Qe, 1, 1)




        #hi = Qi[0, :, :]

        ## if neighbor is dry
        #Ij = np.where(Qj[3] > 0, -1, 1)

        #hj = Qj[0, :, :]
        #Qn = np.where(Qn[3] > 0, hi, hn)
        #Qs = np.where(Qs[3] > 0, hi, hs)

    
    #Qn = Qn.at[2, :, :].multiply(Ii)
    #Qs = Qs.at[2, :, :].multiply(Ii)
    #Qw = Qw.at[1, :, :].multiply(Iw)
    #Qe = Qe.at[1, :, :].multiply(Ie)

    #Qn = Qn.at[1, :, :].multiply(In)
    #Qs = Qs.at[1, :, :].multiply(Is)
    #Qw = Qw.at[2, :, :].multiply(Ii)
    #Qe = Qe.at[2, :, :].multiply(Ii)



    Fi, Gi = tc.compute_flux(Qi)
    Fn, Gn = tc.compute_flux(Qn)
    Fs, Gs = tc.compute_flux(Qs)
    Fw, Gw = tc.compute_flux(Qw)
    Fe, Ge = tc.compute_flux(Qe)

    # LF
    # max_speed = dx/dt
    # Rusanov
    max_speed = tc.compute_max_abs_eigenvalue(Q)
    dt = (np.max(np.array([np.min(np.array([tc.CFL *  dx / max_speed, tc.dtmax])), tc.dtmin])))
    if max_speed * dt / dx > tc.CFL * 1.001:
        print(f"CFL condition violated with value {max_speed * dt / dx}")
        assert False

    #I_w = Ii * Iw 
    #I_n = Ii * In 
    #I_s = Ii * Is 
    #I_e = Ii * Ie 

    F_west_interface = tc.compute_numerical_flux(Qw, Qi, Fw, Fi, max_speed)
    F_east_interface = tc.compute_numerical_flux(Qi, Qe, Fi, Fe, max_speed)
    F_north_interface = tc.compute_numerical_flux(Qi, Qn, Gi, Gn, max_speed)
    F_south_interface = tc.compute_numerical_flux(Qs, Qi, Gs, Gi, max_speed)

    flux_contribution = dt / dx * (F_west_interface - F_east_interface + F_south_interface - F_north_interface)

    source_contribution = dt * tc.compute_source(Qi)

    Qnew = Q[:, 1:-1, 1:-1] + flux_contribution + source_contribution

    Q = Q.at[:, 1:-1, 1:-1].set(Qnew)

    Q = wet_dry_fix(Q)

    Q = tc.apply_boundary_conditions(Q)

    return Q


def setup():

    X = tc.X

    Q = np.zeros((tc.n_elements, tc.n_dof))
    Q = tc.apply_initial_conditions(X)
    Q = tc.apply_boundary_conditions(Q)

    # step = jax.jit(step_fvm_conservative)
    step = step_fvm_conservative
    return Q, step

