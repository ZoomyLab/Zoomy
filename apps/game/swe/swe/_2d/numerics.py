import jax.numpy as np

def lax_friedrichs(Ql, Qr, Fl, Fr, max_speed):
    dQ = Qr - Ql
    #I1 = np.where(Ql[3] > 0, 0, 1)
    #I2 = np.where(Qr[3] > 0, 0, 1)
    #I = I1 * I2
    dissipation = max_speed * (dQ)
    #dissipation = dissipation.at[0, :, :].multiply(I)
    #dissipation = dissipation.at[1, :, :].multiply(I)
    #dissipation = dissipation.at[2, :, :].multiply(I)
    #dissipation = dissipation.at[3, :, :].multiply(I)
    dissipation = dissipation.at[3, :, :].set(0.)
    #dissipation = dissipation.at[0, :, :].add(dQ[3, :, :])
    #dissipation[3, :, :] = 0.
    out =  0.5 * (Fl + Fr - dissipation)
    #Il = np.where(Ql[3] > 0, 0., 1.)
    #Ir = np.where(Qr[3] > 0, 0., 1.)
    #I = np.tile((Il*Ir)[np.newaxis, :, :], 4, axis=0)
    #out *= I
    return out
    

