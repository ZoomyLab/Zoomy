from firedrake import *
from firedrake.__future__ import interpolate
from mpi4py import MPI

#TODO use extr_rshape only in the beginning and at the end!!

class DepthIntegrator():

    def __init__(self, num_layers, num_dofs_per_cell, dim_space_vert):
        self.num_layers = num_layers
        self.num_dofs_per_cell = num_dofs_per_cell
        self.dim_space_vert=dim_space_vert
    
    def extr_reshape(self, field):
        return field.dat.data[:].reshape((-1, self.num_layers, self.num_dofs_per_cell, self.dim_space_vert+1))


    def integrate(self, H, HU, Hb, HUm, omega, dof_points, _mesh, rank_field, phi, dxh, dyh, dxhb, dyhb):
        """
        We perform the midpoint rule for integration along the extrusion direction. 
        As the DG-1 element has two dof in z-direction (legendre-integration points inside the cells (at z_low, z_high), we need to compute the exact integration points. The midpoints of the fields are already the location of the dof. 
        """
        #__HU = self.extr_reshape(HU.sub(0))
        #__HV = self.extr_reshape(HU.sub(1))
        #__HUm = np.zeros_like(__HU)
        #__HVm = np.zeros_like(__HV)
        #for i in range(__HU.shape[0]):
        #    for j in range(__HU.shape[2]):
        #        __HUm[i, :, j, :] = np.mean(__HU[i, :, j, :])
        #        __HVm[i, :, j, :] = np.mean(__HV[i, :, j, :])
        #HUm.sub(0).dat.data[:] = __HUm.flatten()
        #HUm.sub(1).dat.data[:] = __HVm.flatten()
        #HUm.assign(HUm)
        #_omega = self.extr_reshape(omega)
        #omega.dat.data[:] = _omega.flatten()
        #omega.assign(omega)
        #return HUm, omega


        layer_shape = self.extr_reshape(H)[:, 0, :, 0].shape
        tmp_HUm = np.zeros(layer_shape, dtype=float)
        tmp_HVm = np.zeros(layer_shape, dtype=float)
        tmp_phim = np.zeros(layer_shape, dtype=float)
        tmp_psi = np.zeros(layer_shape, dtype=float)

        h_layer = self.extr_reshape(H)[:, 0, :, 0]
        #V = HU.split()[0].function_space()
        #V = H.function_space()
        #phi_symbolic = HU.sub(0).dx(0) + HU.sub(1).dx(1)
        #phi = Function(V).interpolate(phi_symbolic)
        #dxh = self.extr_reshape(Function(V).interpolate(H.dx(0)))[:, 0, :, 0]
        #dyh = self.extr_reshape(Function(V).interpolate(H.dx(1)))[:, 0, :, 0]
        #dxhb = self.extr_reshape(Function(V).interpolate(Hb.dx(0)))[:, 0, :, 0]
        #dyhb = self.extr_reshape(Function(V).interpolate(Hb.dx(1)))[:, 0, :, 0]
        _dxh = self.extr_reshape (dxh)[:, 0, :, 0]
        _dyh = self.extr_reshape (dyh)[:, 0, :, 0]
        _dxhb = self.extr_reshape(dxhb)[:, 0, :, 0]
        _dyhb = self.extr_reshape(dyhb)[:, 0, :, 0]

        _z = self.extr_reshape(dof_points.sub(2))
        _phi = self.extr_reshape(phi)
        _HUm = self.extr_reshape(HUm.sub(0))
        _HVm = self.extr_reshape(HUm.sub(1))
        _HU =  self.extr_reshape(HU.sub(0))
        _HV =  self.extr_reshape(HU.sub(1))
        _HW = self.extr_reshape(HU.sub(2))
        _omega = self.extr_reshape(omega)

        for layer in range(self.num_layers):  # Loop through layers except the top one
            if layer == 0:
                z_low = _z[:, layer, :, 0]
                z_high = _z[:, layer, :, 1]
                z_prev = np.zeros_like(z_low)
                z_next = _z[:, layer+1, :, 0]

                phi_low = _phi[:, layer, :, 0]
                phi_high = _phi[:, layer, :, 1]
                HU_low = _HU[:, layer, :, 0]
                HU_high = _HU[:, layer, :, 1]
                HV_low = _HV[:, layer, :, 0]
                HV_high = _HV[:, layer, :, 1]
                z_start = z_prev
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
            elif layer == self.num_layers-1:
                z_prev = _z[:, layer-1, :, 1]
                z_low =  _z[:, layer, :, 0]
                z_high = _z[:, layer, :, 1]
                z_next = np.ones_like(z_low)
                phi_low =  _phi[:, layer, :, 0] 
                phi_high = _phi[:, layer, :, 1]
                HU_low =  _HU[:, layer, :, 0] 
                HU_high = _HU[:, layer, :, 1]
                HV_low =  _HV[:, layer, :, 0] 
                HV_high = _HV[:, layer, :, 1]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = z_next
            else:
                z_prev = _z[:, layer-1, :, 1]
                z_low =  _z[:, layer, :, 0]
                z_high = _z[:, layer, :, 1]
                z_next = _z[:, layer+1, :, 0]
                phi_low =  _phi[:, layer, :, 0] 
                phi_high = _phi[:, layer, :, 1] 
                HU_low =  _HU[:, layer, :, 0] 
                HU_high = _HU[:, layer, :, 1]
                HV_low =  _HV[:, layer, :, 0] 
                HV_high = _HV[:, layer, :, 1]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
    
            dz_low =  z_mid - z_start
            dz_high =  z_end - z_mid
        
            
            #_omega[:, layer, :, 0] = tmp_psi + dz_low * phi_low
            #_omega[:, layer, :, 1] = tmp_psi + dz_low * phi_low + dz_high * phi_high

            # vertical velocity (3.12)
            #NOTE: omega is currently still psi!
            #w_term1 = -(tmp_psi + dz_low * phi_low)
            #u_tilde = tmp_HUm + dz_low * HU_low
            #w_term2 = u_tilde*(z_low * dxh + dxhb)
            #v_tilde = tmp_HVm + dz_low * HV_low
            #w_term3 = v_tilde*(z_low * dyh + dyhb)
            ##_HW[:, layer, :, 0] =  h_layer * (w_term1 + w_term2  + w_term3)
            #w_term1 = -(tmp_psi + dz_low * phi_low + dz_high * phi_high)
            #u_tilde = tmp_HUm + dz_low * HU_low + dz_high * HU_high
            #w_term2 = u_tilde*(z_low * dxh + dxhb)
            #v_tilde = tmp_HVm + dz_low * HV_low + dz_high * HV_high
            #w_term3 = v_tilde*(z_low * dyh + dyhb)
            ##_HW[:, layer, :, 1] =  h_layer * (w_term1 +w_term2 + w_term3)

            #tmp_phim +=  dz_low * phi_low + dz_high * phi_high
            tmp_HUm += dz_low * HU_low + dz_high * HU_high
            tmp_HVm += dz_low * HV_low + dz_high * HV_high
            #tmp_psi += dz_low * phi_low + dz_high * phi_high
    
        for layer in range(self.num_layers):  # Loop through layers except the top one
            _HUm[:, layer, :, 0] = tmp_HUm
            _HUm[:, layer, :, 1] = tmp_HUm
            _HVm[:, layer, :, 0] = tmp_HVm
            _HVm[:, layer, :, 1] = tmp_HVm
        HUm.sub(0).dat.data[:] = _HUm.flatten()
        HUm.sub(1).dat.data[:] = _HVm.flatten()
        #HUm.assign(HUm)
        omega.dat.data[:] = _omega.flatten()
        #omega.assign(omega)
        return HUm, omega


        #omega.interpolate(0.)
        #omega.assign(omega)
        #HU.assign(HU)
        #HUm.assign(HU)

