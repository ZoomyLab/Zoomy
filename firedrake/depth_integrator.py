from firedrake import *
from firedrake.__future__ import interpolate

class DepthIntegrator():

    def __init__(self, num_layers, num_cells_base, num_dofs_per_cell, dim_space_vert):
        self.num_layers = num_layers
        self.num_cells_base = num_cells_base
        self.num_dofs_per_cell = num_dofs_per_cell
        self.num_cells_extruded = num_cells_base * num_layers
        self.dim_space_vert=dim_space_vert
    
    def extr_reshape(self, field):
        return field.dat.data[:].reshape((-1, self.num_layers, self.num_dofs_per_cell, self.dim_space_vert+1))


    def integrate(self, h, U, hphi, omega, hh, Um, phim, dof_points):
        """
        We perform the midpoint rule for integration along the extrusion direction. 
        As the DG-1 element has two dof in z-direction (legendre-integration points inside the cells (at z_low, z_high), we need to compute the exact integration points. The midpoints of the fields are already the location of the dof. 
        """
        for layer in range(self.num_layers):  # Loop through layers except the top one
            if layer == 0:
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_prev = np.zeros_like(z_low)
                z_next = self.extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]
                h_re = self.base_reshape(hh)
                phi_low = self.extr_reshape(hphi)[:, layer, :, 0] / h_re
                phi_high = self.extr_reshape(hphi)[:, layer, :, 1] / h_re
                phi_pre = np.zeros_like(phi_low)
                psi_pre = np.zeros_like(phi_low)
                u_low = self.extr_reshape(U.sub(0))[:, layer, :, 0]
                u_high = self.extr_reshape(U.sub(0))[:, layer, :, 1]
                u_pre = np.zeros_like(u_low)
                v_low = self.extr_reshape(U.sub(1))[:, layer, :, 0]
                v_high = self.extr_reshape(U.sub(1))[:, layer, :, 1]
                v_pre = np.zeros_like(u_low)
                z_start = z_prev
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
            elif layer == self.num_layers-1:
                z_prev = self.extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_next = np.ones_like(z_low)
                h_re = self.base_reshape(hh)
                phi_low = self.extr_reshape(hphi)[:, layer, :, 0] / h_re
                phi_high = self.extr_reshape(hphi)[:, layer, :, 1] / h_re
                phi_pre = self.base_reshape(phim)[:]
                psi_pre = self.extr_reshape(omega)[:, layer-1, :, 1]
                u_low = self.extr_reshape(U.sub(0))[:, layer, :, 0] 
                u_high = self.extr_reshape(U.sub(0))[:, layer, :, 1]
                u_pre = self.base_reshape(Um.sub(0))[:]
                v_low = self.extr_reshape(U.sub(1))[:, layer, :, 0] 
                v_high = self.extr_reshape(U.sub(1))[:, layer, :, 1]
                v_pre = self.base_reshape(Um.sub(1))[:]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = z_next
            else:
                z_prev = self.extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_next = self.extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]
                h_re = self.base_reshape(hh)
                phi_low = self.extr_reshape(hphi)[:, layer, :, 0] / h_re
                phi_high = self.extr_reshape(hphi)[:, layer, :, 1] / h_re
                phi_pre = self.base_reshape(phim)[:]
                psi_pre = self.extr_reshape(omega)[:, layer-1, :, 1]
                u_low = self.extr_reshape(U.sub(0))[:, layer, :, 0] 
                u_high = self.extr_reshape(U.sub(0))[:, layer, :, 1]
                u_pre = self.base_reshape(Um.sub(0))[:]
                v_low = self.extr_reshape(U.sub(1))[:, layer, :, 0] 
                v_high = self.extr_reshape(U.sub(1))[:, layer, :, 1]
                v_pre = self.base_reshape(Um.sub(1))[:]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
    
            dz_low =  z_mid - z_start
            dz_high =  z_end - z_mid
        
            
            self.base_reshape(phim)[:] = phi_pre + dz_low * phi_low + dz_high * phi_high
            self.base_reshape(Um.sub(0))[:] = u_pre + dz_low * u_low + dz_high * u_high
            self.base_reshape(Um.sub(1))[:] = v_pre + dz_low * v_low + dz_high * v_high
            self.extr_reshape(omega)[:, layer, :, 0] = psi_pre + dz_low * phi_low
            self.extr_reshape(omega)[:, layer, :, 1] = psi_pre + dz_low * phi_low + dz_high * phi_high
            h_reshaped = self.base_reshape(hh)[:]
            self.extr_reshape(h)[:, layer, :, 0] =  h_reshaped
            self.extr_reshape(h)[:, layer, :, 1] = h_reshaped
    
        for layer in range(self.num_layers):  # Loop through layers except the top one
            self.extr_reshape(omega)[:, layer, :, 0] = 1./h_reshaped * (self.base_reshape(phim)[:] - self.extr_reshape(omega)[:, layer, :, 0])
            self.extr_reshape(omega)[:, layer, :, 1] = 1./h_reshaped * (self.base_reshape(phim)[:] - self.extr_reshape(omega)[:, layer, :, 1])

        for layer in range(self.num_layers):  # Loop through layers except the top one
            self.extr_reshape(omega)[:, layer, :, 0] = 0.
            self.extr_reshape(omega)[:, layer, :, 1] = 0.
