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


    def integrate(self, H, HU, HUm, omega, dof_points):
        """
        We perform the midpoint rule for integration along the extrusion direction. 
        As the DG-1 element has two dof in z-direction (legendre-integration points inside the cells (at z_low, z_high), we need to compute the exact integration points. The midpoints of the fields are already the location of the dof. 
        """
        layer_shape = self.extr_reshape(H)[:, 0, :, 0].shape
        tmp_HUm = np.zeros(layer_shape, dtype=float)
        tmp_HVm = np.zeros(layer_shape, dtype=float)
        tmp_phim = np.zeros(layer_shape, dtype=float)
        tmp_psi = np.zeros(layer_shape, dtype=float)
        #V = HU.split()[0].function_space()
        V = H.function_space()
        phi_symbolic = HU.sub(0).dx(0) + HU.sub(1).dx(1)
        phi = Function(V).interpolate(phi_symbolic)
        for layer in range(self.num_layers):  # Loop through layers except the top one
            if layer == 0:
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_prev = np.zeros_like(z_low)
                z_next = self.extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]

                phi_low = self.extr_reshape(phi)[:, layer, :, 0]
                phi_high = self.extr_reshape(phi)[:, layer, :, 1]
                HU_low = self.extr_reshape(HU.sub(0))[:, layer, :, 0]
                HU_high = self.extr_reshape(HU.sub(0))[:, layer, :, 1]
                HV_low = self.extr_reshape(HU.sub(1))[:, layer, :, 0]
                HV_high = self.extr_reshape(HU.sub(1))[:, layer, :, 1]
                z_start = z_prev
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
            elif layer == self.num_layers-1:
                z_prev = self.extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_next = np.ones_like(z_low)
                phi_low = self.extr_reshape(phi)[:, layer, :, 0] 
                phi_high = self.extr_reshape(phi)[:, layer, :, 1]
                HU_low = self.extr_reshape(HU.sub(0))[:, layer, :, 0] 
                HU_high = self.extr_reshape(HU.sub(0))[:, layer, :, 1]
                HV_low = self.extr_reshape(HU.sub(1))[:, layer, :, 0] 
                HV_high = self.extr_reshape(HU.sub(1))[:, layer, :, 1]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = z_next
            else:
                z_prev = self.extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
                z_low = self.extr_reshape(dof_points.sub(2))[:, layer, :, 0]
                z_high = self.extr_reshape(dof_points.sub(2))[:, layer, :, 1]
                z_next = self.extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]
                phi_low = self.extr_reshape(phi)[:, layer, :, 0] 
                phi_high = self.extr_reshape(phi)[:, layer, :, 1] 
                HU_low = self.extr_reshape(HU.sub(0))[:, layer, :, 0] 
                HU_high = self.extr_reshape(HU.sub(0))[:, layer, :, 1]
                HV_low = self.extr_reshape(HU.sub(1))[:, layer, :, 0] 
                HV_high = self.extr_reshape(HU.sub(1))[:, layer, :, 1]
                z_start = 0.5 * (z_prev + z_low)
                z_mid = 0.5 * (z_low + z_high) 
                z_end = 0.5 * (z_high + z_next)
    
            dz_low =  z_mid - z_start
            dz_high =  z_end - z_mid
        
            
            tmp_phim +=  dz_low * phi_low + dz_high * phi_high
            tmp_HUm += dz_low * HU_low + dz_high * HU_high
            tmp_HVm += dz_low * HV_low + dz_high * HV_high
            self.extr_reshape(omega)[:, layer, :, 0] = tmp_psi + dz_low * phi_low
            tmp_psi += dz_low * phi_low + dz_high * phi_high
            self.extr_reshape(omega)[:, layer, :, 1] = tmp_psi
    
        for layer in range(self.num_layers):  # Loop through layers except the top one
            self.extr_reshape(omega)[:, layer, :, 0] =  1./self.extr_reshape(H)[:, 0, :, 0] * (tmp_phim - self.extr_reshape(omega)[:, layer, :, 0]) 
            self.extr_reshape(omega)[:, layer, :, 1] =  1./self.extr_reshape(H)[:, 0, :, 0] * (tmp_phim - self.extr_reshape(omega)[:, layer, :, 1]) 
            self.extr_reshape(HUm.sub(0))[:, layer, :, 0] = tmp_HUm
            self.extr_reshape(HUm.sub(0))[:, layer, :, 1] = tmp_HUm
            self.extr_reshape(HUm.sub(1))[:, layer, :, 0] = tmp_HVm
            self.extr_reshape(HUm.sub(1))[:, layer, :, 1] = tmp_HVm

