from library.model.models.shallow_moments import (IC, Matrix, Model, eigenvalue_dict_to_matrix,
                                                  register_sympy_attribute, sympy)


class ShallowMomentsSediment(Model):
    """
    Shallow Moments Sediment 1d

    :gui:
    - tab: model
    - requires: [ 'mesh.dimension': 1 ]

    """

    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        aux_initial_conditions=IC.Constant(),
        dimension=1,
        fields=2,
        aux_fields=0,
        parameters={},
        parameters_default={"g": 1.0, "ex": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis(),
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = self.n_fields - 2
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            aux_initial_conditions=aux_initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        flux = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        flux[0] = ha[0]
        flux[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels + 1):
            for i in range(self.levels + 1):
                for j in range(self.levels + 1):
                    flux[k + 1] += (
                        ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[k, k]
                    )
        return [flux]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        um = ha[0] / h
        # nc[1, 0] = - p.g * p.ez * h
        for k in range(1, self.levels + 1):
            nc[k + 1, k + 1] += um
        for k in range(self.levels + 1):
            for i in range(1, self.levels + 1):
                for j in range(1, self.levels + 1):
                    nc[k + 1, i + 1] -= (
                        ha[j] / h * self.basis.B[k, i, j] / self.basis.M[k, k]
                    )
        return [nc]

    def eigenvalues(self):
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2:]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        return out

    def inclined_plane(self):
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        out[1] = h * p.g * (p.ex)
        return out

    def newtonian(self):
        """
        :gui:
            - requires_parameter: ('nu', 0.0)
        """
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        for k in range(1 + self.levels):
            for i in range(1 + self.levels):
                out[1 + k] += (
                    -p.nu / h * ha[i] / h * self.basis.D[i, k] / self.basis.M[k, k]
                )
        return out

    def slip(self):
        """
        :gui:
            - requires_parameter: ('lamda', 0.0)
            - requires_parameter: ('rho', 1.0)
        """
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        for k in range(1 + self.levels):
            for i in range(1 + self.levels):
                out[1 + k] += -1.0 / p.lamda / p.rho * ha[i] / h / self.basis.M[k, k]
        return out

    def chezy(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.levels):
            for j in range(1 + self.levels):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.levels):
            for l in range(1 + self.levels):
                out[1 + k] += -1.0 / (p.C**2 * self.basis.M[k, k]) * ha[l] * sqrt / h
        return out

    def chezy_ssf(self):
        """
        :gui:
            - requires_parameter: ('C', 1000.0)
        """
        assert "Cf" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.levels):
            for j in range(1 + self.levels):
                tmp += ha[i] * ha[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.levels):
            for l in range(1 + self.levels):
                out[1 + k] += -(p.Cf * self.basis.M[k, k]) * ha[l] * sqrt / h
        return out


class ShallowMomentsSediment2d(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters={},
        parameters_default={"g": 1.0, "ex": 0.0, "ey": 0.0, "ez": 1.0},
        settings={},
        settings_default={"topography": False, "friction": []},
        basis=Basis(),
    ):
        self.basis = basis
        self.variables = register_sympy_attribute(fields, "q")
        self.n_fields = self.variables.length()
        self.levels = int((self.n_fields - 1) / 2) - 1
        self.basis.compute_matrices(self.levels)
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_fields=aux_fields,
            parameters=parameters,
            parameters_default=parameters_default,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def flux(self):
        offset = self.levels + 1
        flux_x = Matrix([0 for i in range(self.n_fields)])
        flux_y = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        hb = self.variables[1 + self.levels + 1 : 1 + 2 * (self.levels + 1)]
        p = self.parameters
        flux_x[0] = ha[0]
        flux_x[1] = p.g * p.ez * h * h / 2
        for k in range(self.levels + 1):
            for i in range(self.levels + 1):
                for j in range(self.levels + 1):
                    # TODO avoid devision by zero
                    flux_x[k + 1] += (
                        ha[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[k, k]
                    )
        for k in range(self.levels + 1):
            for i in range(self.levels + 1):
                for j in range(self.levels + 1):
                    # TODO avoid devision by zero
                    flux_x[k + 1 + offset] += (
                        hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[k, k]
                    )

        flux_y[0] = hb[0]
        flux_y[1 + offset] = p.g * p.ez * h * h / 2
        for k in range(self.levels + 1):
            for i in range(self.levels + 1):
                for j in range(self.levels + 1):
                    # TODO avoid devision by zero
                    flux_y[k + 1] += (
                        hb[i] * ha[j] / h * self.basis.A[k, i, j] / self.basis.M[k, k]
                    )
        for k in range(self.levels + 1):
            for i in range(self.levels + 1):
                for j in range(self.levels + 1):
                    # TODO avoid devision by zero
                    flux_y[k + 1 + offset] += (
                        hb[i] * hb[j] / h * self.basis.A[k, i, j] / self.basis.M[k, k]
                    )
        return [flux_x, flux_y]

    def nonconservative_matrix(self):
        offset = self.levels + 1
        nc_x = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        nc_y = Matrix([[0 for i in range(self.n_fields)] for j in range(self.n_fields)])
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        hb = self.variables[1 + offset : 1 + offset + self.levels + 1]
        p = self.parameters
        um = ha[0] / h
        vm = hb[0] / h
        for k in range(1, self.levels + 1):
            nc_x[k + 1, k + 1] += um
            nc_y[k + 1, k + 1 + offset] += um
        for k in range(self.levels + 1):
            for i in range(1, self.levels + 1):
                for j in range(1, self.levels + 1):
                    nc_x[k + 1, i + 1] -= (
                        ha[j] / h * self.basis.B[k, i, j] / self.basis.M[k, k]
                    )
                    nc_y[k + 1, i + 1 + offset] -= (
                        ha[j] / h * self.basis.B[k, i, j] / self.basis.M[k, k]
                    )

        for k in range(1, self.levels + 1):
            nc_x[k + 1 + offset, k + 1] += vm
            nc_y[k + 1 + offset, k + 1 + offset] += vm
        for k in range(self.levels + 1):
            for i in range(1, self.levels + 1):
                for j in range(1, self.levels + 1):
                    nc_x[k + 1 + offset, i + 1] -= (
                        hb[j] / h * self.basis.B[k, i, j] / self.basis.M[k, k]
                    )
                    nc_y[k + 1 + offset, i + 1 + offset] -= (
                        hb[j] / h * self.basis.B[k, i, j] / self.basis.M[k, k]
                    )
        return [nc_x, nc_y]

    def eigenvalues(self):
        # we delete heigher order moments (level >= 2) for analytical eigenvalues
        offset = self.levels + 1
        A = self.sympy_normal[0] * self.sympy_quasilinear_matrix[0]
        for d in range(1, self.dimension):
            A += self.sympy_normal[d] * self.sympy_quasilinear_matrix[d]
        alpha_erase = self.variables[2 : 2 + self.levels]
        beta_erase = self.variables[2 + offset : 2 + offset + self.levels]
        for alpha_i in alpha_erase:
            A = A.subs(alpha_i, 0)
        for beta_i in beta_erase:
            A = A.subs(beta_i, 0)
        return eigenvalue_dict_to_matrix(A.eigenvals())

    def source(self):
        out = Matrix([0 for i in range(self.n_fields)])
        if self.settings.topography:
            out += self.topography()
        if self.settings.friction:
            for friction_model in self.settings.friction:
                out += getattr(self, friction_model)()
        return out

    def topography(self):
        assert "dhdx" in vars(self.aux_variables)
        assert "dhdy" in vars(self.aux_variables)
        offset = self.levels + 1
        out = Matrix([0 for i in range(self.n_fields)])
        h = self.variables[0]
        p = self.parameters
        dhdx = self.aux_variables.dhdx
        dhdy = self.aux_variables.dhdy
        out[1] = h * p.g * (p.ex - p.ez * dhdx)
        out[1 + offset] = h * p.g * (p.ey - p.ez * dhdy)
        return out

    def newtonian(self):
        assert "nu" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        hb = self.variables[1 + offset : 1 + self.levels + 1 + offset]
        p = self.parameters
        for k in range(1 + self.levels):
            for i in range(1 + self.levels):
                out[1 + k] += (
                    -p.nu / h * ha[i] / h * self.basis.D[i, k] / self.basis.M[k, k]
                )
                out[1 + k + offset] += (
                    -p.nu / h * hb[i] / h * self.basis.D[i, k] / self.basis.M[k, k]
                )
        return out

    def slip(self):
        assert "lamda" in vars(self.parameters)
        assert "rho" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels + 1
        h = self.variables[0]
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        hb = self.variables[1 + offset : 1 + self.levels + 1 + offset]
        p = self.parameters
        for k in range(1 + self.levels):
            for i in range(1 + self.levels):
                out[1 + k] += -1.0 / p.lamda / p.rho * ha[i] / h / self.basis.M[k, k]
                out[1 + k + offset] += (
                    -1.0 / p.lamda / p.rho * hb[i] / h / self.basis.M[k, k]
                )
        return out

    def chezy(self):
        assert "C" in vars(self.parameters)
        out = Matrix([0 for i in range(self.n_fields)])
        offset = self.levels + 1
        h = self.variables[0]
        ha = self.variables[1 : 1 + self.levels + 1]
        hb = self.variables[1 + offset : 1 + self.levels + 1 + offset]
        p = self.parameters
        tmp = 0
        for i in range(1 + self.levels):
            for j in range(1 + self.levels):
                tmp += ha[i] * ha[j] / h / h + hb[i] * hb[j] / h / h
        sqrt = sympy.sqrt(tmp)
        for k in range(1 + self.levels):
            for l in range(1 + self.levels):
                out[1 + k] += -1.0 / (p.C**2 * self.basis.M[k, k]) * ha[l] * sqrt / h
                out[1 + k + offset] += (
                    -1.0 / (p.C**2 * self.basis.M[k, k]) * hb[l] * sqrt / h
                )
        return out
