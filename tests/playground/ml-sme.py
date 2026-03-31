import sympy as sp

def run_comparisons():
    # 1. Define independent variables and geometry
    t, x, z = sp.symbols('t x z', real=True)
    b = sp.Function('b')(x)  # Bed elevation
    H = sp.Function('H')(x)  # Total depth (Independent Symbol)
    zeta = (z - b) / H       # Global sigma coordinate in [0, 1]

    def legendre_01(j, xi):
        if j == 0: return sp.S(1)
        elif j == 1: return 2*xi - 1
        elif j == 2: return 6*xi**2 - 6*xi + 1
        else: raise ValueError("Degree not implemented")

    def analyze_case(name, N_layers, max_degree):
        print(f"\n{'='*60}")
        print(f"CASE: {name}")
        print(f"({N_layers} Layers, Polynomial Degree {max_degree})")
        print(f"{'='*60}")
        
        sigmas = [sp.Rational(i, N_layers) for i in range(N_layers + 1)] 
        
        u_coeffs = {}
        for k in range(N_layers):
            for j in range(max_degree + 1):
                u_coeffs[(k, j)] = sp.Function(f'u_{k}_{j}')(t, x)

        u_expr = sp.S(0)
        for k in range(N_layers):
            zeta_local = (zeta - sigmas[k]) / (sigmas[k+1] - sigmas[k])
            layer_u = sum(u_coeffs[(k, j)] * legendre_01(j, zeta_local) for j in range(max_degree + 1))
            window = sp.Heaviside(zeta - sigmas[k]) - sp.Heaviside(zeta - sigmas[k+1])
            u_expr += layer_u * window

        # Custom Integration Engine
        def custom_integrate(expr, var, lower, upper, z_to_zeta_map):
            interface_fluxes = {}  
            zeta_sym = sp.Symbol('zeta_sym', real=True)
            
            if not expr.has(sp.Heaviside) and not expr.has(sp.DiracDelta):
                return sp.simplify(sp.integrate(expr, (var, lower, upper))), interface_fluxes

            total_integral = sp.S(0)
            
            for term in sp.Add.make_args(expr.expand()):
                if term.has(sp.DiracDelta):
                    delta_args = [arg for arg in term.args if isinstance(arg, sp.DiracDelta)]
                    if delta_args:
                        delta_func = delta_args[0]
                        delta_arg = delta_func.args[0] 
                        
                        # Find the physical depth of the interface (z_k)
                        z_k = sp.solve(delta_arg, var)[0]
                        g_prime = sp.diff(delta_arg, var)
                        f_z = term / delta_func
                        
                        # Evaluate the flux jump strictly at the interface depth
                        evaluated_flux = sp.simplify((f_z / g_prime).subs(var, z_k))
                        
                        z_k_clean = sp.simplify(z_k)
                        if z_k_clean not in interface_fluxes:
                            interface_fluxes[z_k_clean] = sp.S(0)
                        interface_fluxes[z_k_clean] += evaluated_flux
                else:
                    for k in range(N_layers):
                        midpoint = (sigmas[k] + sigmas[k+1]) / 2
                        subs_dict = {}
                        for h_func in term.atoms(sp.Heaviside):
                            mapped_arg = h_func.args[0].subs(z_to_zeta_map)
                            mapped_arg = sp.cancel(mapped_arg)  # Safely cancel H/H
                            arg_val = mapped_arg.subs(zeta_sym, midpoint)
                            subs_dict[h_func] = 1 if arg_val > 0 else 0
                        
                        smooth_term = term.subs(subs_dict)
                        z_lower = b + sigmas[k] * H
                        z_upper = b + sigmas[k+1] * H
                        total_integral += sp.integrate(smooth_term, (var, z_lower, z_upper))
                        
            for z_k in interface_fluxes:
                interface_fluxes[z_k] = sp.simplify(interface_fluxes[z_k])
                
            return sp.simplify(total_integral), interface_fluxes

        # Execute
        zeta_sym = sp.Symbol('zeta_sym', real=True)
        z_map = {z: b + zeta_sym * H}

        print("\n1. Integral of u dz (Total Volume / Transport):")
        int_u, _ = custom_integrate(u_expr, z, b, b + H, z_map)
        print(f"  {int_u}")
        
        print("\n2. Jumps from \\partial_z u dz (Interface Conditions):")
        dz_u = sp.diff(u_expr, z)
        _, fluxes_dz_u = custom_integrate(dz_u, z, b, b + H, z_map)
        
        # Sort interfaces from bottom to top for printing
        sorted_interfaces = sorted(fluxes_dz_u.keys(), key=lambda k: k.subs({b: 0, H: 1}))
        for z_k in sorted_interfaces:
            print(f"  Jump at z = {z_k}: \n      {fluxes_dz_u[z_k]}")

    # Run the three cases
    analyze_case("1. ML-SWE (3 constant layers)", N_layers=3, max_degree=0)
    analyze_case("2. SME / VAM (1 layer, 2nd order polynomial)", N_layers=1, max_degree=2)
    analyze_case("3. Hybrid ML-SME (2 layers, 1st order polynomials)", N_layers=2, max_degree=1)

run_comparisons()
