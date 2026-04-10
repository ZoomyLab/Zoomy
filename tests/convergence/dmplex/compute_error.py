"""Compute L1 and Linf errors for the DMPlex scalar advection convergence test.

Usage:
    python compute_error.py <vtu_path> <mesh_path> <t_end> <ax> <ay> <sigma> <kappa> <nx>
"""
import sys
import os
import numpy as np
import meshio


def compute_error(vtu_path, mesh_path, t_end, ax, ay, sigma, kappa, nx):
    """Compute L1 and Linf errors against exact solution."""
    # Read numerical solution from VTU
    vtu = meshio.read(vtu_path)
    tris_vtu = vtu.cells_dict.get("triangle")
    pts_vtu = vtu.points[:, :2]
    centroids = pts_vtu[tris_vtu].mean(axis=1)

    # Extract solution -- PETSc names it "StateField_0" for FV data
    u_num = None
    for key in ["StateField_0", "State", "q0"]:
        if key in vtu.cell_data:
            data = vtu.cell_data[key][0]
            u_num = data[:, 0] if data.ndim > 1 else data
            break
    if u_num is None:
        # Try first available cell data
        for key, vals in vtu.cell_data.items():
            if key == "Rank":
                continue
            data = vals[0]
            u_num = data[:, 0] if data.ndim > 1 else data
            break

    if u_num is None:
        print("ERROR: Could not find solution data in VTU")
        return None

    n_cells = len(u_num)

    # Exact solution at t_end
    # Pure advection: u(x,y,t) = u0(x - ax*t, y - ay*t)
    #   = exp(-((x-0.25-ax*t)^2 + (y-0.25-ay*t)^2) / (2*sigma^2))
    # Advection-diffusion: Gaussian spreads
    #   sigma_t = sqrt(sigma^2 + 2*kappa*t)
    #   amplitude = sigma^2 / sigma_t^2
    #   u(x,y,t) = amp * exp(-((x-xc)^2+(y-yc)^2) / (2*sigma_t^2))
    if kappa > 0:
        sigma_t = np.sqrt(sigma ** 2 + 2 * kappa * t_end)
        amp = sigma ** 2 / sigma_t ** 2
    else:
        sigma_t = sigma
        amp = 1.0

    x_c = 0.25 + ax * t_end
    y_c = 0.25 + ay * t_end
    dx = centroids[:, 0] - x_c
    dy = centroids[:, 1] - y_c
    u_exact = amp * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma_t ** 2))

    # Compute errors with proper cell area
    h = 1.0 / nx
    cell_area = h * h / 2.0  # Structured mesh: each triangle has area h^2/2
    error = np.abs(u_num - u_exact)
    L1 = np.sum(error * cell_area)
    Linf = np.max(error)

    return h, L1, Linf, n_cells


if __name__ == "__main__":
    vtu_path = sys.argv[1]
    mesh_path = sys.argv[2]
    t_end = float(sys.argv[3])
    ax = float(sys.argv[4])
    ay = float(sys.argv[5])
    sigma = float(sys.argv[6])
    kappa = float(sys.argv[7])
    nx = int(sys.argv[8])

    result = compute_error(vtu_path, mesh_path, t_end, ax, ay, sigma, kappa, nx)
    if result is None:
        sys.exit(1)

    h, L1, Linf, n_cells = result
    print(f"h={h:.4f}  L1={L1:.6e}  Linf={Linf:.6e}  n_cells={n_cells}")

    # Write to file
    out_dir = os.path.dirname(vtu_path)
    with open(os.path.join(out_dir, "error.txt"), "w") as f:
        f.write(f"{h} {L1} {Linf}\n")
