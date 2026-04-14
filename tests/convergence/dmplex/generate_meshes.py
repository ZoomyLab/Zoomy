"""Generate Gmsh .msh meshes for convergence studies.

Creates structured triangular meshes on [0,1]^2 with a single "wall"
boundary tag (reflective / copy BCs). PETSc reads these directly.

Usage:
    python generate_meshes.py --resolutions 10 20 40 80 --outdir meshes
"""
import argparse
import os

def generate_square_mesh_msh(nx, outpath):
    """Write a Gmsh 2.2 .msh file with nx*ny*2 triangles on [0,1]^2.

    Boundary physical tag 1 (dim=1) named "wall" on all 4 edges.
    Volume physical tag 2 (dim=2) named "surface".
    """
    ny = nx
    # Vertices
    n_verts = (nx + 1) * (ny + 1)
    # Triangles: 2 per quad cell
    n_tris = 2 * nx * ny
    # Boundary edges: 4 * nx (assuming ny == nx)
    n_bnd_edges = 2 * nx + 2 * ny

    def vid(ix, iy):
        return ix * (ny + 1) + iy + 1  # 1-based

    with open(outpath, "w") as f:
        # Header
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Physical names
        f.write("$PhysicalNames\n")
        f.write("2\n")  # 2 physical groups
        f.write('1 1 "wall"\n')
        f.write('2 2 "surface"\n')
        f.write("$EndPhysicalNames\n")

        # Nodes
        f.write("$Nodes\n")
        f.write(f"{n_verts}\n")
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                x = ix / nx
                y = iy / ny
                f.write(f"{vid(ix, iy)} {x:.15e} {y:.15e} 0.0\n")
        f.write("$EndNodes\n")

        # Elements: boundary edges (type 1, 2-node line) + triangles (type 2)
        n_elems = n_bnd_edges + n_tris
        f.write("$Elements\n")
        f.write(f"{n_elems}\n")

        eid = 1

        # Bottom edge (y=0): ix from 0 to nx-1
        for ix in range(nx):
            f.write(f"{eid} 1 2 1 1 {vid(ix, 0)} {vid(ix+1, 0)}\n")
            eid += 1
        # Top edge (y=1): ix from 0 to nx-1
        for ix in range(nx):
            f.write(f"{eid} 1 2 1 1 {vid(ix, ny)} {vid(ix+1, ny)}\n")
            eid += 1
        # Left edge (x=0): iy from 0 to ny-1
        for iy in range(ny):
            f.write(f"{eid} 1 2 1 1 {vid(0, iy)} {vid(0, iy+1)}\n")
            eid += 1
        # Right edge (x=1): iy from 0 to ny-1
        for iy in range(ny):
            f.write(f"{eid} 1 2 1 1 {vid(nx, iy)} {vid(nx, iy+1)}\n")
            eid += 1

        # Triangles (2 per quad, physical tag 2, elementary tag 2)
        for ix in range(nx):
            for iy in range(ny):
                v0 = vid(ix, iy)
                v1 = vid(ix + 1, iy)
                v2 = vid(ix + 1, iy + 1)
                v3 = vid(ix, iy + 1)
                # Lower-left triangle
                f.write(f"{eid} 2 2 2 2 {v0} {v1} {v3}\n")
                eid += 1
                # Upper-right triangle
                f.write(f"{eid} 2 2 2 2 {v1} {v2} {v3}\n")
                eid += 1

        f.write("$EndElements\n")

    print(f"  Wrote {outpath}: {n_verts} vertices, {n_tris} triangles, h={1.0/nx:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", nargs="+", type=int, default=[10, 20, 40, 80])
    parser.add_argument("--outdir", default="meshes")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for nx in args.resolutions:
        outpath = os.path.join(args.outdir, f"square_{nx}.msh")
        generate_square_mesh_msh(nx, outpath)
