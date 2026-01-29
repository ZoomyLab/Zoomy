import sys
import os
import argparse
import numpy as np
import meshio
import h5py
from scipy.spatial import KDTree

# Initialize PETSc with args to suppress warnings
import petsc4py
petsc4py.init([sys.argv[0]])
from petsc4py import PETSc

def parse_mapping(mapping_str):
    mapping = {}
    if not mapping_str: return mapping
    try:
        pairs = mapping_str.split(',')
        for pair in pairs:
            if ':' in pair:
                idx, field = pair.split(':')
                mapping[int(idx.strip())] = field.strip()
    except Exception as e:
        print(f"Error parsing mapping: {e}")
        sys.exit(1)
    return mapping

def generate(input_mesh_file, output_h5_file, num_components, mapping_str):
    mapping = parse_mapping(mapping_str)
    
    if not os.path.exists(input_mesh_file):
        print(f"Error: Mesh file {input_mesh_file} not found.")
        sys.exit(1)

    print(f"Reading mesh (Source Data): {input_mesh_file}")
    
    # 1. Load Source Data (Meshio Order)
    mesh = meshio.read(input_mesh_file)
    cells = mesh.cells_dict.get("triangle")
    if cells is None:
        print("Error: No triangles found in mesh.")
        sys.exit(1)
        
    points = mesh.points[:, :2] # 2D
    
    # Calculate Source Centroids
    # Shape: (N_source, 2)
    source_centroids = points[cells].mean(axis=1)
    
    # Prepare Data Dictionary
    data_map = {}
    def get_field(name):
        if name in mesh.point_data: return mesh.point_data[name][cells].mean(axis=1)
        if name in mesh.cell_data and 'triangle' in mesh.cell_data[name]: return mesh.cell_data[name]['triangle']
        return None

    # Load fields
    for field in ['B', 'H', 'U', 'V']:
        val = get_field(field)
        if val is not None: data_map[field] = val
    
    # Derived
    if 'H' in data_map and 'U' in data_map: data_map['HU'] = data_map['H'] * data_map['U']
    if 'H' in data_map and 'V' in data_map: data_map['HV'] = data_map['H'] * data_map['V']
    data_map['ZERO'] = np.zeros(len(cells))

    # 2. Load Target Topology (PETSc Order)
    print("Loading mesh into PETSc to determine 'Natural' ordering...")
    dm = PETSc.DMPlex().createFromFile(input_mesh_file)
    dm.distribute() 
    
    # Get PETSc Cell Centroids
    cStart, cEnd = dm.getHeightStratum(0)
    num_petsc_cells = cEnd - cStart
    
    print(f"  Source Cells (Meshio): {len(cells)}")
    print(f"  Target Cells (PETSc):  {num_petsc_cells}")
    
    if len(cells) != num_petsc_cells:
        print("Error: Mismatch in cell counts! Is the mesh 2D?")
        sys.exit(1)

    petsc_centroids = []
    # We strictly iterate c from cStart to cEnd. 
    # This defines the "0, 1, 2..." order for the serial HDF5 file.
    for c in range(cStart, cEnd):
        # computeCellGeometryFVM returns (vol, centroid, normal)
        # We handle versions of petsc4py where return might differ slightly
        geo = dm.computeCellGeometryFVM(c)
        # geo[1] is usually the centroid
        petsc_centroids.append(geo[1][:2])
        
    petsc_centroids = np.array(petsc_centroids)

    # 3. Map Source -> Target using KDTree
    print("Mapping data via KDTree (matching centroids)...")
    tree = KDTree(source_centroids)
    dists, indices = tree.query(petsc_centroids)
    
    max_dist = np.max(dists)
    if max_dist > 1e-5:
        print(f"Warning: Significant centroid mismatch detected (max {max_dist}). Check mesh scaling/dim.")

    # 4. Fill Ordered Buffer
    # We write a flat array: [Cell0_Var0, Cell0_Var1, ... CellN_Var0...]
    # This matches the layout PETSc expects for a block size of 'num_components'
    output_data = np.zeros(num_petsc_cells * num_components, dtype=np.float64)
    
    for i in range(num_petsc_cells):
        source_idx = indices[i] # The meshio cell index that matches PETSc cell 'i'
        
        # Base offset for Cell 'i' in the flat array
        base_off = i * num_components
        
        for comp_idx, key in mapping.items():
            if comp_idx < num_components:
                val = data_map[key][source_idx]
                output_data[base_off + comp_idx] = val

    # 5. Write using h5py (Bypassing PETSc viewer issues)
    print(f"Writing reordered data to {output_h5_file}...")
    with h5py.File(output_h5_file, 'w') as f:
        dset = f.create_dataset("state", data=output_data)
        dset.attrs["num_cells"] = num_petsc_cells
        dset.attrs["num_components"] = num_components
    
    print("Done. Initial condition is now in PETSc Natural Order.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--num-components", "-n", type=int, default=4)
    parser.add_argument("--mapping", "-m", type=str, default="0:B, 1:H, 2:HU, 3:HV")
    args = parser.parse_args()
    
    generate(args.input, args.output, args.num_components, args.mapping)