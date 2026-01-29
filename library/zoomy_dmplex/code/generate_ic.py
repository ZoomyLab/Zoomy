import sys
import os
import argparse
import numpy as np
import meshio
import h5py

def parse_mapping(mapping_str):
    """Parses string '0:B, 1:H' into dict {0:'B', 1:'H'}"""
    mapping = {}
    if not mapping_str:
        return mapping
    try:
        pairs = mapping_str.split(',')
        for pair in pairs:
            if ':' in pair:
                idx, field = pair.split(':')
                mapping[int(idx.strip())] = field.strip()
    except Exception as e:
        print(f"Error parsing mapping string: {e}")
        sys.exit(1)
    return mapping

def generate(input_mesh_file, output_h5_file, num_components, mapping_str):
    mapping = parse_mapping(mapping_str)
    
    if not os.path.exists(input_mesh_file):
        print(f"Error: Mesh file {input_mesh_file} not found.")
        sys.exit(1)

    print(f"Reading mesh: {input_mesh_file}")
    print(f"Output Configuration: {num_components} components")
    print(f"Mapping: {mapping}")

    # 1. Read Data using Meshio
    # This reads data in the exact order it appears in the file (Natural Ordering)
    mesh = meshio.read(input_mesh_file)
    
    cells = mesh.cells_dict.get("triangle")
    if cells is None:
        print("Error: No triangles found in mesh.")
        sys.exit(1)
    
    num_cells = len(cells)
    print(f"Processing {num_cells} cells...")

    # 2. Prepare Source Data Dictionary (Cell Averaged)
    source_data = {}
    
    # Helper to safe-get field
    def get_avg_field(name):
        if name in mesh.point_data:
            # Average vertex data to cell center
            return mesh.point_data[name][cells].mean(axis=1)
        elif name in mesh.cell_data:
            # Use cell data directly (assuming it matches triangle block)
            # meshio cell_data is dict: type -> array
            if 'triangle' in mesh.cell_data[name]:
                return mesh.cell_data[name]['triangle']
        return None

    # Load Standard Fields
    b_val = get_avg_field('B')
    if b_val is not None: source_data['B'] = b_val
    
    h_val = get_avg_field('H')
    if h_val is not None: source_data['H'] = h_val
    
    u_val = get_avg_field('U')
    if u_val is not None: source_data['U'] = u_val
    
    v_val = get_avg_field('V')
    if v_val is not None: source_data['V'] = v_val
    
    # Derived Fields
    if 'H' in source_data and 'U' in source_data:
        source_data['HU'] = source_data['H'] * source_data['U']
    if 'H' in source_data and 'V' in source_data:
        source_data['HV'] = source_data['H'] * source_data['V']
        
    source_data['ZERO'] = np.zeros(num_cells)

    # Validate Mapping
    for key in mapping.values():
        if key not in source_data:
            print(f"Error: Requested field '{key}' not available/derivable.")
            print(f"Available: {list(source_data.keys())}")
            sys.exit(1)

    # 3. Interleave Data into Global Array
    # Shape: (Num_Cells * Num_Components)
    # Order: Cell0_Comp0, Cell0_Comp1, ... Cell1_Comp0 ...
    
    # Pre-allocate output array
    # PETSc HDF5 reader expects a flat 1D array of floats
    output_data = np.zeros(num_cells * num_components, dtype=np.float64)
    
    for comp_idx in range(num_components):
        if comp_idx in mapping:
            key = mapping[comp_idx]
            data = source_data[key]
            
            # Vectorized assignment: Assign every nth element
            # output_data[start::step] = values
            output_data[comp_idx::num_components] = data
            
    print("Data interleaved successfully.")

    # 4. Write to HDF5 using h5py (Bypassing PETSc complexity)
    print(f"Writing to {output_h5_file}...")
    
    with h5py.File(output_h5_file, 'w') as f:
        # Create a dataset named "state"
        # PETSc's VecLoad simply reads the dataset.
        dset = f.create_dataset("state", data=output_data)
        
        # Optional: Add attributes if useful for debugging
        dset.attrs["num_cells"] = num_cells
        dset.attrs["num_components"] = num_components
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Raw HDF5 initial condition from Gmsh mesh.")
    
    parser.add_argument("--input", "-i", required=True, help="Input mesh file (.msh)")
    parser.add_argument("--output", "-o", required=True, help="Output HDF5 file")
    
    parser.add_argument("--num-components", "-n", type=int, default=4, 
                        help="Number of components in the output vector (default: 4)")
    
    parser.add_argument("--mapping", "-m", type=str, default="0:B, 1:H, 2:HU, 3:HV", 
                        help="Mapping string 'index:field'. Available: B, H, U, V, HU, HV, ZERO.")

    args = parser.parse_args()
    
    generate(args.input, args.output, args.num_components, args.mapping)