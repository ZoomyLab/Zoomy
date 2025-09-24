import os
import shutil

# Path to your OpenFOAM case directory
case_dir = "./"

# Loop through subfolders
for entry in os.listdir(case_dir):
    full_path = os.path.join(case_dir, entry)

    # Only consider directories starting with a digit (0, 1, 2, ...)
    if os.path.isdir(full_path) and entry[0].isdigit():
        alpha_file = os.path.join(full_path, "alpha.water")

        if not os.path.isfile(alpha_file):
            print(f"Deleting folder: {full_path} (no alpha.water found)")
            shutil.rmtree(full_path)
        else:
            print(f"Keeping folder: {full_path}")

