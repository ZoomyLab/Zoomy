import os
import numpy as np
import h5py


def read_common_data(open_file):
    coordinates = open_file["mesh"]["centers"][()]
    parameters = {key: value[()] for key, value in open_file["parameters"].items()}
    n_snapshots = len(list(open_file["timeseries"].keys()))

    return coordinates, parameters, n_snapshots


def read_fields_at_snapshot(open_file, i_snapshot, fields):
    if type(fields) == str:
        fields = [fields]

    out = []
    for field_name in fields:
        out.append(open_file["timeseries"][str(i_snapshot)][field_name][()])
    return out


def find_all_folders_starting_with(path, startswith):
    return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(startswith)]


if __name__ == "__main__":
    folders = find_all_folders_starting_with("./", "out_")
    filename = "calibration_data.hdf5"

    # test with a single file
    filepath = os.path.join(folders[0], filename)
    with h5py.File(filepath, "r") as f:
        X, parameters, n_snapshots = read_common_data(f)
        # for i_snapshot in range(n_snapshots):
        #     h, u, Fr, E = read_fields_at_snapshot(f, i_snapshot, ['h', 'u', 'Fr', 'E'])
        # print(h.shape)
        # print(u.shape)
        # print(Fr.shape)
        # print(E.shape)
        # print('---')
        Fr_max_last_timestep = read_fields_at_snapshot(f, n_snapshots - 1, "Fr")[
            0
        ].max()
        print(parameters["nm"])
        print(Fr_max_last_timestep)
