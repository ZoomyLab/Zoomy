import os
import numpy as np
import h5py


def read_common_data(f):
    coordinates = f['mesh']['centers'][()]
    parameters = {key: value[()] for key, value in f['parameters'].items()}
    n_snapshots = len(list(f['timeseries'].keys()))

    return coordinates, parameters, n_snapshots

def read_fields_at_snapshot(f, i_snapshot, fields):
    if type(fields) == str:
        fields = [fields]

    out = []
    for field_name in fields:
        out.append(f['timeseries'][str(i_snapshot)][field_name][()])
    return out


def find_all_folders_starting_with(path, startswith):
    return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(startswith)]

if __name__ == "__main__":

    folders = find_all_folders_starting_with('./', 'out_')
    filename = 'calibration_data.hdf5'

    # test with a single file
    filepath = os.path.join(folders[0], filename)
    with h5py.File(filepath, "r") as f:
        X, parameters, n_snapshots = read_common_data(f)
        for i_snapshot in range(n_snapshots):
            h, u, Fr, E = read_fields_at_snapshot(f, i_snapshot, ['h', 'u', 'Fr', 'E'])
            print(h.shape)
            print(u.shape)
            print(Fr.shape)
            print(E.shape)
            print('---')