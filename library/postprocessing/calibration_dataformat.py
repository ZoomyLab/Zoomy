import os
import numpy as np
import h5py


def read_static_data(f):
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

if __name__ == "__main__":
    filepath = 'out_4/calibration_data.hdf5'
    with h5py.File(filepath, "r") as f:
        X, parameters, n_snapshots = read_static_data(f)
        for i_snapshot in range(n_snapshots):
            h, u, Fr, E = read_fields_at_snapshot(f, i_snapshot, ['h', 'u', 'Fr', 'E'])
            print(h.shape)
            print(u.shape)
            print(Fr.shape)
            print(E.shape)
            print('---')