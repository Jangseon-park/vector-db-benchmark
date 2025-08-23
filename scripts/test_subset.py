import h5py
with h5py.File("datasets/glove-25-angular/glove-25-angular.hdf5", "r") as f:
    def print_name(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("DATASET:", name, "shape:", obj.shape, "dtype:", obj.dtype)
    f.visititems(print_name)