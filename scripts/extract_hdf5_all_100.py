import h5py
import argparse

parser = argparse.ArgumentParser(description="Extract first 100 items from all datasets in an HDF5 file.")
parser.add_argument("--input", required=True, help="Input HDF5 file path")
parser.add_argument("--output", required=True, help="Output HDF5 file path")
parser.add_argument("--count", type=int, default=100, help="Number of items to extract from each dataset (default: 100)")
args = parser.parse_args()

with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
    def copy_first_n(name, obj):
        if isinstance(obj, h5py.Dataset):
            n = min(args.count, obj.shape[0])
            data = obj[:n]
            dst.create_dataset(name, data=data)
            for k, v in obj.attrs.items():
                dst[name].attrs[k] = v
            print(f"Copied {name}: shape {data.shape}, dtype {data.dtype}")
    src.visititems(copy_first_n)
print(f"Done. Saved all datasets (up to {args.count} items each) to {args.output}")
