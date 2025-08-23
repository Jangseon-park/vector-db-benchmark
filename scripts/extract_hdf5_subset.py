import h5py
import argparse

parser = argparse.ArgumentParser(description="Extract a subset of an HDF5 dataset.")
parser.add_argument("--input", required=True, help="Input HDF5 file path")
parser.add_argument("--output", required=True, help="Output HDF5 file path")
parser.add_argument("--dataset", default="embeddings", help="Dataset name in HDF5 file (default: embeddings)")
parser.add_argument("--count", type=int, default=10000, help="Number of items to extract (default: 10000)")
args = parser.parse_args()

with h5py.File(args.input, "r") as src:
    data = src[args.dataset][:args.count]
    # Copy other datasets/attributes if needed
    with h5py.File(args.output, "w") as dst:
        dst.create_dataset(args.dataset, data=data)
        # Optionally copy attributes
        for k, v in src[args.dataset].attrs.items():
            dst[args.dataset].attrs[k] = v
print(f"Extracted {data.shape[0]} items from '{args.input}' to '{args.output}' (dataset: {args.dataset})")

with h5py.File("datasets/glove-25-angular/glove-25-angular.hdf5", "r") as f:
    def print_name(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("DATASET:", name, "shape:", obj.shape, "dtype:", obj.dtype)
    f.visititems(print_name)