import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def convert_h5_to_npz(h5_path: str, output_dir: str):
    """
    Reads datasets from an HDF5 file and saves them as a compressed NPZ file.

    The HDF5 file is expected to have the following keys:
    - 'train': Training data vectors
    - 'test': Test data vectors
    - 'neighbors': Indices of nearest neighbors for test data
    - 'distances': Distances of nearest neighbors for test data
    """
    print(f"Reading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Check for expected keys and load data
        datasets_to_save = {}
        expected_keys = ['train', 'test', 'neighbors', 'distances']
        for key in expected_keys:
            if key in f:
                data = f[key]
                # Cast float datasets to float64 as requested
                if data.dtype.kind == 'f':
                    datasets_to_save[key] = np.array(data, dtype=np.float64)
                else:
                    datasets_to_save[key] = np.array(data)
                print(f"  - Found and loaded '{key}' dataset (shape: {datasets_to_save[key].shape}, dtype: {datasets_to_save[key].dtype}).")
            else:
                print(f"  - Warning: '{key}' dataset not found in the HDF5 file.")


    if not datasets_to_save:
        print("Error: No valid datasets found in the HDF5 file. Aborting conversion.")
        return

    output_filename = f"{Path(h5_path).stem}.npz"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving data to {output_path}...")
    np.savez_compressed(output_path, **datasets_to_save)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HDF5 file (from ann-benchmarks) to NPZ format."
    )
    parser.add_argument(
        "h5_file",
        type=str,
        help="Path to the input HDF5 file."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output NPZ file. Defaults to the current directory."
    )
    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"Error: Input file not found at {args.h5_file}")
    else:
        if not os.path.exists(args.output_dir):
            print(f"Output directory {args.output_dir} does not exist. Creating it...")
            os.makedirs(args.output_dir)
        convert_h5_to_npz(args.h5_file, args.output_dir)
