from typing import Iterator

import numpy as np

from dataset_reader.base_reader import BaseReader, Query, Record


class NpzReader(BaseReader):
    def __init__(self, path, normalize: bool = False):
        self.data = np.load(path)
        self.normalize = normalize

    def read_queries(self) -> Iterator[Query]:
        vectors = self.data.get("test", [])
        neighbors = self.data.get("neighbors", [])
        distances = self.data.get("distances", [])

        for vector, expected_result, expected_scores in zip(
            vectors, neighbors, distances
        ):
            if self.normalize:
                vector /= np.linalg.norm(vector)
            yield Query(
                vector=vector.tolist(),
                sparse_vector=None,
                meta_conditions=None,
                expected_result=expected_result.tolist(),
                expected_scores=expected_scores.tolist(),
            )

    def read_data(self) -> Iterator[Record]:
        vectors = self.data.get("train", [])
        for idx, vector in enumerate(vectors):
            if self.normalize:
                vector /= np.linalg.norm(vector)
            yield Record(
                id=idx,
                vector=vector.tolist(),
                sparse_vector=None,
                metadata=None,
            )


if __name__ == "__main__":
    import os

    from benchmark import DATASETS_DIR

    # You need to run h5_to_npz.py first to generate this file
    test_path = os.path.join(
        DATASETS_DIR, "glove-100-angular", "glove-100-angular.npz"
    )

    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        print("Please run `python dataset_reader/h5_to_npz.py <path_to_h5_file>` first.")
    else:
        record = next(NpzReader(test_path).read_data())
        print("Record example:")
        print(record, end="\n\n")

        query = next(NpzReader(test_path).read_queries())
        print("Query example:")
        print(query)
