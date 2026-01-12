#!/usr/bin/env python3
"""
Create multiple "dimension variants" from an ann-benchmarks style HDF5 dataset.

This repo's `AnnH5Reader` expects root-level datasets:
  - train: float32/float64 vectors, shape (N, D)
  - test: float32/float64 vectors, shape (Q, D)
  - neighbors: int indices, shape (Q, K)
  - distances: float distances, shape (Q, K)

For each requested dimension `d`, we create a new HDF5 file where:
  - train/test are transformed to `d` dimensions (method: slice/rp/pca)
  - optionally recompute ground-truth neighbors/distances for the new space

The copy is streaming/chunked over rows so it won't load the whole dataset into RAM.

Notes about ground-truth:
- For ann-benchmarks "angular/cosine" datasets, the stored `distances` are cosine distance:
    distance = 1 - cosine_similarity(a, b)
  (confirmed for glove-100-angular in this repo).
- If you change vectors (even by truncation), the old `neighbors/distances` no longer match.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

import h5py

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from sklearn.decomposition import IncrementalPCA  # type: ignore
except Exception:  # pragma: no cover
    IncrementalPCA = None


@dataclass(frozen=True)
class NamingPlan:
    output_dirname: str
    output_filename: str


def _parse_ann_benchmarks_name(name: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse names like 'glove-100-angular' -> ('glove', 100, 'angular')
    More generally: <prefix>-<dim>-<suffix...>
    """
    parts = name.split("-")
    for i, p in enumerate(parts):
        if p.isdigit():
            prefix = "-".join(parts[:i])
            dim = int(p)
            suffix = "-".join(parts[i + 1 :])
            if prefix and suffix:
                return prefix, dim, suffix
            return None
    return None


def _compute_output_paths(
    input_path: Path,
    dim: int,
    output_root: Path,
    naming: str,
    output_pattern: str,
) -> Path:
    """
    naming:
      - auto: try ann-benchmarks naming first; fallback to flat pattern
      - ann: enforce ann-benchmarks naming
      - flat: always output_root/<pattern>.hdf5 (pattern supports {stem} and {dim})
    """
    if naming not in {"auto", "ann", "flat"}:
        raise ValueError(f"Unknown naming: {naming}")

    parent_name = input_path.parent.name
    stem = input_path.stem

    ann_parent = _parse_ann_benchmarks_name(parent_name)
    ann_stem = _parse_ann_benchmarks_name(stem)

    def ann_variant(parsed: Tuple[str, int, str]) -> NamingPlan:
        prefix, _, suffix = parsed
        variant = f"{prefix}-{dim}-{suffix}"
        return NamingPlan(output_dirname=variant, output_filename=f"{variant}.hdf5")

    if naming in {"auto", "ann"}:
        # Prefer filename-based parsing over directory-based parsing.
        # This avoids surprises when directories are misnamed (e.g. gist-960-angular/ containing gist-960-euclidean.hdf5).
        parsed = ann_stem or ann_parent
        if parsed is None:
            if naming == "ann":
                raise ValueError(
                    "Cannot parse ann-benchmarks-style name from input path. "
                    "Use --naming flat or provide ann-benchmarks style names like glove-100-angular."
                )
        else:
            plan = ann_variant(parsed)
            return output_root / plan.output_dirname / plan.output_filename

    # flat fallback
    filename = output_pattern.format(stem=stem, dim=dim)
    if not filename.endswith(".hdf5"):
        filename += ".hdf5"
    return output_root / filename


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for k in src.keys():
        dst[k] = src[k]


def _create_like_vectors(
    dst: h5py.File,
    name: str,
    src_ds: h5py.Dataset,
    new_dim: int,
    *,
    dtype: np.dtype,
    compression: Optional[str],
) -> h5py.Dataset:
    if len(src_ds.shape) != 2:
        raise ValueError(f"Expected 2D dataset for {name}, got shape {src_ds.shape}")
    n, d = src_ds.shape
    if new_dim > d:
        raise ValueError(
            f"Requested dim {new_dim} exceeds source dim {d} for dataset '{name}'"
        )

    chunks = None
    if src_ds.chunks is not None:
        c0, c1 = src_ds.chunks
        chunks = (c0, min(c1, new_dim))

    if compression is None:
        compression = src_ds.compression

    return dst.create_dataset(
        name,
        shape=(n, new_dim),
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=src_ds.compression_opts if compression == src_ds.compression else None,
        shuffle=src_ds.shuffle,
        fletcher32=src_ds.fletcher32,
        scaleoffset=src_ds.scaleoffset,
    )


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


NormalizeMode = Literal["none", "before", "after", "both"]


def _apply_normalize(mode: NormalizeMode, when: Literal["before", "after"], x: np.ndarray) -> np.ndarray:
    if mode in {"both", when}:
        return _l2_normalize_rows(x)
    return x


def _copy_or_transform_rows(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    *,
    method: Literal["slice", "rp", "pca"],
    dim: int,
    chunk_rows: int,
    normalize: NormalizeMode,
    rp_matrix: Optional[np.ndarray] = None,
    pca: Optional[object] = None,
) -> None:
    n, src_dim = src_ds.shape
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        x = src_ds[start:stop, :].astype(np.float32, copy=False)
        x = _apply_normalize(normalize, "before", x)

        if method == "slice":
            y = x[:, :dim]
        elif method == "rp":
            if rp_matrix is None:
                raise ValueError("rp_matrix is required for method=rp")
            if rp_matrix.shape != (src_dim, dim):
                raise ValueError(f"rp_matrix shape mismatch: {rp_matrix.shape} != {(src_dim, dim)}")
            y = x @ rp_matrix
        elif method == "pca":
            if pca is None:
                raise ValueError("pca model is required for method=pca")
            y = pca.transform(x).astype(np.float32, copy=False)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unknown method: {method}")

        y = _apply_normalize(normalize, "after", y)
        dst_ds[start:stop, :] = y


def _make_rp_matrix(src_dim: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # GaussianRandomProjection uses N(0, 1/sqrt(n_components))
    return (rng.standard_normal(size=(src_dim, dim), dtype=np.float32) / np.sqrt(dim)).astype(np.float32)


def _recompute_gt_faiss_cosine(
    out_path: Path,
    *,
    train_key: str,
    test_key: str,
    k: int,
    chunk_rows: int,
) -> None:
    if faiss is None:
        raise RuntimeError(
            "FAISS is not available (import faiss failed). "
            "Install faiss-cpu (or faiss-gpu) to recompute ground-truth at scale."
        )

    with h5py.File(out_path, "r+") as f:
        train = f[train_key]
        test = f[test_key]
        if len(train.shape) != 2 or len(test.shape) != 2:
            raise ValueError("train/test must be 2D")

        d = int(train.shape[1])
        index = faiss.IndexFlatIP(d)

        # add train vectors (normalized) in chunks
        for start in range(0, train.shape[0], chunk_rows):
            stop = min(train.shape[0], start + chunk_rows)
            xb = train[start:stop, :].astype(np.float32, copy=False)
            xb = _l2_normalize_rows(xb)
            index.add(xb)

        # create/replace neighbors & distances
        if "neighbors" in f:
            del f["neighbors"]
        if "distances" in f:
            del f["distances"]
        neighbors_ds = f.create_dataset("neighbors", shape=(test.shape[0], k), dtype=np.int32)
        distances_ds = f.create_dataset("distances", shape=(test.shape[0], k), dtype=np.float32)

        for start in range(0, test.shape[0], chunk_rows):
            stop = min(test.shape[0], start + chunk_rows)
            xq = test[start:stop, :].astype(np.float32, copy=False)
            xq = _l2_normalize_rows(xq)
            sims, idx = index.search(xq, k)
            neighbors_ds[start:stop, :] = idx.astype(np.int32, copy=False)
            distances_ds[start:stop, :] = (1.0 - sims).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create dimension-variant HDF5 datasets (slice/rp/pca) with optional GT recompute."
    )
    parser.add_argument("--input", required=True, help="Input HDF5 file path")
    parser.add_argument(
        "--dimensions",
        "--dims",
        required=True,
        nargs="+",
        type=int,
        help="List of target dimensions (e.g. 16 32 64)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root directory. Default: input file's parent directory.",
    )
    parser.add_argument(
        "--train-key", default="train", help="Dataset key for base vectors (default: train)"
    )
    parser.add_argument(
        "--test-key", default="test", help="Dataset key for query vectors (default: test)"
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=10000,
        help="Rows per copy chunk (default: 10000). Reduce if RAM is tight.",
    )
    parser.add_argument(
        "--method",
        choices=["slice", "rp", "pca"],
        default="slice",
        help="How to produce lower-dim vectors (default: slice).",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "before", "after", "both"],
        default="none",
        help="L2-normalize vectors before/after transform (default: none). For cosine datasets, 'both' is common.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for RP/PCA (default: 0).")
    parser.add_argument(
        "--compression",
        choices=["inherit", "gzip", "none"],
        default="inherit",
        help="Compression for output train/test datasets (default: inherit).",
    )
    parser.add_argument(
        "--recompute-gt",
        action="store_true",
        help="Recompute neighbors/distances for the new vectors (recommended if you care about precision/recall).",
    )
    parser.add_argument("--k", type=int, default=100, help="Top-k for ground-truth neighbors (default: 100).")
    parser.add_argument(
        "--gt-metric",
        choices=["cosine"],
        default="cosine",
        help="Ground-truth metric (default: cosine). For ann-benchmarks angular datasets distances=1-cosine.",
    )
    parser.add_argument(
        "--naming",
        choices=["auto", "ann", "flat"],
        default="auto",
        help="Output naming scheme (default: auto).",
    )
    parser.add_argument(
        "--output-pattern",
        default="{stem}-dim{dim}.hdf5",
        help="Used when --naming flat (default: {stem}-dim{dim}.hdf5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else input_path.parent.resolve()
    )

    dims = sorted(set(args.dimensions))
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dimensions must be positive, got: {dims}")

    with h5py.File(input_path, "r") as src:
        if args.train_key not in src or args.test_key not in src:
            raise KeyError(
                f"Input file must contain datasets '{args.train_key}' and '{args.test_key}'. "
                f"Found keys: {list(src.keys())}"
            )

        train_src = src[args.train_key]
        test_src = src[args.test_key]

        if len(train_src.shape) != 2 or len(test_src.shape) != 2:
            raise ValueError(
                f"Expected 2D train/test. Got train={train_src.shape}, test={test_src.shape}"
            )

        src_dim = train_src.shape[1]
        if test_src.shape[1] != src_dim:
            raise ValueError(
                f"train/test dimension mismatch: train D={src_dim}, test D={test_src.shape[1]}"
            )

        normalize: NormalizeMode = args.normalize  # type: ignore[assignment]
        method: Literal["slice", "rp", "pca"] = args.method

        for dim in dims:
            if dim > src_dim:
                raise ValueError(
                    f"Requested dim {dim} exceeds source dim {src_dim} (input: {input_path})"
                )

            out_path = _compute_output_paths(
                input_path=input_path,
                dim=dim,
                output_root=output_root,
                naming=args.naming,
                output_pattern=args.output_pattern,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                if args.overwrite:
                    os.remove(out_path)
                else:
                    raise FileExistsError(
                        f"Output already exists: {out_path}. Use --overwrite to replace it."
                    )

            print(f"[dim={dim}] writing -> {out_path}")
            with h5py.File(out_path, "w") as dst:
                # File-level attributes
                _copy_attrs(src.attrs, dst.attrs)

                # Copy non train/test datasets at root level.
                # If we're recomputing GT, skip old neighbors/distances.
                for name, obj in src.items():
                    if name in {args.train_key, args.test_key}:
                        continue
                    if args.recompute_gt and name in {"neighbors", "distances"}:
                        continue
                    src.copy(name, dst)

                compression = None if args.compression == "inherit" else (None if args.compression == "none" else "gzip")

                # Create train/test and stream-transform data
                train_dst = _create_like_vectors(
                    dst,
                    args.train_key,
                    train_src,
                    dim,
                    dtype=np.dtype(np.float32),
                    compression=compression,
                )
                test_dst = _create_like_vectors(
                    dst,
                    args.test_key,
                    test_src,
                    dim,
                    dtype=np.dtype(np.float32),
                    compression=compression,
                )

                # If dim == src_dim, keep the original vector space (no RP/PCA),
                # even if user requested rp/pca. This makes the "same-dim variant"
                # identical (modulo optional normalization).
                effective_method: Literal["slice", "rp", "pca"] = method
                rp_matrix = None
                pca = None
                if dim == src_dim:
                    effective_method = "slice"
                else:
                    if method == "rp":
                        rp_matrix = _make_rp_matrix(int(src_dim), int(dim), int(args.seed))
                    elif method == "pca":
                        if IncrementalPCA is None:
                            raise RuntimeError(
                                "scikit-learn is required for --method pca, but it's not installed in this env."
                            )
                        # Fit PCA on train in a streaming fashion (2-pass overall: fit then transform)
                        pca = IncrementalPCA(n_components=dim, batch_size=args.chunk_rows)
                        for start in range(0, train_src.shape[0], args.chunk_rows):
                            stop = min(train_src.shape[0], start + args.chunk_rows)
                            xb = train_src[start:stop, :].astype(np.float32, copy=False)
                            xb = _apply_normalize(normalize, "before", xb)
                            pca.partial_fit(xb)

                _copy_or_transform_rows(
                    train_src,
                    train_dst,
                    method=effective_method,
                    dim=dim,
                    chunk_rows=args.chunk_rows,
                    normalize=normalize,
                    rp_matrix=rp_matrix,
                    pca=pca,
                )
                _copy_or_transform_rows(
                    test_src,
                    test_dst,
                    method=effective_method,
                    dim=dim,
                    chunk_rows=args.chunk_rows,
                    normalize=normalize,
                    rp_matrix=rp_matrix,
                    pca=pca,
                )

                # Dataset-level attributes
                _copy_attrs(train_src.attrs, train_dst.attrs)
                _copy_attrs(test_src.attrs, test_dst.attrs)

            if args.recompute_gt:
                if args.gt_metric != "cosine":
                    raise ValueError(f"Unsupported gt metric: {args.gt_metric}")
                print(f"[dim={dim}] recomputing GT (k={args.k}, metric=cosine) -> {out_path}")
                _recompute_gt_faiss_cosine(
                    out_path,
                    train_key=args.train_key,
                    test_key=args.test_key,
                    k=int(args.k),
                    chunk_rows=int(args.chunk_rows),
                )

            print(
                f"[dim={dim}] done (train: {train_src.shape[0]}x{dim}, test: {test_src.shape[0]}x{dim}, method={effective_method})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


