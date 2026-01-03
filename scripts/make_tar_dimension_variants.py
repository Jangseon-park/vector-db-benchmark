#!/usr/bin/env python3
"""
Create multiple "dimension variants" for tar/directory datasets used by
`dataset_reader/ann_compound_reader.py` (ann-filtered-benchmark-datasets format).

Expected input layout (either a directory or a .tgz/.tar.gz archive):
  - vectors.npy    (shape: [N, D], float32/float64)
  - tests.jsonl    (each line JSON; must contain "query" (list[float]),
                    and typically contains "conditions", "closest_ids", "closest_scores")

This script produces, for each requested dimension d, an output directory containing:
  - vectors.npy    (shape [N, d])
  - tests.jsonl    (query vectors transformed to dimension d)

Transform methods:
  - slice: take first d dimensions ([:, :d])
  - rp: Gaussian random projection (fixed seed)
  - pca: IncrementalPCA (requires scikit-learn)

Ground-truth note:
`tests.jsonl` usually includes "closest_ids"/"closest_scores" computed for the original
vector space. After any transform, that GT is no longer valid. If you don't care about
precision/recall (e.g. fingerprinting latency/throughput), you can set GT to empty
arrays so the benchmark's precision calculation becomes a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

try:
    from sklearn.decomposition import IncrementalPCA  # type: ignore
except Exception:  # pragma: no cover
    IncrementalPCA = None


NormalizeMode = Literal["none", "before", "after", "both"]
Method = Literal["slice", "rp", "pca"]


@dataclass(frozen=True)
class NamingPlan:
    output_dirname: str


def _parse_ann_benchmarks_name(name: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse names like 'dbpedia-openai-100K-1536-angular' -> ('dbpedia-openai', 100, 'K-1536-angular')
    or 'glove-100-angular' -> ('glove', 100, 'angular').
    We only need a generic "<prefix>-<dim>-<suffix>" split.
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


def _compute_output_dir(
    input_path: Path,
    dim: int,
    output_root: Path,
    naming: Literal["auto", "ann", "flat"],
    output_pattern: str,
) -> Path:
    parent_name = input_path.parent.name
    stem = input_path.stem

    ann_parent = _parse_ann_benchmarks_name(parent_name)
    ann_stem = _parse_ann_benchmarks_name(stem)

    if naming in {"auto", "ann"}:
        parsed = ann_parent or ann_stem
        if parsed is None:
            if naming == "ann":
                raise ValueError(
                    "Cannot parse ann-style name from input path. Use --naming flat "
                    "or provide names like dbpedia-openai-100K-1536-angular."
                )
        else:
            prefix, _, suffix = parsed
            variant = f"{prefix}-{dim}-{suffix}"
            return output_root / variant

    # flat
    dirname = output_pattern.format(stem=stem, dim=dim)
    return output_root / dirname


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _apply_normalize(mode: NormalizeMode, when: Literal["before", "after"], x: np.ndarray) -> np.ndarray:
    if mode in {"both", when}:
        return _l2_normalize_rows(x)
    return x


def _make_rp_matrix(src_dim: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # GaussianRandomProjection uses N(0, 1/sqrt(n_components))
    return (rng.standard_normal(size=(src_dim, dim), dtype=np.float32) / np.sqrt(dim)).astype(np.float32)


def _fit_pca_incremental(vectors: np.memmap, dim: int, batch: int, normalize: NormalizeMode) -> object:
    if IncrementalPCA is None:
        raise RuntimeError("scikit-learn is required for --method pca, but it's not installed.")
    pca = IncrementalPCA(n_components=dim, batch_size=batch)
    n = vectors.shape[0]
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        xb = np.asarray(vectors[start:stop, :], dtype=np.float32)
        xb = _apply_normalize(normalize, "before", xb)
        pca.partial_fit(xb)
    return pca


def _transform_matrix(
    x: np.ndarray,
    *,
    method: Method,
    dim: int,
    normalize: NormalizeMode,
    rp_matrix: Optional[np.ndarray],
    pca: Optional[object],
) -> np.ndarray:
    x = _apply_normalize(normalize, "before", x)
    if method == "slice":
        y = x[:, :dim]
    elif method == "rp":
        if rp_matrix is None:
            raise ValueError("rp_matrix required for method=rp")
        y = x @ rp_matrix
    elif method == "pca":
        if pca is None:
            raise ValueError("pca required for method=pca")
        y = pca.transform(x).astype(np.float32, copy=False)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown method: {method}")
    y = _apply_normalize(normalize, "after", y)
    return y.astype(np.float32, copy=False)


def _write_vectors_npy(
    in_vectors_path: Path,
    out_vectors_path: Path,
    *,
    dim: int,
    method: Method,
    normalize: NormalizeMode,
    seed: int,
    chunk_rows: int,
) -> Tuple[int, int, Optional[np.ndarray], Optional[object]]:
    vectors = np.load(in_vectors_path, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError(f"vectors.npy must be 2D, got shape={vectors.shape}")
    n, src_dim = vectors.shape
    if dim > src_dim:
        raise ValueError(f"Requested dim {dim} exceeds source dim {src_dim}")

    rp_matrix = None
    pca = None
    if method == "rp":
        rp_matrix = _make_rp_matrix(int(src_dim), int(dim), int(seed))
    elif method == "pca":
        pca = _fit_pca_incremental(vectors, dim=dim, batch=chunk_rows, normalize=normalize)

    out = np.lib.format.open_memmap(
        out_vectors_path, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        xb = np.asarray(vectors[start:stop, :], dtype=np.float32)
        out[start:stop, :] = _transform_matrix(
            xb, method=method, dim=dim, normalize=normalize, rp_matrix=rp_matrix, pca=pca
        )
    out.flush()
    return int(n), int(src_dim), rp_matrix, pca


def _write_tests_jsonl(
    in_tests_path: Path,
    out_tests_path: Path,
    *,
    dim: int,
    method: Method,
    normalize: NormalizeMode,
    rp_matrix: Optional[np.ndarray],
    pca: Optional[object],
    empty_gt: bool,
) -> int:
    count = 0
    with open(in_tests_path, "r", encoding="utf-8") as fin, open(
        out_tests_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = np.asarray(row["query"], dtype=np.float32)[None, :]
            q2 = _transform_matrix(
                q, method=method, dim=dim, normalize=normalize, rp_matrix=rp_matrix, pca=pca
            )[0]
            row["query"] = q2.tolist()
            if empty_gt:
                # Keep keys present but empty so AnnCompoundReader can parse,
                # and benchmark precision calc becomes a no-op (empty list is falsy).
                row["closest_ids"] = []
                row["closest_scores"] = []
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _resolve_input_to_dir(input_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Returns (directory_path, tempdir_handle_or_None).
    If input is a tgz/tar.gz, extract to temp dir and return extracted dir.
    If input is a directory, return it.
    """
    if input_path.is_dir():
        return input_path, None
    if input_path.name.endswith(".tgz") or input_path.name.endswith(".tar.gz"):
        td = tempfile.TemporaryDirectory(prefix="tar_dataset_")
        out_dir = Path(td.name)
        with tarfile.open(input_path, "r:gz") as tf:
            tf.extractall(out_dir)
        return out_dir, td
    raise ValueError("Input must be a directory or a .tgz/.tar.gz archive")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create dimension variants for tar/directory datasets (vectors.npy + tests.jsonl)."
    )
    ap.add_argument("--input", required=True, help="Input dataset dir or .tgz/.tar.gz")
    ap.add_argument(
        "--dims",
        "--dimensions",
        required=True,
        nargs="+",
        type=int,
        help="Target dimensions (e.g. 128 256 512)",
    )
    ap.add_argument(
        "--output-root",
        default=None,
        help="Output root directory. Default: input's parent directory.",
    )
    ap.add_argument(
        "--method",
        choices=["slice", "rp", "pca"],
        default="rp",
        help="Transform method (default: rp).",
    )
    ap.add_argument(
        "--normalize",
        choices=["none", "before", "after", "both"],
        default="both",
        help="L2-normalize before/after transform (default: both).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for RP/PCA (default: 0).")
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=10000,
        help="Rows per chunk when processing vectors.npy (default: 10000).",
    )
    ap.add_argument(
        "--naming",
        choices=["auto", "ann", "flat"],
        default="auto",
        help="Output directory naming scheme (default: auto).",
    )
    ap.add_argument(
        "--output-pattern",
        default="{stem}-dim{dim}",
        help="Used when --naming flat. Output dir name pattern (default: {stem}-dim{dim}).",
    )
    ap.add_argument(
        "--empty-gt",
        action="store_true",
        help="Set closest_ids/closest_scores to [] in output tests.jsonl (recommended if you won't use precision/recall).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output directories if exist.")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else input_path.parent.resolve()
    )

    dims = sorted(set(args.dims))
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dims must be positive, got: {dims}")

    method: Method = args.method
    normalize: NormalizeMode = args.normalize  # type: ignore[assignment]

    data_dir, td = _resolve_input_to_dir(input_path)
    try:
        vectors_path = data_dir / "vectors.npy"
        tests_path = data_dir / "tests.jsonl"
        if not vectors_path.exists() or not tests_path.exists():
            raise FileNotFoundError(
                f"Expected vectors.npy and tests.jsonl in {data_dir}, found: {list(p.name for p in data_dir.iterdir())}"
            )

        for dim in dims:
            out_dir = _compute_output_dir(
                input_path=input_path,
                dim=dim,
                output_root=output_root,
                naming=args.naming,
                output_pattern=args.output_pattern,
            )

            if out_dir.exists():
                if args.overwrite:
                    # best-effort cleanup
                    for p in out_dir.glob("**/*"):
                        if p.is_file():
                            p.unlink()
                    for p in sorted(out_dir.glob("**/*"), reverse=True):
                        if p.is_dir():
                            p.rmdir()
                    out_dir.rmdir()
                else:
                    raise FileExistsError(f"Output dir exists: {out_dir}. Use --overwrite to replace.")

            out_dir.mkdir(parents=True, exist_ok=True)
            out_vectors = out_dir / "vectors.npy"
            out_tests = out_dir / "tests.jsonl"

            print(f"[dim={dim}] writing -> {out_dir}")
            n, src_dim, rp_matrix, pca = _write_vectors_npy(
                vectors_path,
                out_vectors,
                dim=dim,
                method=method,
                normalize=normalize,
                seed=int(args.seed),
                chunk_rows=int(args.chunk_rows),
            )
            qn = _write_tests_jsonl(
                tests_path,
                out_tests,
                dim=dim,
                method=method,
                normalize=normalize,
                rp_matrix=rp_matrix,
                pca=pca,
                empty_gt=bool(args.empty_gt),
            )
            print(f"[dim={dim}] done (vectors: {n}x{dim} from {src_dim}, queries: {qn}, method={method})")

    finally:
        if td is not None:
            td.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


