DATASET_NAME="gist-960-euclidean"

poetry run python3 make_hdf5_dimension_variants.py \
    --input /dev/shm/${DATASET_NAME}/${DATASET_NAME}.hdf5 \
    --dimensions 128 256 512 768 \
    --method slice \
    --normalize none \
    --recompute-gt \
    --overwrite

