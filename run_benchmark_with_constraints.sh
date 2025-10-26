#!/bin/bash

# This script automates running the vector-db-benchmark with cgroup resource constraints.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Path to the docker-compose file for the desired engine
COMPOSE_FILE="engine/servers/milvus-single-node/docker-compose.yaml"
# The name of the cgroup slice defined in the docker-compose file -> ## NOTE: SHOULD BE THE SAME AS THE ONE IN THE DOCKER-COMPOSE FILE ##
SLICE_NAME="ex.slice"
SLICE_FILE="$SLICE_NAME.d"
SLICE_PATH="/etc/systemd/system.control/$SLICE_FILE"
# move to project root directory and search for venv path: /home/wolf/.local/bin/poetry env info -p
VENV_PATH="/home/wolf/.cache/pypoetry/virtualenvs/vector-db-benchmark-3zx8bqwV-py3.10"

# Engine name to pass to the benchmark script
ENGINE_NAME="milvus-default-hnsw"
# Dataset to use for the benchmark (e.g., "glove-100-angular")
# You can find more datasets in the `datasets` directory or the project's documentation.
DATASET_NAME="glove-25-angular"

# --- Script ---

echo "Ensuring a clean environment by stopping any running containers..."
# Use --remove-orphans to also remove containers from old services. -v to remove volumes.
sudo docker compose -f "$COMPOSE_FILE" down --remove-orphans -v

echo "Resetting any failed cgroup slice from previous runs..."
# This is to prevent conflicts with zombie cgroups. Ignore errors if the slice doesn't exist.
sudo systemctl stop "$SLICE_NAME"
sudo rm -rf "$SLICE_PATH"
sudo systemctl daemon-reload


echo "Starting database engine using docker-compose..."
# Start the service in detached mode
sudo docker compose -f "$COMPOSE_FILE" up -d
echo "Waiting for the cgroup slice to be created (5 seconds)..."
sleep 5

if sudo systemctl set-property "$SLICE_NAME" AllowedMemoryNodes=2 "AllowedCPUs=0-19"; then
    echo "Successfully applied AllowedMemoryNodes=2 and AllowedCPUs from node 0 to $SLICE_NAME."
else
    echo "Failed to apply systemctl properties. Cleaning up..."
    sudo docker compose -f "$COMPOSE_FILE" down
    exit 1
fi
#echo "Applying resource constraints to the cgroup slice..."
# This command requires sudo privileges.
echo "Uploading the dataset for engine '$ENGINE_NAME' with dataset '$DATASET_NAME'..."
# Run the benchmark. Use absolute path to the virtualenv python to avoid path issues.
"$VENV_PATH/bin/python" run.py --engines "$ENGINE_NAME" --datasets "$DATASET_NAME" --skip-search
echo "Searching the dataset for engine '$ENGINE_NAME' with dataset '$DATASET_NAME'..."
"$VENV_PATH/bin/python" run.py --engines "$ENGINE_NAME" --datasets "$DATASET_NAME" --skip-upload --drop-caches

echo "Benchmark finished. Cleaning up..."
# Stop and remove the containers
sudo docker compose -f "$COMPOSE_FILE" down

echo "Stopping the cgroup slice..."
sudo systemctl stop "$SLICE_NAME"
sudo rm -rf "$SLICE_PATH"
sudo systemctl daemon-reload
echo "Script completed successfully."
