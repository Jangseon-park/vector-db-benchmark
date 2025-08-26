# Estimate the size of the vector database

# 1. load dataset type
# 2. start the docker containers
# 3. upload the dataset
# 4. estimate the size
# 5. stop the docker containers
# 6. print the size & save the size to a file

import typer
import os
import json
import time
import random
import string
import subprocess
import shutil
import logging
import requests
import sys
import re
from benchmark.config_read import read_dataset_config, read_engine_configs
from pymilvus import utility, connections, Collection
from engine.clients.milvus.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)

# Use the same collection name as in the benchmark runs
MILVUS_COLLECTION_NAME = "benchmark"


def start_docker_containers():
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", "milvus-single-node"
    )

    # Clean up previous run's volumes before starting for a clean slate
    volume_path = os.path.join(path, "volumes")
    if os.path.exists(volume_path):
        print("Removing existing volumes for a clean start...")
        shutil.rmtree(volume_path)
        print("Volumes directory removed.")

    try:
        # Use check=True to catch errors if docker-compose fails
        print("Starting Milvus containers...")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=path, check=True)
    except Exception as e:
        raise Exception(f"Failed to start containers: {e}")

    # Wait for a moment to ensure the service is responsive
    time.sleep(15)

    # check if the containers are running
    if (
        subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "-f",
                "name=milvus-standalone",
                "-f",
                "status=running",
                "-q",
            ],
            cwd=path,
            capture_output=True,
        ).stdout.strip()
        == b""
    ):
        raise Exception("Milvus containers did not start or are not running")
    else:
        print("Milvus containers started")
    return path


def stop_docker_containers():
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", "milvus-single-node"
    )
    try:
        # Stop and remove containers, networks, and volumes
        subprocess.run(["docker", "compose", "down", "-v"], cwd=path, check=True)
        print("Milvus containers and volumes stopped and removed.")
    except Exception as e:
        raise Exception(f"Failed to stop containers and remove volumes: {e}")
    time.sleep(5)
    # check if the containers are stopped
    if (
        subprocess.run(
            ["docker", "ps", "-a", "-f", "name=milvus-standalone", "-q"],
            cwd=path,
            capture_output=True,
        ).stdout.strip()
        == b""
    ):
        print("Milvus containers confirmed to be stopped.")
    else:
        # The new cleanup logic in start_docker_containers makes this fallback less critical,
        # but we leave it as a last resort.
        print("Warning: Milvus containers did not appear to stop cleanly.")


def upload_dataset(dataset_name: str, engine_name: str):
    path = os.path.dirname(__file__)
    # Use sys.executable to ensure we are using the same python interpreter
    # that is running this script (which is the one from the poetry venv)
    python_executable = sys.executable
    cmd = [
        python_executable,
        "run.py",
        "--engines",
        engine_name,
        "--datasets",
        dataset_name,
        "--no-skip-upload",
        "--skip-search",
    ]
    try:
        print(f"Running upload command: {' '.join(cmd)}")
        # The CWD should be the project root where run.py is located.
        run_path = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(cmd, cwd=run_path, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to upload dataset {dataset_name} with engine {engine_name}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during upload: {e}")
    return path

def wait_for_indexing_completion():
    """Connects to Milvus and waits for indexing and compaction to complete."""
    print("Connecting to Milvus to wait for indexing completion...")
    connections.connect("default", host="localhost", port="19530")

    if not utility.has_collection(MILVUS_COLLECTION_NAME):
        print(f"Collection {MILVUS_COLLECTION_NAME} not found. Skipping wait.")
        connections.disconnect("default")
        return

    try:
        collection = Collection(MILVUS_COLLECTION_NAME)
        print(f"Flushing collection '{MILVUS_COLLECTION_NAME}'...")
        collection.flush()
        print("Flush completed. Now waiting for indexing...")

        while True:
            progress = utility.index_building_progress(MILVUS_COLLECTION_NAME)
            if not progress or progress.get("total_rows", 0) == progress.get("indexed_rows", 0):
                print("Indexing appears to be complete.")
                break
            else:
                print(f"Indexing in progress: {progress.get('indexed_rows', 0)} / {progress.get('total_rows', 0)} rows indexed.")
            time.sleep(10)
        
        print("Waiting for any compaction to complete...")
        while True:
            compaction_state = collection.get_compaction_state()
            if compaction_state.state.name == 'Completed':
                 print("Compaction is complete.")
                 break
            else:
                print(f"Compaction in progress, state: {compaction_state.state.name}...")
            time.sleep(10)

    except Exception as e:
        print(f"An error occurred while waiting for indexing: {e}")
    finally:
        connections.disconnect("default")
        print("Disconnected from Milvus.")


def estimate_size(dataset_name: str, engine_name: str):
    result = {
        "dataset_name": dataset_name,
        "engine_name": engine_name,
        "disk_size": {},
    }
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", "milvus-single-node"
    )

    # Measure disk usage for Milvus, MinIO and ETCD volumes
    print("Measuring disk usage...")
    
    volumes_to_measure = ["milvus", "etcd", "minio"]
    for volume in volumes_to_measure:
        volume_path = os.path.join(path, "volumes", volume)
        cmd = f"sudo du -sh {volume_path}"
        try:
            size_result = subprocess.run(
                cmd, shell=True, cwd=path, capture_output=True, check=True, text=True
            )
            result["disk_size"][volume] = size_result.stdout.strip().split("\t")[0]
        except Exception as e:
            print(f"Failed to estimate size for {volume}: {e}")
            result["disk_size"][volume] = "Error"

    return result


def save_result(result: dict, dataset_name: str, engine_name: str):
    path = os.path.join(
        os.path.dirname(__file__), "results", f"{dataset_name}_{engine_name}_size.json"
    )
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(result, f, indent=4)
    return path


def test():
    engine_config = ["milvus-diskann", "milvus-hnsw", "milvus-ivf-flat"]
    dataset_config = [
        "glove-25-angular",
        "glove-100-angular",
        "deep-image-96-angular",
        "gist-960-angular",
        "dbpedia-openai-1M-1536-angular",
    ]
    for dataset_name in dataset_config:
        for engine_name in engine_config:
            print(f"Estimating size for {dataset_name} with {engine_name}")
            start_docker_containers()
            try:
                upload_dataset(dataset_name, engine_name)
                wait_for_indexing_completion()
                result = estimate_size(dataset_name, engine_name)
                print(result)
                save_result(result, dataset_name, engine_name)
            finally:
                # Ensure containers are stopped even if there is an error
                stop_docker_containers()


if __name__ == "__main__":
    test()


