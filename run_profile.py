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


def start_docker_containers(size: int):
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", f"milvus-single-node", f"{size}"
    )

    # Clean up previous run's volumes before starting for a clean slate
    volume_path = os.path.join(path, "volumes")
    if os.path.exists(volume_path):
        print("Removing existing volumes for a clean start...")
        # Use sudo to remove the directory which is owned by root (from docker)
        subprocess.run(["sudo", "rm", "-rf", volume_path], check=True)
        print("Volumes directory removed.")

    try:
        # Use check=True to catch errors if docker-compose fails
        print("Starting Milvus containers...")
        subprocess.run(["sudo", "docker", "compose", "up", "-d"], cwd=path, check=True)
    except Exception as e:
        raise Exception(f"Failed to start containers: {e}")

    # Wait for a moment to ensure the service is responsive
    time.sleep(15)

    # check if the containers are running
    if (
        subprocess.run(
            [
                "sudo",
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


def stop_docker_containers(size: int):
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", f"milvus-single-node", f"{size}"
    )
    try:
        # Stop and remove containers, networks, and volumes
        subprocess.run(["sudo", "docker", "compose", "down", "-v"], cwd=path, check=True)
        print("Milvus containers and volumes stopped and removed.")
    except Exception as e:
        raise Exception(f"Failed to stop containers and remove volumes: {e}")
    time.sleep(5)
    # check if the containers are stopped
    if (
        subprocess.run(
            ["sudo", "docker", "ps", "-a", "-f", "name=milvus-standalone", "-q"],
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

    volume_path = os.path.join(path, "volumes")
    if os.path.exists(volume_path):
        print("Removing existing volumes for a clean start...")
        # Use sudo to remove the directory which is owned by root (from docker)
        subprocess.run(["sudo", "rm", "-rf", volume_path], check=True)
        print("Volumes directory removed.")

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
    # flush the cache
    print("Flushing system cache...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
    time.sleep(10)

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


def run_profile(dataset_name: str, engine_name: str, size: int, iteration_num: int):
    path = os.path.dirname(__file__)
    # Use sys.executable to ensure we are using the same python interpreter
    # that is running this script (which is the one from the poetry venv)
    output_path = os.path.join(path, f"profile_results/{dataset_name}/{size}", f"{engine_name}_{iteration_num}.txt")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    python_executable = sys.executable
    profile_cmd = [
        "sudo",
        "-E",
        "python3",
        "trace_system_fault.py",
        "--output",
        f"{output_path}",
    ]

    cmd = [
        python_executable,
        "run.py",
        "--engines",
        engine_name,
        "--datasets",
        dataset_name,
        "--skip-upload",
        "--no-skip-search",
    ]
    # flush the cache
    print("Flushing system cache...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
    time.sleep(10)

    profiler_process = None
    try:
        print(f"Starting profiler in background: {' '.join(profile_cmd)}")
        run_path = os.path.dirname(os.path.abspath(__file__))
        profiler_process = subprocess.Popen(profile_cmd, cwd=run_path)

        # Give the profiler a moment to start up
        time.sleep(1)

        print(f"Running main command: {' '.join(cmd)}")
        # This is the main, blocking operation.
        subprocess.run(cmd, cwd=run_path, check=True)

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to run main command for {dataset_name} with engine {engine_name}: {e}"
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred during run: {e}")
    finally:
        if profiler_process:
            print("Main command finished. Terminating profiler...")
            profiler_process.terminate()
            try:
                # Wait for a moment for graceful shutdown
                profiler_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Profiler did not terminate gracefully. Killing...")
                profiler_process.kill()
            print("Profiler terminated.")


    
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


def test():
    engine_config = ["milvus-default-self"]
    dataset_config = [
        "gist-960-angular",
        "dbpedia-openai-1M-1536-angular",
    ]
    size_config = [256, 512, 768, 1024, 2048, 4096]
    iteration_num = 3
    for dataset_name in dataset_config:
        for engine_name in engine_config:
            for size in size_config:
                for i in range(iteration_num):
                    print(f"Running iteration {i} of {iteration_num} for {dataset_name} with {engine_name} and size {size}")
                    try:
                        start_docker_containers(size)
                        upload_dataset(dataset_name, engine_name)
                        wait_for_indexing_completion()
                        run_profile(dataset_name, engine_name, size, i)
                        stop_docker_containers(size)
                    finally:
                        # Ensure containers are stopped even if there is an error
                        stop_docker_containers(size)


if __name__ == "__main__":
    test()


