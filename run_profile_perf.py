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
from datetime import datetime
from termcolor import colored
from typing import Dict, Any, List
from benchmark.config_read import read_dataset_config, read_engine_configs
from pymilvus import utility, connections, Collection
from engine.clients.milvus.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)

# Use the same collection name as in the benchmark runs
MILVUS_COLLECTION_NAME = "benchmark"


def remove_volumes(model: str, size: int):
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", f"{model}", f"{size}"
    )   
    volume_path = os.path.join(path, "volumes")
    if os.path.exists(volume_path):
        subprocess.run(["sudo", "rm", "-rf", volume_path], check=True)
        print("Volumes directory removed.")
    else:
        print("Volumes directory does not exist.")


def start_docker_containers(size: int):
    path = os.path.join(
        os.path.dirname(__file__), "engine", "servers", f"milvus-single-node", f"{size}"
    )

    # Clean up previous run's volumes before starting for a clean slate
    #
    #volume_path = os.path.join(path, "volumes")
    #if os.path.exists(volume_path):
    #    print("Removing existing volumes for a clean start...")
    #    # Use sudo to remove the directory which is owned by root (from docker)
    #    subprocess.run(["sudo", "rm", "-rf", volume_path], check=True)
    #    print("Volumes directory removed.")
    is_exist = False
    volume_path = os.path.join(path, "volumes")
    if os.path.exists(volume_path):
        is_exist = True

    try:
        # Use check=True to catch errors if docker-compose fails
        print("Starting Milvus containers...")
        subprocess.run(["sudo", "docker", "compose", "up", "-d"], cwd=path, check=True)
    except Exception as e:
        raise Exception(f"Failed to start containers: {e}")

    # Wait for a moment to ensure the service is responsive
    time.sleep(5)

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
    return is_exist


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

    #volume_path = os.path.join(path, "volumes")
    #if os.path.exists(volume_path):
    #    print("Removing existing volumes for a clean start...")
    #    # Use sudo to remove the directory which is owned by root (from docker)
    #    subprocess.run(["sudo", "rm", "-rf", volume_path], check=True)
    #    print("Volumes directory removed.")

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
        "--drop-caches",
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


def parse_perf_text(perf_output: str) -> str:
    """
    Parses the raw text output of `perf script`.
    This approach is more robust than using the Python scripting extension.
    """
    output_lines = ["{:<12} {:<16} {:<10} {:<12} {}".format(
        'TIME(s)', 'COMM', 'PID', 'EVENT', 'DETAILS')]

    # Regex to capture the common header of a perf script line
    # e.g., " python3  736818 [017] 12241.973464: ..."
    line_regex = re.compile(r'^\s*(?P<comm>.+?)\s+(?P<pid>\d+)\s+\[\d+\]\s+(?P<time>[\d\.]+):')

    for line in perf_output.splitlines():
        match = line_regex.match(line)
        if not match:
            continue

        data = match.groupdict()
        comm = data['comm'].strip()
        pid = data['pid']
        ts = float(data['time'])

        if 'major-faults' in line:
            output_lines.append("{:<12.6f} {:<16} {:<10} {:<12}".format(
                ts, comm, pid, 'Major Fault'))
        elif 'block:block_rq_issue' in line:
            # For block I/O, find the 'bytes' field in the details part
            bytes_match = re.search(r'bytes=(\d+)', line)
            bytes_val = bytes_match.group(1) if bytes_match else 'N/A'
            output_lines.append("{:<12.6f} {:<16} {:<10} {:<12} Size={}".format(
                ts, comm, pid, 'Disk IO', bytes_val))

    return "\n".join(output_lines)


def run_profile(dataset_name: str, engine_name: str, size: int, iteration_num: int):
    path = os.path.dirname(__file__)
    # Use sys.executable to ensure we are using the same python interpreter
    # that is running this script (which is the one from the poetry venv)
    python_executable = sys.executable

    output_dir = os.path.join(path, f"profile_results/{dataset_name}/{size}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{engine_name}_{iteration_num}.txt")
    perf_data_path = os.path.join(output_dir, f"{engine_name}_{iteration_num}.data")

    # sudo perf record creates a root-owned file, so we need sudo to remove it if it exists.
    if os.path.exists(perf_data_path):
        subprocess.run(["sudo", "rm", "-f", perf_data_path], check=True)

    profile_cmd = [
        "sudo",
        "-E",
        "perf",
        "record",
        "-a",
        "-e",
        "major-faults",
        "-e",
        "block:block_rq_issue",
        "-o",
        perf_data_path,
        "--",
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
        "--drop-caches",
    ]
    # flush the cache
    print("Flushing system cache...")
    subprocess.run(["sudo", "sync"], check=True)
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
    time.sleep(10)

    run_path = os.path.dirname(os.path.abspath(__file__))
    # Combine the perf command and the command to be profiled.
    profile_and_run_cmd = profile_cmd + cmd
    try:
        print(f"Running command with perf: {' '.join(profile_and_run_cmd)}")
        # This is the main, blocking operation. perf will start, run the command, and stop automatically.
        subprocess.run(profile_and_run_cmd, cwd=run_path, check=True)

    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to run main command with perf for {dataset_name} with engine {engine_name}: {e}"
        )
    except Exception as e:
        raise Exception(f"An unexpected error occurred during run: {e}")

    # Process the collected perf data
    if os.path.exists(perf_data_path):
        print(f"Processing perf data from {perf_data_path}...")
        try:
            # The perf.data file is owned by root, change ownership to process without sudo.
            user_id = os.getuid()
            group_id = os.getgid()
            subprocess.run(
                ["sudo", "chown", f"{user_id}:{group_id}", perf_data_path], check=True
            )

            script_cmd = ["perf", "script", "-i", perf_data_path]
            result = subprocess.run(
                script_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            parsed_content = parse_perf_text(result.stdout)
            with open(output_path, "w") as f:
                f.write(parsed_content)
            print(f"✅ Parsed perf data saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(colored(f"❌ Error parsing perf data: {e}", "red"))
            print(colored(f"Stderr: {e.stderr}", "red"))

        # Clean up the raw perf data file
        if os.path.exists(perf_data_path):
             os.remove(perf_data_path)
    else:
        print(f"Warning: perf data file not found at {perf_data_path}")


    
def confirm_loaded(dataset_name: str, engine_name: str, size: int):
    path = os.path.dirname(__file__)
    python_executable = sys.executable
    cmd = [python_executable, "run.py", "--engines", engine_name, "--datasets", dataset_name, "--check-loaded"]
    print(f"Running check-loaded command: {' '.join(cmd)}")
    # This is the main, blocking operation.
    subprocess.run(cmd, cwd=path, check=True)
    print("Check-loaded command finished.")

def clear_all_collections():
    engine_config = ["milvus-single-node"]
    size_config = [256, 512, 768, 1024, 2048, 4096]
    for engine_name in engine_config:
        for size in size_config:
            remove_volumes(engine_name, size)


def test():
    engine_config = ["milvus-default-self"]
    dataset_config = [
        "glove-100-angular",
        "gist-960-angular",
        "dbpedia-openai-1M-1536-angular",
    ]
    size_config = [256, 512, 768, 1024, 2048, 4096]
    iteration_num = 10
    for dataset_name in dataset_config:
        clear_all_collections()
        for engine_name in engine_config:
            for size in size_config:
                for i in range(iteration_num):
                    print(f"Running iteration {i} of {iteration_num} for {dataset_name} with {engine_name} and size {size}")
                    try:
                        is_exist = start_docker_containers(size)
                        if not is_exist:
                            upload_dataset(dataset_name, engine_name)
                        confirm_loaded(dataset_name, engine_name, size)
                        run_profile(dataset_name, engine_name, size, i)
                        stop_docker_containers(size)
                    finally:
                        # Ensure containers are stopped even if there is an error
                        stop_docker_containers(size)


if __name__ == "__main__":
    test()
