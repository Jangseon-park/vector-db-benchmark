#!/usr/bin/env python3
"""
This script automates running the vector-db-benchmark with cgroup resource constraints.
It is a Python equivalent of run_benchmark_with_constraints.sh, refactored into a class.
"""

import subprocess
import os
import time
import sys
import threading

class BenchmarkRunner:
    """
    Encapsulates the logic for running the vector-db-benchmark with cgroup constraints.
    """
    def __init__(self):
        """Initializes the BenchmarkRunner."""
        self.compose_file = None
        self.slice_name = None
        self.engine_name = None
        self.dataset_name = None
        self.venv_path = None
        self.slice_file = None
        self.slice_path = None
        self.python_exec = None
        self.run_py_script = "run.py"
        self.commands = {}

    def _create_commands(self):
        """Creates a dictionary of commands to be executed."""
        return {
            "docker_compose_down": ["sudo", "docker", "compose", "-f", self.compose_file, "down", "--remove-orphans", "-v"],
            "docker_compose_up": ["sudo", "docker", "compose", "-f", self.compose_file, "up", "-d"],
            "systemctl_stop": ["sudo", "systemctl", "stop", self.slice_name],
            "rm_slice_path": ["sudo", "rm", "-rf", self.slice_path],
            "systemctl_daemon_reload": ["sudo", "systemctl", "daemon-reload"],
            "set_property": [
                "sudo", "systemctl", "set-property", self.slice_name,
                "AllowedMemoryNodes=2", "AllowedCPUs=0-19"
            ],
            "upload": [self.python_exec, self.run_py_script, "--engines", self.engine_name, "--datasets", self.dataset_name, "--skip-search"],
            "search": [self.python_exec, self.run_py_script, "--engines", self.engine_name, "--datasets", self.dataset_name, "--skip-upload", "--drop-caches"]
        }

    def _run_command(self, command, check=True):
        """Runs a command, streams its output, and raises an exception on failure."""
        
        def stream_reader(pipe, output_list, stream_to_print):
            """Reads lines from a pipe, appends them to a list, and prints them to a stream."""
            for line in iter(pipe.readline, ''):
                stream_to_print.write(line)
                output_list.append(line)
            pipe.close()

        print(f"Executing: {' '.join(command)}")
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            stdout_output = []
            stderr_output = []

            stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, stdout_output, sys.stdout))
            stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, stderr_output, sys.stderr))

            stdout_thread.start()
            stderr_thread.start()

            stdout_thread.join()
            stderr_thread.join()
            
            return_code = process.wait()
            
            if check and return_code != 0:
                full_stderr = "".join(stderr_output)
                raise subprocess.CalledProcessError(return_code, command, output=None, stderr=full_stderr)
                
        except FileNotFoundError:
            print(f"Error: Command not found: {command[0]}", file=sys.stderr)
            print("Please ensure the command and its path are correct.", file=sys.stderr)
            raise
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}: {' '.join(command)}", file=sys.stderr)
            if e.stderr:
                print(f"stderr:\n{e.stderr}", file=sys.stderr)
            raise


    def _prepare(self, compose_file, slice_name, engine_name, dataset_name, venv_path):
        self.compose_file = compose_file
        self.slice_name = slice_name
        self.engine_name = engine_name
        self.dataset_name = dataset_name
        self.venv_path = venv_path
        
        self.slice_file = f"{self.slice_name}.d"
        self.slice_path = f"/etc/systemd/system.control/{self.slice_file}"
        self.python_exec = os.path.join(self.venv_path, "bin/python")
        self.commands = self._create_commands()

        if not os.path.exists(self.run_py_script) or not os.path.exists(self.compose_file):
            print(f"run_py_script: {self.run_py_script}")
            print(f"compose_file: {self.compose_file}")
            print(f"This script must be run from the 'vector-db-benchmark' directory.", file=sys.stderr)
            sys.exit(1)
            
    def _cleanup_before_run(self):
        """Ensures a clean environment before the benchmark starts."""
        print("🧹 Ensuring a clean environment by stopping any running containers...")
        self._run_command(self.commands["docker_compose_down"], check=False)

        print("🧹 Resetting any failed cgroup slice from previous runs...")
        self._run_command(self.commands["systemctl_stop"], check=False)
        self._run_command(self.commands["rm_slice_path"], check=False)
        self._run_command(self.commands["systemctl_daemon_reload"], check=False)

    def wrapup(self):
        """Cleans up all resources after the benchmark is finished."""
        print("🧹 Benchmark finished. Cleaning up...")
        try:
            self._run_command(self.commands["docker_compose_down"], check=False)
            self._run_command(self.commands["systemctl_stop"], check=False)
            self._run_command(self.commands["rm_slice_path"], check=False)
            self._run_command(self.commands["systemctl_daemon_reload"], check=False)
            print("Script completed successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n Benchmark wrapup failed: {e}", file=sys.stderr)

    def _start_engine(self):
        """Starts the database engine and waits for it to initialize."""
        print("🚀 Starting database engine using docker-compose...")
        self._run_command(self.commands["docker_compose_up"])
        print("⏳ Waiting for engine to initialize (2 seconds)...")
        time.sleep(2)
        print("🧹 Engine started successfully.")
    
    def _set_constraints(self):
        print("🛠️ Applying resource constraints to the cgroup slice...")
        self._run_command(self.commands["set_property"])
        print(f"Successfully applied AllowedMemoryNodes=2 and AllowedCPUs from node 0 to {self.slice_name}.")
        print("🧹 Constraints set successfully.")
    
    def _upload_dataset(self):
        print(f"Uploading the dataset for engine '{self.engine_name}' with dataset '{self.dataset_name}'...")
        self._run_command(self.commands["upload"])
        print("🧹 Dataset uploaded successfully.")
    
    def search_dataset(self):
        print(f"Searching the dataset for engine '{self.engine_name}' with dataset '{self.dataset_name}'...")
        self._run_command(self.commands["search"])
        print("🧹 Dataset searched successfully.")
    
    def run(self, compose_file, slice_name, engine_name, dataset_name, venv_path):
        self._prepare(compose_file, slice_name, engine_name, dataset_name, venv_path)   
        try:
            self._cleanup_before_run()
            self._start_engine()
            self._set_constraints()
            self._upload_dataset()
            self.search_dataset()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n Benchmark run failed: {e}", file=sys.stderr)
        finally:
            self.wrapup()

    def prepare(self, compose_file, slice_name, engine_name, dataset_name, venv_path):
        try:
            self._prepare(compose_file, slice_name, engine_name, dataset_name, venv_path)
            self._cleanup_before_run()
            self._start_engine()
            self._set_constraints()
            self._upload_dataset()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n Benchmark prepare failed: {e}", file=sys.stderr)



if __name__ == "__main__":
    
    # --- Configuration ---

    server_list = ["milvus", "qdrant", "weaviate", "pgvector"]
    for server in server_list:
        COMPOSE_FILE = f"engine/servers/{server}-single-node/docker-compose.yaml"
        SLICE_NAME = "ex.slice"
        VENV_PATH = "/home/wolf/.cache/pypoetry/virtualenvs/vector-db-benchmark-3zx8bqwV-py3.10"
        ENGINE_NAME = f"{server}-default-self"
        DATASET_NAME = "glove-25-angular"
        runner = BenchmarkRunner()
        runner.prepare(
            compose_file=COMPOSE_FILE,
            slice_name=SLICE_NAME,
            engine_name=ENGINE_NAME,
            dataset_name=DATASET_NAME,
            venv_path=VENV_PATH
        )
        runner.search_dataset()
        runner.wrapup()
