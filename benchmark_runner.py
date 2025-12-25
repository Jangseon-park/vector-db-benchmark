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
from contention.Amplifier import Amplifier
from contention.Prober import Prober
from contention.Utils import Utils
from contention.Utils import run as run_sudo_cmd
from multiprocessing import Process

class BenchmarkRunner:
    """
    Encapsulates the logic for running the vector-db-benchmark with cgroup constraints.
    """
    def __init__(self, result_path, start_ch, end_ch):
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
        self.result_path = result_path
        self.start_channel_num = start_ch
        self.end_channel_num = end_ch
    def _create_commands(self):
        """Creates a dictionary of commands to be executed."""
        return {
            "docker_compose_down": f"sudo docker compose -f {self.compose_file} down --remove-orphans -v",
            "docker_compose_up": f"sudo docker compose -f {self.compose_file} up -d",
            "systemctl_stop": f"sudo systemctl stop {self.slice_name}",
            "rm_slice_path": f"sudo rm -rf {self.slice_path}",
            "systemctl_daemon_reload": "sudo systemctl daemon-reload",
            "set_property": f"sudo systemctl set-property {self.slice_name} AllowedMemoryNodes=2 AllowedCPUs=0-31",
            "upload": [f"{self.python_exec}", f"{self.run_py_script}", "--engines", f"{self.engine_name}", "--datasets", f"{self.dataset_name}", "--skip-search"],
            "search": f"{self.python_exec} {self.run_py_script} --engines {self.engine_name} --datasets {self.dataset_name} --skip-upload --drop-caches"
            #"search": [f"{self.python_exec}", f"{self.run_py_script}", "--engines", f"{self.engine_name}", "--datasets", f"{self.dataset_name}", "--skip-upload", "--drop-caches"]
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
        self.utils = Utils()
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
        run_sudo_cmd(self.commands["docker_compose_down"], sudo=True)

        print("🧹 Resetting any failed cgroup slice from previous runs...")
        run_sudo_cmd(self.commands["systemctl_stop"], sudo=True)
        run_sudo_cmd(self.commands["rm_slice_path"], sudo=True)
        run_sudo_cmd(self.commands["systemctl_daemon_reload"], sudo=True)

    def wrapup(self):
        """Cleans up all resources after the benchmark is finished."""
        print("🧹 Benchmark finished. Cleaning up...")
        try:
            run_sudo_cmd(self.commands["docker_compose_down"], sudo=True)
            run_sudo_cmd(self.commands["systemctl_stop"], sudo=True)
            run_sudo_cmd(self.commands["rm_slice_path"], sudo=True)
            run_sudo_cmd(self.commands["systemctl_daemon_reload"], sudo=True)
            print("Script completed successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n Benchmark wrapup failed: {e}", file=sys.stderr)

    def _start_engine(self):
        """Starts the database engine and waits for it to initialize."""
        print("🚀 Starting database engine using docker-compose...")
        run_sudo_cmd(self.commands["docker_compose_up"], sudo=True)
        print("⏳ Waiting for engine to initialize (2 seconds)...")
        time.sleep(2)
        print("🧹 Engine started successfully.")
    
    def _set_constraints(self):
        print("🛠️ Applying resource constraints to the cgroup slice...")
        run_sudo_cmd(self.commands["set_property"], sudo=True)
        print(f"Successfully applied AllowedMemoryNodes=2 and AllowedCPUs from node 0 to {self.slice_name}.")
        print("🧹 Constraints set successfully.")
    
    def _upload_dataset(self):
        print(f"Uploading the dataset for engine '{self.engine_name}' with dataset '{self.dataset_name}'...")
        self._run_command(self.commands["upload"])
        print("🧹 Dataset uploaded successfully.")
    
    def search_dataset(self):
        print(f"Searching the dataset for engine '{self.engine_name}' with dataset '{self.dataset_name}'...")
        #self._run_command(self.commands["search"])
        self.search_process = Process(target=self.utils.run_proc, args=(self.commands["search"],))
        self.search_process.start()
        print("Search started.")
        
    def stop_search(self):
        print("Stopping search...")
        cmd = f"sudo pkill -f {self.run_py_script}"
        run_sudo_cmd(cmd, sudo=True)
        print("Search stopped.")

    
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

    def bench(self, compose_file, slice_name, engine_name, dataset_name, venv_path, target_numa_node, timeout):
        for channel_num in range(self.start_channel_num, self.end_channel_num):
            result_path = f"{self.result_path}/{channel_num}"
            prober = Prober(result_path, target_numa_node, channel_num)
            amplifier = Amplifier(result_path, target_numa_node, channel_num)
            prober.build()
            prober.confirm_build_success()
            amplifier.build()
            amplifier.confirm_build_success()
            self.prepare(compose_file, slice_name, engine_name, dataset_name, venv_path)
            amplifier.start()
            time.sleep(10)
            for index in range(0, 100):
                print(f"run test iteration:{index}")
                result_path_tmp = f"{result_path}/{time.strftime('%Y%m%d_%H%M%S')}-{target_numa_node}-{index}.csv"
                prober.reset_cmd(result_path_tmp)
                prober.start()
                self.search_dataset()
                time.sleep(timeout)
                prober.stop()
                self.stop_search()
                if index % 10 == 0:
                    utils = Utils()
                    df = utils.parse_log_file(result_path_tmp)
                    utils.plot_data_from_df(df, result_path_tmp.replace(".csv", ".pdf"))
                    print(f"Plot saved to {result_path_tmp.replace('.csv', '.pdf')}") 
            self.wrapup()


if __name__ == "__main__":
    # --- Configuration ---
    #server_list = ["qdrant", "weaviate", "pgvector", "milvus"]
    server_list = ["milvus"] # all servers
    target_numa = 2
    start_ch = 0
    end_ch = 1
    timeout = 105
    
    #for server in server_list:
    #    for m in [16, 32, 64]:
    #        COMPOSE_FILE = f"engine/servers/{server}-single-node/docker-compose.yaml"
    #        SLICE_NAME = "ex.slice"
    #        VENV_PATH = "/home/wolf/.cache/pypoetry/virtualenvs/vector-db-benchmark-3zx8bqwV-py3.11"
    #        ENGINE_NAME = f"{server}-default-self-m-{m}"
    #        DATASET_NAME = "glove-25-angular"
    #        result_path = f"/home/wolf/workspace/cxl-contention-llm/vector-db-benchmark/contention-results/search-linknum/{server}-m-{m}"
    #        runner = BenchmarkRunner(result_path, start_ch, end_ch)
    #        runner.bench(
    #            compose_file=COMPOSE_FILE,
    #            slice_name=SLICE_NAME,
    #            engine_name=ENGINE_NAME,
    #            dataset_name=DATASET_NAME,
    #            venv_path=VENV_PATH,
    #            target_numa_node = target_numa,
    #            timeout = timeout
    #        )
    for server in server_list:
        for index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_PQ", "DISKANN", "HNSW"]:
            COMPOSE_FILE = f"engine/servers/{server}-single-node/docker-compose.yaml"
            SLICE_NAME = "ex.slice"
            VENV_PATH = "/home/wolf/.cache/pypoetry/virtualenvs/vector-db-benchmark-3zx8bqwV-py3.11"
            ENGINE_NAME = f"{server}-default-self-{index_type}"
            DATASET_NAME = "glove-25-angular"
            result_path = f"/home/wolf/workspace/cxl-contention-llm/vector-db-benchmark/contention-results/search-index/{server}-{index_type}"
            runner = BenchmarkRunner(result_path, start_ch, end_ch)
            runner.bench(
                compose_file=COMPOSE_FILE,
                slice_name=SLICE_NAME,
                engine_name=ENGINE_NAME,
                dataset_name=DATASET_NAME,
                venv_path=VENV_PATH,
                target_numa_node = target_numa,
                timeout = timeout
            )
       