import os
import shutil
from multiprocessing import Process
from contention.Utils import Utils
from contention.Utils import run


class Prober:
    def __init__(self, result_path: str, numa_node: int, channel_num: int):

        self.result_path = result_path
        if os.path.exists(self.result_path):
            if os.path.isdir(self.result_path):
                shutil.rmtree(self.result_path)
            else:
                os.remove(self.result_path)
        os.makedirs(self.result_path, exist_ok=True)
        self.utils = Utils()
        self.cmd_for_build = f"gcc contention/user_reader.c -o contention/user_reader -lnuma -DSCAN_DELAY_S=100 -DNUMA_NODE={numa_node} -DCHANNEL_NUM={channel_num}"
        self.run_cmd = f"taskset -c 16-63 contention/user_reader > {self.result_path}"

    def reset_cmd(self, result_path: str):
        self.run_cmd = f"taskset -c 16-63 contention/user_reader > {result_path}"

    def build(self):
        if os.path.exists("contention/user_reader"):
            os.remove("contention/user_reader")
        print("Building prober...")
        self.build_process = Process(target=self.utils.run_proc, args=(self.cmd_for_build,))
        self.build_process.start()
        self.build_process.join()

    def start(self):
        print("Starting prober...")
        self.run_process = Process(target=self.utils.run_proc, args=(self.run_cmd,))
        self.run_process.start()

    def stop(self):
        print("Stopping prober...")
        cmd = "pkill -x user_reader || true"
        run(cmd, sudo=True)

    def join(self):
        print("Joining prober...")
        self.run_process.join()

    def confirm_build_success(self):
        while True:
            if os.path.exists("contention/user_reader"):
                print("Prober build success")
                return True
            time.sleep(5)
        return False