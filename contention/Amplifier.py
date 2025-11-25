import os
import shutil
from multiprocessing import Process
from contention.Utils import Utils
from contention.Utils import run


class Amplifier:
    def __init__(self, result_path: str, numa_node: int, channel_num: int):
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.utils = Utils()
        self.cmd_for_build = f"gcc contention/user_amplifier.c -o contention/user_amplifier -lnuma -fopenmp -DNUMA_NODE={numa_node} -DCHANNEL_NUM={channel_num}"
        self.cmd_for_pqos = f"sudo pqos -R && sudo pqos -e 'llc@0:1=0xffff0;llc@0:2=0x0000f;' && sudo pqos -a 'llc:1=16-31;llc:2=0-15;' && sudo pqos -s"
        self.run_cmd = f"taskset -c 0-15 contention/user_amplifier"

    def build(self):
        if os.path.exists("contention/user_amplifier"):
            os.remove("contention/user_amplifier")
        print("Building amplifier...")
        self.build_process = Process(target=self.utils.run_proc, args=(self.cmd_for_build,))
        self.build_process.start()
        self.build_process.join()


    def start(self):
        print("Starting amplifier...")
        run(self.cmd_for_pqos, sudo=True)
        self.run_process = Process(target=self.utils.run_proc, args=(self.run_cmd,))
        self.run_process.start()
    
    def join(self):
        print("Joining amplifier...")
        self.run_process.join()
    
    def stop(self):
        #kill all process in the amplifier
        print("Stopping amplifier...")
        cmd = "sudo pkill -f user_amplifier"
        run(cmd, sudo=True)

    def confirm_build_success(self):
        while True:
            if os.path.exists("contention/user_amplifier"):
                print("Amplifier build success")
                return True
            time.sleep(5)
        return False

        