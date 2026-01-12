import os
import shutil
from multiprocessing import Process
from contention.Utils import Utils
from contention.Utils import run
from contention.smt import turn_off_smt, turn_on_smt

class Amplifier:
    @staticmethod
    def _detect_cpu_vendor() -> str:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("vendor_id"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
        except Exception:
            pass
        return "unknown"

    def __init__(self, result_path: str, numa_node: int, channel_num: int):
        self.result_path = result_path
        if os.path.exists(self.result_path):
            if os.path.isdir(self.result_path):
                shutil.rmtree(self.result_path)
            else:
                os.remove(self.result_path)
        os.makedirs(self.result_path, exist_ok=True)
        self.utils = Utils()
        self.cmd_for_build = f"gcc contention//user_amplifier.c -o contention//user_amplifier -lnuma -fopenmp -DNUMA_NODE={numa_node} -DCHANNEL_NUM={channel_num}"
        # Cache partitioning:
        # - Intel: pqos(CAT)
        # - AMD:  resctrl
        cache_mode = os.environ.get("CACHE_PARTITIONING", "auto").strip().lower()
        vendor = self._detect_cpu_vendor()
        if cache_mode in ("none", "off", "disable", "disabled"):
            self.cmd_for_cache_partitioning = None
        elif cache_mode in ("intel", "pqos"):
            self.cmd_for_cache_partitioning = (
                "pqos -R && "
                "pqos -e 'llc@0:1=0x7ff0;llc@0:2=0x0000f;' && "
                "pqos -a 'llc:1=10-19;llc:2=0-9;' && "
                "pqos -s"
            )
        elif cache_mode in ("amd", "resctrl"):
            self.cmd_for_cache_partitioning = "python3 contention/amd_cat.py --auto-cpus"
        else:
            if vendor == "AuthenticAMD":
                self.cmd_for_cache_partitioning = "python3 contention/amd_cat.py --auto-cpus"
            elif vendor == "GenuineIntel":
                self.cmd_for_cache_partitioning = (
                    "pqos -R && "
                    "pqos -e 'llc@0:1=0x7ff0;llc@0:2=0x0000f;' && "
                    "pqos -a 'llc:1=5-9;llc:2=0-4;' && "
                    "pqos -s"
                )
            else:
                self.cmd_for_cache_partitioning = None
        self.run_cmd = f"taskset -c 0-15 contention/user_amplifier"

    def build(self):
        if os.path.exists("contention/user_amplifier"):
            os.remove("contention//user_amplifier")
        print("Building amplifier...")
        self.build_process = Process(target=self.utils.run_proc, args=(self.cmd_for_build,))
        self.build_process.start()
        self.build_process.join()


    def start(self):
        print("Starting amplifier...")
        turn_off_smt()
        if getattr(self, "cmd_for_cache_partitioning", None):
            run(self.cmd_for_cache_partitioning, sudo=True)
        self.run_process = Process(target=self.utils.run_proc, args=(self.run_cmd,))
        self.run_process.start()
    
    def join(self):
        print("Joining amplifier...")
        self.run_process.join()
    
    def stop(self):
        #kill all process in the amplifier
        print("Stopping amplifier...")
        cmd = "pkill -x user_amplifier || true"
        run(cmd, sudo=True)
        turn_on_smt()

    def confirm_build_success(self):
        while True:
            if os.path.exists("contention/user_amplifier"):
                print("Amplifier build success")
                return True
            time.sleep(5)
        return False

        