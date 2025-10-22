from bcc import BPF, PerfType, PerfHWConfig
import ctypes as ct
import argparse
import sys
import os
from time import sleep
import psutil

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Trace memory access and instruction count for a specific process using eBPF.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    "-p", "--pid",
    type=int,
    required=True,
    help="The process ID to trace."
)
parser.add_argument(
    "-o", "--output",
    help="Redirect all output to the specified file."
)
args = parser.parse_args()

# --- Output File Handling ---
if args.output:
    try:
        output_file = open(args.output, 'w')
    except IOError as e:
        print(f"Error: Could not open output file {args.output}: {e}", file=sys.stderr)
        sys.exit(1)
else:
    output_file = sys.stdout

# --- eBPF C Program ---
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>
#include <linux/sched.h>

struct data_t {
    u64 ip;
    u64 addr;
    u64 insn_delta;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(events);
BPF_PERF_EVENT_ARRAY(insn_counters, NUM_CPUS);
BPF_HASH(last_insn_count, u32, u64);

int do_trace(struct bpf_perf_event_data *ctx) {
    u32 cpu = bpf_get_smp_processor_id();
    
    // Read instruction count from the perf event file descriptor for the current CPU
    u64 insn_count = bpf_perf_event_read_value(&insn_counters, cpu);
    if (insn_count == (u64)-1) {
        // Error reading value, likely fd is not available
        return 0;
    }

    u64 *last_count = last_insn_count.lookup(&cpu);
    u64 delta = 0;
    if (last_count != 0) {
        if (insn_count > *last_count) {
            delta = insn_count - *last_count;
        }
    }
    
    last_insn_count.update(&cpu, &insn_count);

    struct data_t data = {};
    data.ip = ctx->ip;
    data.addr = ctx->addr;
    data.insn_delta = delta;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    events.perf_submit(ctx, &data, sizeof(data));

    return 0;
}
"""

# --- Constants for perf_event_open syscall ---
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403

if hasattr(os, 'syscall'):
    __NR_perf_event_open = 298  # Default for x86_64, will be looked up
    if 'perf_event_open' in os.syscall_names:
        __NR_perf_event_open = os.syscall_names['perf_event_open']

class PerfEventAttr(ct.Structure):
    _fields_ = [
        ('type', ct.c_uint),
        ('size', ct.c_uint),
        ('config', ct.c_ulong),
        ('sample_period', ct.c_ulong),
        ('sample_type', ct.c_ulong),
        ('read_format', ct.c_ulong),
        ('flags', ct.c_ulong),
        ('wakeup_events', ct.c_uint),
        ('bp_type', ct.c_uint),
        ('bp_addr', ct.c_ulong),
        ('bp_len', ct.c_ulong),
        ('sample_regs_user', ct.c_ulong),
        ('sample_stack_user', ct.c_uint),
        ('clockid', ct.c_int),
        ('sample_regs_intr', ct.c_ulong),
        ('aux_watermark', ct.c_uint),
        ('sample_max_stack', ct.c_ushort),
        ('aux_sample_size', ct.c_uint),
    ]

def perf_event_open(pid, cpu, event_attr):
    if not hasattr(os, 'syscall'):
        raise OSError("os.syscall is not available on this system")
    return os.syscall(__NR_perf_event_open, ct.byref(event_attr), pid, cpu, -1, 0)

def open_instruction_counter(pid, cpu):
    attr = PerfEventAttr()
    attr.type = PerfType.HARDWARE
    attr.size = ct.sizeof(PerfEventAttr)
    attr.config = PerfHWConfig.INSTRUCTIONS
    attr.sample_period = 0
    attr.sample_type = 0
    attr.read_format = 0
    attr.flags = 1 << 0  # disabled
    attr.flags |= 1 << 1  # inherit
    
    fd = perf_event_open(pid, cpu, attr)
    if fd < 0:
        # The EPERM error may happen in virtualized environments
        errno = ct.get_errno()
        raise OSError(errno, f"perf_event_open failed for PID {pid} on CPU {cpu}: {os.strerror(errno)}")
    return fd

# --- Main Program Logic ---
try:
    num_cpus = psutil.cpu_count()
except Exception as e:
    print(f"Could not get CPU count: {e}", file=sys.stderr)
    sys.exit(1)

bpf_text = bpf_text.replace('NUM_CPUS', str(num_cpus))

b = BPF(text=bpf_text)

# Setup instruction counters
insn_counters = b.get_table("insn_counters")
insn_fds = []
try:
    for cpu in range(num_cpus):
        fd = open_instruction_counter(args.pid, cpu)
        insn_fds.append(fd)
        insn_counters[cpu] = ct.c_int(fd)
except OSError as e:
    print(f"Error setting up instruction counters: {e}", file=sys.stderr)
    print("Please check your kernel permissions for perf_event_open. Running as root might be required.", file=sys.stderr)
    sys.exit(1)

# Enable instruction counters using ioctl
import fcntl
for fd in insn_fds:
    fcntl.ioctl(fd, PERF_EVENT_IOC_RESET, 0)
    fcntl.ioctl(fd, PERF_EVENT_IOC_ENABLE, 0)

# Attach eBPF program to memory load events
# This traces L1 data cache read accesses, as a proxy for memory loads.
# To trace stores, you can change CACHE_OP_READ to CACHE_OP_WRITE.
event_config = (PerfHWConfig.CACHE_L1D | 
                (PerfHWConfig.CACHE_OP_READ << 8) | 
                (PerfHWConfig.CACHE_RESULT_ACCESS << 16))

try:
    b.attach_perf_event(
        ev_type=PerfType.HW_CACHE,
        ev_config=event_config,
        fn_name="do_trace",
        pid=args.pid,
        sample_period=100
    )
except Exception as e:
    print(f"Failed to attach perf event: {e}", file=sys.stderr)
    sys.exit(1)


print(f"Tracing memory loads for PID {args.pid}... Ctrl-C to stop.", file=output_file)
print(f"{'COMM':<16} {'IP':<18} {'ADDR':<18} {'INSN_DELTA'}", file=output_file)

def print_event(cpu, data, size):
    event = b["events"].event(data)
    print(f"{event.comm.decode('utf-8', 'replace'):<16} "
          f"{hex(event.ip):<18} "
          f"{hex(event.addr):<18} "
          f"{event.insn_delta}", file=output_file)

b["events"].open_perf_buffer(print_event)

while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        if args.output:
            print(f"\nOutput saved to {args.output}", file=sys.stdout)
            output_file.close()
        # Cleanup
        for fd in insn_fds:
            fcntl.ioctl(fd, PERF_EVENT_IOC_DISABLE, 0)
            os.close(fd)
        sys.exit(0)
