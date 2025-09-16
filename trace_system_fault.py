from bcc import BPF
import ctypes as ct
import argparse
import sys

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Trace system-wide major faults and disk I/O using BPF.",
    formatter_class=argparse.RawTextHelpFormatter
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
        print(f"Error: Could not open output file {args.output}: {e}")
        sys.exit(1)
else:
    output_file = sys.stdout


# Let's use kprobes which are more stable across kernel versions
prog_corrected = """
#include <linux/sched.h>
#include <linux/mm.h>

struct fault_data_t {
    u32 pid;
    u64 ts;
    char comm[TASK_COMM_LEN];
};

struct io_data_t {
    u32 pid;
    u64 ts;
    u64 bytes;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(fault_events);
BPF_PERF_OUTPUT(io_events);

// Kprobe on the function that handles page faults.
// The return value indicates the type of fault.
int handle_mm_fault_ret(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    // VM_FAULT_MAJOR is the flag for a major fault
    if (!(ret & VM_FAULT_MAJOR))
        return 0;

    struct fault_data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    fault_events.perf_submit(ctx, &data, sizeof(data));

    return 0;
}

TRACEPOINT_PROBE(block, block_rq_issue) {
    struct io_data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    data.bytes = args->bytes;
    io_events.perf_submit(args, &data, sizeof(data));
    return 0;
}
"""

b = BPF(text=prog_corrected)
b.attach_kretprobe(event="handle_mm_fault", fn_name="handle_mm_fault_ret")

print(f"{'TIME(s)':<12} {'COMM':<16} {'PID':<10} {'EVENT':<12} {'DETAILS'}", file=output_file)

def print_fault_event(cpu, data, size):
    event = b["fault_events"].event(data)
    print(f"{event.ts / 1e9:<12.6f} {event.comm.decode('utf-8', 'replace'):<16} {event.pid:<10} {'Major Fault':<12}", file=output_file)

def print_io_event(cpu, data, size):
    event = b["io_events"].event(data)
    print(f"{event.ts / 1e9:<12.6f} {event.comm.decode('utf-8', 'replace'):<16} {event.pid:<10} {'Disk IO':<12} Size={event.bytes}", file=output_file)

b["fault_events"].open_perf_buffer(print_fault_event)
b["io_events"].open_perf_buffer(print_io_event)

print("Tracing system-wide major faults & disk IO... Ctrl-C to stop.", file=output_file)
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        if args.output:
            print(f"\nOutput saved to {args.output}", file=sys.stdout)
            output_file.close()
        exit()
