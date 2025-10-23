#include "mem_trace.h"
#include <iostream>
#include <fstream>
#include <unistd.h> // for getpid() and sleep()
#include <vector>
#include <stdexcept>

int main() {
    pid_t pid = getpid();
    // In this example, we trace the current process.
    // The 'tid' parameter to MemTracer is not used for opening the perf event,
    // so we can just pass 'pid'. The perf event will be for the whole process.
    int tid = pid; 
    int sample_period = 1000;
    std::string outdir = std::filesystem::current_path().string();
    std::string outfile = "mem_trace.txt";
    try {
        std::cout << "Starting memory tracer for current process (PID: " << pid << ")" << std::endl;

        MemTracer tracer(tid, pid, sample_period, outdir, outfile);

        std::cout << "Tracer started. Generating memory traffic for 10 seconds..." << std::endl;
        std::vector<int> memory_hog;
        for (int i = 0; i < 10; ++i) {
            // Allocate some memory to generate L3 misses
            for(int j=0; j<10000; ++j) {
                memory_hog.push_back(j);
            }
            tracer.read();
            sleep(1);
            std::cout << "." << std::flush;
        }
        std::cout << std::endl;

        // Final read to get any remaining events
        tracer.read();
        tracer.stop();
        std::cout << "Stopping tracer." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
        return 1;
    }

    std::cout << "Tracing complete. Output written to mem_trace_output.txt" << std::endl;

    return 0;
}
