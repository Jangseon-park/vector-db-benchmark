#ifndef MEM_TRACE_H
#define MEM_TRACE_H

#include <cstdint>
#include <filesystem>
#include <ranges>
#include <vector>
#include <x86intrin.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <linux/perf_event.h>
#include <fstream>

#define PAGE_SIZE 4096
#define DATA_SIZE PAGE_SIZE
#define MMAP_SIZE (PAGE_SIZE + DATA_SIZE)
#define barrier() _mm_mfence()

struct perf_sample {
    perf_event_header header;
    uint32_t pid;
    uint32_t tid;
    uint64_t timestamp;
    uint64_t addr;
    uint64_t value;
    uint64_t time_enabled;
    uint64_t phys_addr;
};

class MemTracer{
public:
    MemTracer(int tid, pid_t pid, int sample_period, const std::string outdir, const std::string& outfile);
    ~MemTracer();
    int read();
    int start();
    int stop();

private:
    int fd;
    int tid;
    pid_t pid;
    int sample_period;
    uint32_t seq{};
    size_t rdlen{};
    size_t mplen{};
    perf_event_mmap_page *mp;
    std::ofstream output;
    std::vector<perf_sample> samples;
    };


#endif // MEM_TRACE_H