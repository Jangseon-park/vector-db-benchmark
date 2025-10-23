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


class MemTracer{
public:
    MemTracer(int tid, pid_t pid, int sample_period);
    ~MemTracer();
    int read(std::ofstream &out);
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
    };


#endif // MEM_TRACE_H