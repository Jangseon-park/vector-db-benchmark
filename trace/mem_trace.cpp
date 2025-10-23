#include "mem_trace.h"
#include <cstdio>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <cstring>
#include <memory.h>
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

long perf_event_open(struct perf_event_attr *event_attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, event_attr, pid, cpu, group_fd, flags);
}

MemTracer::MemTracer(int tid, pid_t pid, int sample_period)
    : tid(tid), pid(pid), sample_period(sample_period), seq(0), rdlen(0), mplen(0), mp(nullptr)
{
    perf_event_attr pe = {};
    memset(&pe, 0, sizeof(struct perf_event_attr));
    //pe.type = PERF_TYPE_RAW;
    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(struct perf_event_attr);
    //pe.config = (0x01 << 8) | 0x34; // UMask: 0x01, EventCode: 0x34
    pe.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    pe.sample_period = static_cast<uint64_t>(sample_period);
    pe.sample_type = PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_ADDR | PERF_SAMPLE_READ | PERF_SAMPLE_PHYS_ADDR;
    pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED;
    pe.disabled = 1; // Event is initially disabled
    //pe.exclude_user = 1;
    pe.precise_ip = 2; // 0: skid, 1: constant skid, 2: try to be precise
    //pe.config1 = 0x00001fc1; // UMaskExt for UNC_CHA_LLC_LOOKUP.DATA_READ_MISS
    int cpu = 0; // measure on any cpu
    int group_fd = -1;
    unsigned long flags = 0;

    this->fd = perf_event_open(&pe, this->pid, -1, group_fd, flags);
    if (this->fd == -1) {
        std::cerr << "perf_event_open failed" << std::endl;
        perror("perf_event_open");
        throw;
    }

    this->mplen = MMAP_SIZE;
    this->mp = (perf_event_mmap_page *)mmap(nullptr, MMAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);

    if (this->mp == MAP_FAILED) {
        std::cout << "mp:" << this->mp << std::endl;
        std::cerr << "mmap failed" << std::endl;
        perror("mmap");
        throw;
    }

    if (this->start() < 0) {
        std::cerr << "start failed" << std::endl;
        perror("start");
        throw;
    }
}


int MemTracer::start() {
    if (this->fd < 0) {
        return 0;
    }
    std::cout << "Starting MemTracer" << std::endl;
    if (ioctl(this->fd, PERF_EVENT_IOC_RESET, 0) < 0) {
        perror("ioctl");
        return -1;
    }

    if (ioctl(this->fd, PERF_EVENT_IOC_ENABLE, 0) < 0) {
        perror("ioctl");
        return -1;
    }
    return 0;
}

int MemTracer::stop() {
    std::cout << "Stopping MemTracer" << std::endl;
    if (this->fd < 0) {
        return 0;
    }
    if (ioctl(this->fd, PERF_EVENT_IOC_DISABLE, 0) < 0) {
        perror("ioctl");
        return -1;
    }
    return 0;
}

MemTracer::~MemTracer() {
    if (this->stop() < 0) {
        std::cerr << "failed to stop MemTracer" << std::endl;
    }

    if (this->fd < 0) {
        return;
    }

    if (this->mp != MAP_FAILED) {
        munmap(this->mp, this->mplen);
        this->mp = static_cast<perf_event_mmap_page *>(MAP_FAILED);
        this->mplen = 0;
    }

    if (this->fd != -1) {
        close(this->fd);
        this->fd = -1;
    }

    this->pid = -1;
}


int MemTracer::read(std::ofstream &out) {
    if (this->fd < 0) {
        return 0;
    }

    if (this->mp == MAP_FAILED)
        return -1;

    int r = 0;
    perf_event_header *header;
    perf_sample *data;
    uint64_t last_head;
    char *dp = (char *)this->mp + PAGE_SIZE;

    std::cout << "reading mem_trace" << std::endl;
    do {
        std::cout << "reading mem_trace" << std::endl;
        this->seq = this->mp->lock; // explicit copy
        barrier();
        last_head = this->mp->data_head;
        std::cout << "last_head:" << last_head << std::endl;
        std::cout << "rdlen:" << this->rdlen << std::endl;
        while (this->rdlen < last_head) {
            header = reinterpret_cast<perf_event_header *>(dp + this->rdlen % DATA_SIZE);

            switch (header->type) {
            case PERF_RECORD_LOST:
                std::cout << "received PERF_RECORD_LOST" << std::endl;
                break;
            case PERF_RECORD_SAMPLE:
                data = (struct perf_sample *)(dp + this->rdlen % DATA_SIZE);

                if (header->size < sizeof(*data)) {
                    std::cerr << "size too small. size:" << header->size << std::endl;
                    r = -1;
                    continue;
                }
                std::cout << "received PERF_RECORD_SAMPLE" << std::endl;
                if (static_cast<uint32_t>(this->pid) == data->pid) {
                    out << "pid:" << data->pid << " tid:" << data->tid << " time:" << data->time_enabled << " addr:" << data->addr << " phys_addr:" << data->phys_addr << " llc_miss:" << data->value << " timestamp:" << data->timestamp << std::endl;
                    std::cout << "pid:" << data->pid << " tid:" << data->tid << " time:" << data->time_enabled << " addr:" << data->addr << " phys_addr:" << data->phys_addr << " llc_miss:" << data->value << " timestamp:" << data->timestamp << std::endl;
                } else {
                    std::cout << "pid mismatch. expected:" << this->pid << " actual:" << data->pid << std::endl;
                }
                break;
            case PERF_RECORD_THROTTLE:
                std::cerr << "received PERF_RECORD_THROTTLE" << std::endl;
                break;
            case PERF_RECORD_UNTHROTTLE:
                std::cerr << "received PERF_RECORD_UNTHROTTLE" << std::endl;
                break;
            case PERF_RECORD_LOST_SAMPLES:
                std::cerr << "received PERF_RECORD_LOST_SAMPLES" << std::endl;
                break;
            default:
                std::cerr << "other data received. type:" << header->type << std::endl;
                break;
            }

            this->rdlen += header->size;
        }

        mp->data_tail = last_head;
        barrier();
    } while (mp->lock != this->seq);

    return r;
}