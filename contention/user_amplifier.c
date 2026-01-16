#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <immintrin.h>
#include <stdint.h>
#include <x86intrin.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

#define THREAD_NUM 10
#define SENDER_BUFFER_SIZE (512 * 1024 * 1024) // 512 MB
#define TIME_SLOT_MS 10000ULL
#define TIME_SLOT_US (TIME_SLOT_MS * 1000ULL)
#define TIME_SLOT_NS (TIME_SLOT_US * 1000ULL)

#define TIME_DELAY_MS 200ULL
#define TIME_DELAY_US (TIME_DELAY_MS * 1000ULL)

#ifndef NUMA_NODE
#define NUMA_NODE 2
#endif
#define BLOCK_NUM (4ULL * 1024ULL * 1024ULL)
#define STRIDE_SIZE 128ULL
#ifndef CHANNEL_NUM
#define CHANNEL_NUM 1
#endif

static inline __attribute__((always_inline)) uint64_t pch_rdtscp(void)
{
    uint32_t lo, hi;
    asm volatile (
        "rdtscp\n\t"
        "lfence\n\t"
        : "=a" (lo), "=d" (hi)
        :: "%rcx"
    );
    return ((uint64_t)hi << 32) | lo;
}

static inline __attribute__((always_inline)) void pch_serialization(void)
{
    uint32_t eax = 0;
    uint32_t ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid\n\t"
        "mfence\n\t"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "0" (eax)
    );
}

static inline __attribute__((always_inline)) void pch_mfence(void)
{
    asm volatile("mfence");
}

static inline __attribute__((always_inline)) void pch_clflushopt(void *addr)
{
    asm volatile("clflushopt (%0)" :: "r"(addr));
}


/*
static inline void load_mem(uint64_t *base_addr, uint64_t block_num, uint64_t stride_size)
{
    uint64_t accessed_block_num = 0;
    uint64_t curr_pos = 0;
    uint64_t next_pos = 0;
    
    if (stride_size >= 128) {
        base_addr = (uint64_t *)((uint64_t)base_addr + 64 * CHANNEL_NUM);
    }
    uint64_t *curr_addr = base_addr;

    for (accessed_block_num = 0; accessed_block_num < block_num; accessed_block_num++) {

        __asm__ volatile(
            "vmovntdqa (%[addr]), %%ymm0\n\t"
            :
            : [addr] "b" (curr_addr)
        );

        curr_addr = (uint64_t *)((uint64_t)curr_addr + stride_size);
    }
}
*/
static inline void load_mem(uint64_t *base_addr, uint64_t block_num, uint64_t stride_size)
{
    long size_cnt = 0;
    uint8_t *curr_addr = (uint8_t *)base_addr;
    while (size_cnt < block_num * stride_size) {
      asm volatile("vmovntdqa  0x0(%0), %%zmm0\n\t"
                   "vmovntdqa  0x40(%0), %%zmm1\n\t"
                   "vmovntdqa  0x80(%0), %%zmm2\n\t"
                   "vmovntdqa  0xc0(%0), %%zmm3\n\t"
                   :
                   : "r"(curr_addr + size_cnt)
                   : "zmm0", "zmm1", "zmm2", "zmm3", "memory");
      size_cnt += 0x100;
    }
}



int main() 
{
    omp_set_num_threads(THREAD_NUM);
    uint64_t **sender_buffers = NULL;
    uint64_t *sender_buffer = NULL;
    uint64_t total_scan_num = 0, total_scan_size = 0;
    int i;

    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available on this system\n");
        return -1;
    }

    // printf("Requesting %d threads to test memory bandwidth.\n", THREAD_NUM);
    fprintf(stderr, "======================================================\n");
    fprintf(stderr, 
        "NUMA_NODE: %d\n"
        "BLOCK_NUM: %llu\n"
        "STRIDE_SIZE: %llu\n"
        "CHANNEL_NUM: %d\n"
        , NUMA_NODE, BLOCK_NUM, STRIDE_SIZE, CHANNEL_NUM);
    fprintf(stderr, "======================================================\n");
    sender_buffers = (uint64_t **)malloc(THREAD_NUM * sizeof(uint64_t *));
    if (sender_buffers == NULL) {
        fprintf(stderr, "Failed to allocate array for sender_buffers\n");
        return -1;
    }

    for (i = 0; i < THREAD_NUM; i++) {
        sender_buffers[i] = (uint64_t *)numa_alloc_onnode(SENDER_BUFFER_SIZE, NUMA_NODE);
        if (sender_buffers[i] == NULL) {
            fprintf(stderr, "Failed to allocate sender_buffer for thread %d on NUMA node %d\n", i, NUMA_NODE);
            for(int j = 0; j < i; j++) numa_free(sender_buffers[j], SENDER_BUFFER_SIZE);
            free(sender_buffers);
            return -1;
        }
        memset((void *)sender_buffers[i], 0, SENDER_BUFFER_SIZE);
    }

    total_scan_num = 0;
    // clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma omp parallel reduction(+:total_scan_num)
    {
        int tid = omp_get_thread_num();
        uint64_t *my_sender_buffer = sender_buffers[tid];
        struct timespec thread_start, thread_current;

        clock_gettime(CLOCK_MONOTONIC, &thread_start);
        while (1) {
            load_mem(my_sender_buffer, BLOCK_NUM, STRIDE_SIZE);
            // total_scan_num++;
            // clock_gettime(CLOCK_MONOTONIC, &thread_current);
            // uint64_t elasped_time_ns = (thread_current.tv_sec - thread_start.tv_sec) * 1000000000ULL + (thread_current.tv_nsec - thread_start.tv_nsec);
            // if (elasped_time_ns >= TIME_SLOT_NS) { // 10s
            //     break;
            // }
        }
    }

    total_scan_size = total_scan_num * BLOCK_NUM * STRIDE_SIZE;
    printf("Bandwidth: %.2f GB/s\n", (double)total_scan_size / (1024.0 * 1024.0 * 1024.0) /  (TIME_SLOT_MS / 1000.0));
    

    for (int i = 0; i < THREAD_NUM; i++) {
        if(sender_buffers[i]) numa_free(sender_buffers[i], SENDER_BUFFER_SIZE);
    }
    free(sender_buffers);
}