#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <immintrin.h>
#include <stdint.h>
#include <x86intrin.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>


#define RDRAND_MAX_RETRY 32
#define MAX_SAMPLES 500000ULL
#define RECEIVER_BUFFER_SIZE (512 * 1024 * 1024) // 512 MB
#define TIME_SLOT_US 1200ULL
#define TIME_SLOT_NS (TIME_SLOT_US * 1000ULL)
#ifndef SCAN_DELAY_S
#define SCAN_DELAY_S 300ULL
#endif

#ifndef NUMA_NODE
#define NUMA_NODE 2
#endif

#define BLOCK_NUM 4096
#define STRIDE_SIZE (128 * 1024)
#ifndef CHANNEL_NUM
#define CHANNEL_NUM 1
#endif

/* Path of optional control file; if this file exists the receiver will stop. */
#ifndef STOP_FILE_PATH
#define STOP_FILE_PATH "/tmp/user_reader.stop"
#endif

static volatile sig_atomic_t stop_requested = 0;

static void handle_signal(int sig)
{
    /* set flag to break out of main loop */
    (void)sig;
    stop_requested = 1;
}

static inline int check_stop_file(void)
{
    /* return 1 if control file exists */
    return access(STOP_FILE_PATH, F_OK) == 0;
}

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

static inline __attribute__((always_inline)) void pch_mfence(void)
{
    asm volatile("mfence");
}

static inline __attribute__((always_inline)) void pch_clflush(void *addr)
{
    asm volatile("clflush (%0)" :: "r"(addr));
}

static inline int get_rand(uint64_t *rd, uint64_t range)
{
    uint8_t ok;
    int i = 0;
    for (i = 0; i < RDRAND_MAX_RETRY; i++) {
        asm volatile(
            "rdrand %0\n\t"
            "setc   %1\n\t"
            : "=r"(*rd), "=qm"(ok)
        );
        if (ok) {
            *rd = *rd % range;
            return 0;
        }
    }

    return 1;
}

static int init_chasing_index(uint64_t *cindex, uint64_t csize, uint64_t access_order)
{
    uint64_t curr_pos = 0;
    uint64_t next_pos = 0;
    uint64_t i = 0;
    int ret = 0;

    // pr_info("%s: csize: %llu\n", __func__, csize);

    if (access_order == 0) {
        for (i = 0; i < csize - 1; i++) {
            do {
                ret = get_rand(&next_pos, csize);
                if (ret != 0)
                    return 1;
            } while ((cindex[next_pos] != 0) || (next_pos == curr_pos));
            cindex[curr_pos] = next_pos;
            // printf("cindex[%lu]\t= %lu\n", curr_pos, cindex[curr_pos]);
            curr_pos = next_pos;
        }
        // pr_info("%s: generating random cindex...\n", __func__);
    }
    else {
        for (i = 0; i < csize - 1; i++) {
            next_pos = curr_pos + 1;
            cindex[curr_pos] = next_pos;
            // printf("cindex[%lu]\t= %lu\n", curr_pos, cindex[curr_pos]);
            curr_pos++;
        }
        // pr_info("%s: generating sequential cindex...\n", __func__);
    }
    return 0;
}

static uint64_t load_mem(   uint64_t *base_addr,
                            uint64_t block_num,
                            uint64_t stride_size,
                            uint64_t *cindex_buffer
                        )
{
    uint64_t sum_latency = 0;
    uint64_t accessed_block_num = 0;
    uint64_t curr_pos = 0;
    uint64_t next_pos = 0;
    uint64_t start = 0, end = 0;

    if (stride_size >= 128) {
        base_addr = (uint64_t *)((uint64_t)base_addr + 64 * CHANNEL_NUM);
    }
    uint64_t *curr_addr = base_addr;

    for (accessed_block_num = 0; accessed_block_num < block_num; accessed_block_num++) {
        curr_addr = (uint64_t *)((uint64_t)base_addr + curr_pos * stride_size);

        pch_mfence();
        start = pch_rdtscp();

        __asm__ volatile(
            "vmovntdqa (%[addr]), %%ymm0\n\t"
            :
            : [addr] "b" (curr_addr)
        );

        pch_mfence();
        end = pch_rdtscp();
        sum_latency += end - start;
        pch_clflush(curr_addr);
        // printf("*curr_addr\t= %lu\n", *curr_addr);
        next_pos = cindex_buffer[curr_pos];
        curr_pos = next_pos;
        // curr_addr = (uint64_t *)((uint64_t)curr_addr + stride_size);
    }

    return (sum_latency / block_num);
}

int main() {
    uint64_t *receiver_buffer = NULL;
    uint64_t *cindex_buffer = NULL;
    uint64_t *time_log = NULL;
    uint16_t *latency_log = NULL;
    uint64_t sum_latency = 0, avg_latency = 0;
    struct timespec start, current;
    uint64_t elasped_time_ns = 0;
    uint64_t sample_count = 0;
    struct timespec sleep_time = {0, TIME_SLOT_NS};
    int i;

    uint64_t start_cycle, end_cycle, cycle_diff;
    struct timespec start_time, end_time;
    uint64_t ns_diff;
    double ns_per_cycle;

    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available on this system\n");
        return -1;
    }
    
    receiver_buffer = (uint64_t *)numa_alloc_onnode(RECEIVER_BUFFER_SIZE, NUMA_NODE);
    if (receiver_buffer == NULL) {
        fprintf(stderr, "Failed to allocate receiver_buffer on NUMA node %d\n", NUMA_NODE);
        return -1;
    }
    memset((void *)receiver_buffer, 0, RECEIVER_BUFFER_SIZE);

    cindex_buffer = (uint64_t *)malloc(BLOCK_NUM * sizeof(uint64_t));
    if (cindex_buffer == NULL) {
        fprintf(stderr, "Failed to allocate cindex_buffer\n");
        free(receiver_buffer);
        return -1;
    }
    memset((void *)cindex_buffer, 0, BLOCK_NUM * sizeof(uint64_t));

    time_log = (uint64_t *)malloc(MAX_SAMPLES * sizeof(uint64_t));
    if (time_log == NULL) {
        fprintf(stderr, "Failed to allocate time_log\n");
        free(receiver_buffer);
        return -1;
    }
    memset((void *)time_log, 0, MAX_SAMPLES * sizeof(uint64_t));

    latency_log = (uint16_t *)malloc(MAX_SAMPLES * sizeof(uint16_t));
    if (latency_log == NULL) {
        fprintf(stderr, "Failed to allocate latency_log\n");
        free(receiver_buffer);
        free(time_log);
        return -1;
    }
    memset((void *)latency_log, 0, MAX_SAMPLES * sizeof(uint16_t));
    fprintf(stderr, "======================================================\n");
    fprintf(stderr, 
        "NUMA_NODE: %d\n"
        "BLOCK_NUM: %d\n"
        "STRIDE_SIZE: %d\n"
        "CHANNEL_NUM: %d\n"
        "SCAN_DELAY_S: %llu\n"
        , NUMA_NODE, BLOCK_NUM, STRIDE_SIZE, CHANNEL_NUM, SCAN_DELAY_S);
    fprintf(stderr, "======================================================\n");

    init_chasing_index(cindex_buffer, BLOCK_NUM, 0);

    start_cycle = pch_rdtscp();
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Warm-up
    for (i = 0; i < 10; i++) {
        load_mem(receiver_buffer, BLOCK_NUM, STRIDE_SIZE, cindex_buffer);
    }

    end_cycle = pch_rdtscp();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    cycle_diff = end_cycle - start_cycle;
    ns_diff = (end_time.tv_sec - start_time.tv_sec) * 1000000000ULL + 
              (end_time.tv_nsec - start_time.tv_nsec);
    
    ns_per_cycle = (double)ns_diff / (double)cycle_diff;

    struct timespec warmup_start, warmup_end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    /* register signal handlers so user can terminate with Ctrl-C or kill */
    struct sigaction sa;
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    while (1) {
        // clock_gettime(CLOCK_MONOTONIC, &warmup_start);

        avg_latency = load_mem(receiver_buffer, BLOCK_NUM, STRIDE_SIZE, cindex_buffer);
        clock_gettime(CLOCK_MONOTONIC, &current);
        elasped_time_ns = (current.tv_sec - start.tv_sec) * 1000000000ULL + (current.tv_nsec - start.tv_nsec);

        if (sample_count < MAX_SAMPLES) {
            time_log[sample_count] = elasped_time_ns;
            latency_log[sample_count] = avg_latency;
            sample_count++;
        }

        /* stop if requested by signal or control file exists */
        if (stop_requested || check_stop_file()) {
            fprintf(stderr, "stop requested: stop_requested=%d, stop_file=%s\n", stop_requested, STOP_FILE_PATH);
            break;
        }

        if (elasped_time_ns >= SCAN_DELAY_S * 1e9) {
            break;
        }

        usleep(TIME_SLOT_US);
        // nanosleep(&sleep_time, NULL);
    }
    
    printf("time_ns,latency_ns\n");
    for (i = 100; i < sample_count; i++) {
        printf("%lu,%lu\n", time_log[i], (uint64_t)((double)latency_log[i] * ns_per_cycle));
    }

    numa_free(receiver_buffer, RECEIVER_BUFFER_SIZE);
    free(time_log);
    free(latency_log);
}